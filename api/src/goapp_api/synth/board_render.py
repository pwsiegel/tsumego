"""Render a Go board (full or partial window) to an image.

Uses the conventions real Go book typesetters use: thin interior grid lines,
a ~2x thicker outer border on real board edges only (cut edges fade), hoshi
dots at the 9 standard intersections, black stones filled solid, white
stones as outlined disks. Marks (triangles / squares / circles / numbers 1–N)
are drawn inside stones in the inverted color.

Returns the rendered image plus per-stone pixel centers and which of the 4
image edges correspond to real board boundaries — everything the downstream
detector / edge classifier needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


BOARD_SIZE = 19
HOSHI = {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)}


@dataclass(frozen=True)
class Stone:
    col: int                           # board column (0..18)
    row: int                           # board row (0..18)
    color: str                         # "B" or "W"
    mark: str | None = None            # "triangle" / "square" / "circle" / "1".."99" / None


@dataclass(frozen=True)
class BoardStyle:
    pitch: int = 40                    # pixels between adjacent intersections
    margin: int = 20                   # pixels of blank space around the grid
    grid_width: int = 1                # interior grid line thickness
    border_width: int = 3              # thick border on real edges
    bg: tuple[int, int, int] = (252, 248, 236)  # paper-ish
    ink: tuple[int, int, int] = (20, 20, 20)
    white_outline: int = 2
    hoshi_radius: int = 3
    stone_radius_frac: float = 0.46    # fraction of pitch
    mark_frac: float = 0.45            # mark size vs stone radius


@dataclass
class BoardRender:
    image: Image.Image
    stones: list[tuple[int, int, int, str]]   # (x_px, y_px, radius_px, color)
    edges_on_board: dict[str, bool]           # {left,right,top,bottom}: is a real board edge
    window: tuple[int, int, int, int]         # (col_min, col_max, row_min, row_max)


def render_board(
    stones: list[Stone],
    window: tuple[int, int, int, int] = (0, BOARD_SIZE - 1, 0, BOARD_SIZE - 1),
    style: BoardStyle | None = None,
    font_path: str | None = None,
) -> BoardRender:
    """Render a window of the 19x19 board with the given stones.

    window = (col_min, col_max, row_min, row_max), inclusive on both ends.
    Stones outside the window are silently dropped.
    """
    s = style or BoardStyle()
    col_min, col_max, row_min, row_max = window
    assert 0 <= col_min <= col_max <= BOARD_SIZE - 1
    assert 0 <= row_min <= row_max <= BOARD_SIZE - 1

    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    W = s.margin * 2 + (n_cols - 1) * s.pitch
    H = s.margin * 2 + (n_rows - 1) * s.pitch

    # Render at 2x then downsample for smoother lines.
    SCALE = 2
    img = Image.new("RGB", (W * SCALE, H * SCALE), s.bg)
    draw = ImageDraw.Draw(img)

    def to_px(col: int, row: int) -> tuple[int, int]:
        x = (s.margin + (col - col_min) * s.pitch) * SCALE
        y = (s.margin + (row - row_min) * s.pitch) * SCALE
        return x, y

    edges_on_board = {
        "left": col_min == 0,
        "right": col_max == BOARD_SIZE - 1,
        "top": row_min == 0,
        "bottom": row_max == BOARD_SIZE - 1,
    }

    # ---- grid lines ----
    # Interior grid lines: draw from col_min..col_max and row_min..row_max,
    # but skip the outermost grid line on edges that aren't a real board
    # boundary (standard book convention: cut edges have no line).
    first_col = col_min if edges_on_board["left"] else col_min + 1
    last_col = col_max if edges_on_board["right"] else col_max - 1
    first_row = row_min if edges_on_board["top"] else row_min + 1
    last_row = row_max if edges_on_board["bottom"] else row_max - 1
    for col in range(first_col, last_col + 1):
        x0, y0 = to_px(col, row_min)
        x1, y1 = to_px(col, row_max)
        draw.line([(x0, y0), (x1, y1)], fill=s.ink, width=s.grid_width * SCALE)
    for row in range(first_row, last_row + 1):
        x0, y0 = to_px(col_min, row)
        x1, y1 = to_px(col_max, row)
        draw.line([(x0, y0), (x1, y1)], fill=s.ink, width=s.grid_width * SCALE)

    # ---- thick outer border on real board boundaries only ----
    def thick_line(p0: tuple[int, int], p1: tuple[int, int]) -> None:
        draw.line([p0, p1], fill=s.ink, width=s.border_width * SCALE)
    if edges_on_board["left"]:
        thick_line(to_px(col_min, row_min), to_px(col_min, row_max))
    if edges_on_board["right"]:
        thick_line(to_px(col_max, row_min), to_px(col_max, row_max))
    if edges_on_board["top"]:
        thick_line(to_px(col_min, row_min), to_px(col_max, row_min))
    if edges_on_board["bottom"]:
        thick_line(to_px(col_min, row_max), to_px(col_max, row_max))

    # ---- hoshi ----
    for (c, r) in HOSHI:
        if col_min <= c <= col_max and row_min <= r <= row_max:
            cx, cy = to_px(c, r)
            rr = s.hoshi_radius * SCALE
            draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)], fill=s.ink)

    # ---- stones ----
    stone_r = int(s.pitch * s.stone_radius_frac)
    stones_out: list[tuple[int, int, int, str]] = []
    for st in stones:
        if not (col_min <= st.col <= col_max and row_min <= st.row <= row_max):
            continue
        cx, cy = to_px(st.col, st.row)
        rr = stone_r * SCALE
        if st.color == "B":
            draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)], fill=s.ink)
        else:
            draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)],
                         fill=s.bg, outline=s.ink, width=s.white_outline * SCALE)
        if st.mark:
            _draw_mark(draw, cx, cy, rr, st.color, st.mark, s, font_path)
        # Record the stone in DOWNSAMPLED coordinates.
        stones_out.append((cx // SCALE, cy // SCALE, stone_r, st.color))

    img = img.resize((W, H), Image.LANCZOS)
    return BoardRender(
        image=img,
        stones=stones_out,
        edges_on_board=edges_on_board,
        window=window,
    )


def _draw_mark(
    draw: ImageDraw.ImageDraw,
    cx: int, cy: int, rr: int,
    color: str, mark: str, s: BoardStyle,
    font_path: str | None,
) -> None:
    """Draw a triangle/square/circle or a number centered on a stone."""
    mark_color = s.bg if color == "B" else s.ink
    size = int(rr * s.mark_frac)
    width = max(1, rr // 10)

    if mark == "triangle":
        # Equilateral, point-up, centered.
        from math import cos, radians, sin
        pts = [
            (cx + int(size * cos(radians(-90))), cy + int(size * sin(radians(-90)))),
            (cx + int(size * cos(radians(30))), cy + int(size * sin(radians(30)))),
            (cx + int(size * cos(radians(150))), cy + int(size * sin(radians(150)))),
        ]
        draw.polygon(pts, outline=mark_color, width=width)
    elif mark == "square":
        draw.rectangle(
            [(cx - size, cy - size), (cx + size, cy + size)],
            outline=mark_color, width=width,
        )
    elif mark == "circle":
        draw.ellipse(
            [(cx - size, cy - size), (cx + size, cy + size)],
            outline=mark_color, width=width,
        )
    elif mark.isdigit():
        font_size = int(rr * 1.1)
        font = _load_font(font_path, font_size)
        bbox = draw.textbbox((0, 0), mark, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
            mark, fill=mark_color, font=font,
        )


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    # Fall back to a sensible default. Helvetica ships on macOS.
    for fb in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(fb, size)
        except OSError:
            continue
    return ImageFont.load_default()


# -- convenience helpers ------------------------------------------------------


def random_stones(
    window: tuple[int, int, int, int],
    density: float = 0.3,
    mark_prob: float = 0.2,
    rng: random.Random | None = None,
) -> list[Stone]:
    """Generate a random stone configuration within `window`, covering
    roughly `density` of the visible intersections with stones. `mark_prob`
    is the chance that any given stone also carries a mark."""
    r = rng or random.Random()
    col_min, col_max, row_min, row_max = window
    cells = [(c, row) for c in range(col_min, col_max + 1)
             for row in range(row_min, row_max + 1)]
    r.shuffle(cells)
    n = int(len(cells) * density)
    picks = cells[:n]
    out: list[Stone] = []
    # Stable number counter so "numbered move" marks start at 1.
    next_num = 1
    marks = ["triangle", "square", "circle"]
    for col, row in picks:
        color = r.choice(("B", "W"))
        mark: str | None = None
        if r.random() < mark_prob:
            choice = r.choice(("mark", "number"))
            if choice == "mark":
                mark = r.choice(marks)
            else:
                mark = str(next_num)
                next_num += 1
        out.append(Stone(col=col, row=row, color=color, mark=mark))
    return out


def to_png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
