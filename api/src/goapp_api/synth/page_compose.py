"""Compose a full synthetic Go-book page from board renders + random text.

Strategy: pick a grid layout (e.g. 2x3 or 3x2 cells) and drop a board diagram
plus a caption / question-text block into each cell. Around the grid we
sprinkle header/footer filler text in the page's chosen language. The output
is a page image plus JSON annotations — one bbox per board, each with its
four `edge_on_board` flags, for use in board-detector training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from .board_render import (
    BOARD_SIZE, BoardStyle, Stone, random_stones, render_board,
)
from .text_sources import LANGUAGES, LATIN_FONT_FALLBACK, Language, make_paragraph


@dataclass(frozen=True)
class BoardAnnotation:
    bbox: tuple[int, int, int, int]       # (x0, y0, x1, y1) on page, inclusive.
                                          # This is the TIGHT bbox: its edges
                                          # lie exactly at the outermost visible
                                          # grid intersections. Stones on
                                          # boundary intersections are partially
                                          # clipped.
    loose_bbox: tuple[int, int, int, int] # Full rendered image extent (for
                                          # debugging / back-compat).
    window: tuple[int, int, int, int]     # (col_min, col_max, row_min, row_max)
    edges_on_board: dict[str, bool]       # {left,right,top,bottom}
    edge_class: int                       # 4-bit encoding of the above,
                                          # (L<<3|R<<2|T<<1|B), range 0..15.
    stone_centers: list[tuple[int, int, str]]  # (x_px_in_page, y_px_in_page, color)


@dataclass
class Page:
    image: Image.Image
    lang_code: str
    boards: list[BoardAnnotation] = field(default_factory=list)

    def to_label(self) -> dict:
        return {
            "lang": self.lang_code,
            "size": list(self.image.size),
            "boards": [
                {
                    "bbox": list(b.bbox),
                    "loose_bbox": list(b.loose_bbox),
                    "window": list(b.window),
                    "edges_on_board": b.edges_on_board,
                    "edge_class": b.edge_class,
                    "stones": [[x, y, c] for (x, y, c) in b.stone_centers],
                }
                for b in self.boards
            ],
        }


LAYOUTS = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]


def _pick_window(rng: random.Random) -> tuple[int, int, int, int]:
    """Pick a random sub-window of the 19x19 board.

    Bias toward showing real boundaries (corners and sides) since those
    dominate real Go problem books, but include full boards and middle
    cuts too.
    """
    kind = rng.choices(
        ["full", "corner", "side", "middle"],
        weights=[15, 50, 25, 10],
    )[0]
    if kind == "full":
        return (0, BOARD_SIZE - 1, 0, BOARD_SIZE - 1)
    if kind == "corner":
        size = rng.randint(6, 12)
        corner = rng.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            return (0, size - 1, 0, size - 1)
        if corner == "tr":
            return (BOARD_SIZE - size, BOARD_SIZE - 1, 0, size - 1)
        if corner == "bl":
            return (0, size - 1, BOARD_SIZE - size, BOARD_SIZE - 1)
        return (BOARD_SIZE - size, BOARD_SIZE - 1, BOARD_SIZE - size, BOARD_SIZE - 1)
    if kind == "side":
        depth = rng.randint(5, 10)
        which = rng.choice(("top", "bottom", "left", "right"))
        if which == "top":
            return (0, BOARD_SIZE - 1, 0, depth - 1)
        if which == "bottom":
            return (0, BOARD_SIZE - 1, BOARD_SIZE - depth, BOARD_SIZE - 1)
        if which == "left":
            return (0, depth - 1, 0, BOARD_SIZE - 1)
        return (BOARD_SIZE - depth, BOARD_SIZE - 1, 0, BOARD_SIZE - 1)
    # middle
    size_x = rng.randint(6, 10)
    size_y = rng.randint(6, 10)
    col_min = rng.randint(2, BOARD_SIZE - 2 - size_x)
    row_min = rng.randint(2, BOARD_SIZE - 2 - size_y)
    return (col_min, col_min + size_x - 1, row_min, row_min + size_y - 1)


def _random_style(rng: random.Random) -> BoardStyle:
    # Slight variation so detectors don't memorize a single visual fingerprint.
    pitch = rng.randint(22, 32)
    return BoardStyle(
        pitch=pitch,
        margin=rng.randint(10, 18),
        grid_width=1,
        border_width=rng.choice((2, 3, 3, 4)),
        bg=(
            rng.randint(245, 255),
            rng.randint(240, 252),
            rng.randint(225, 245),
        ),
        ink=(rng.randint(15, 40),) * 3,
        white_outline=2,
        hoshi_radius=max(2, pitch // 12),
        stone_radius_frac=rng.uniform(0.44, 0.48),
    )


def _load_font(path: str, size: int) -> ImageFont.ImageFont:
    for p in (path, LATIN_FONT_FALLBACK):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int], max_w: int, max_h: int,
    text: str, font: ImageFont.ImageFont, ink,
) -> None:
    """Word-wrap `text` into the box and draw it."""
    x0, y0 = xy
    space_w = draw.textlength(" ", font=font)
    line_h = font.size + 4

    # Split into atomic tokens: CJK chars or space-delimited words.
    def tokenize(t: str) -> list[str]:
        has_spaces = " " in t
        if has_spaces:
            return t.split()
        return list(t)

    tokens = tokenize(text)
    joiner = " " if " " in text else ""

    cur_line = ""
    y = y0
    for tok in tokens:
        trial = tok if not cur_line else (cur_line + joiner + tok)
        if draw.textlength(trial, font=font) <= max_w:
            cur_line = trial
            continue
        if cur_line:
            draw.text((x0, y), cur_line, fill=ink, font=font)
            y += line_h
            if y + line_h > y0 + max_h:
                return
        cur_line = tok
    if cur_line and y + line_h <= y0 + max_h:
        draw.text((x0, y), cur_line, fill=ink, font=font)
    # Suppress unused-variable complaint from ruff.
    del space_w


def _fit_board_in_cell(
    cell_w: int, cell_h: int, window: tuple[int, int, int, int],
    style: BoardStyle,
) -> BoardStyle:
    """Reduce pitch if necessary so the rendered board fits the cell. Keep
    other style params unchanged."""
    col_min, col_max, row_min, row_max = window
    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    # Target: render width ≤ 0.9 * cell_w, height ≤ 0.7 * cell_h (leave room
    # for caption + question text).
    max_board_w = int(cell_w * 0.9)
    max_board_h = int(cell_h * 0.65)
    pitch_for_w = (max_board_w - 2 * style.margin) // max(1, n_cols - 1)
    pitch_for_h = (max_board_h - 2 * style.margin) // max(1, n_rows - 1)
    new_pitch = min(style.pitch, max(12, pitch_for_w), max(12, pitch_for_h))
    if new_pitch == style.pitch:
        return style
    return BoardStyle(
        pitch=new_pitch,
        margin=style.margin,
        grid_width=style.grid_width,
        border_width=style.border_width,
        bg=style.bg,
        ink=style.ink,
        white_outline=style.white_outline,
        hoshi_radius=style.hoshi_radius,
        stone_radius_frac=style.stone_radius_frac,
    )


def compose_page(
    lang_code: str | None = None,
    rng: random.Random | None = None,
    page_size: tuple[int, int] = (1000, 1400),
) -> Page:
    r = rng or random.Random()
    if lang_code is None:
        lang_code = r.choice(list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_code]

    W, H = page_size
    img = Image.new("RGB", (W, H), (252, 248, 236))
    draw = ImageDraw.Draw(img)

    body_font = _load_font(lang.font_path, r.randint(14, 18))
    header_font = _load_font(lang.font_path, r.randint(22, 28))
    caption_font = _load_font(lang.font_path, r.randint(16, 20))

    # ---- page header ----
    margin = 60
    draw.text(
        (margin, margin // 2),
        make_paragraph(lang, r.randint(3, 6), r),
        fill=(20, 20, 20), font=header_font,
    )

    # ---- grid of cells, each holding a problem ----
    rows, cols = r.choice(LAYOUTS)
    grid_top = margin + header_font.size + 20
    grid_bottom = H - margin
    grid_h = grid_bottom - grid_top
    grid_w = W - 2 * margin
    cell_w = grid_w // cols
    cell_h = grid_h // rows

    # Decorative choices made once per page so the look stays coherent.
    page_style = {
        "box_boards": r.random() < 0.35,       # draw a box around each diagram
        "row_rules": r.random() < 0.4,         # horizontal rule between rows
        "col_rules": r.random() < 0.2,         # vertical rule between cols
        "outer_border": r.random() < 0.25,     # decorative page frame
        "jitter": r.random() < 0.75,           # per-cell random offset
    }
    rule_ink = (60, 60, 60)

    if page_style["outer_border"]:
        pad = r.randint(18, 34)
        draw.rectangle(
            [(margin - pad, margin - pad), (W - margin + pad, H - margin + pad)],
            outline=rule_ink, width=1,
        )

    if page_style["row_rules"] and rows > 1:
        for ri in range(1, rows):
            y = grid_top + ri * cell_h - r.randint(2, 8)
            draw.line([(margin, y), (W - margin, y)], fill=rule_ink, width=1)
    if page_style["col_rules"] and cols > 1:
        for ci in range(1, cols):
            x = margin + ci * cell_w - r.randint(2, 8)
            draw.line(
                [(x, grid_top), (x, grid_bottom)],
                fill=rule_ink, width=1,
            )

    boards: list[BoardAnnotation] = []
    problem_num = 1
    for row_i in range(rows):
        for col_i in range(cols):
            cx0 = margin + col_i * cell_w
            cy0 = grid_top + row_i * cell_h

            # Per-cell random jitter so the layout doesn't look mechanically
            # gridded. Keeps offsets small enough that content stays inside
            # its allocated cell.
            jitter_x = r.randint(-18, 18) if page_style["jitter"] else 0
            jitter_y = r.randint(-12, 12) if page_style["jitter"] else 0
            caption_x = cx0 + jitter_x
            caption_y = cy0 + jitter_y

            caption = f"{problem_num}"
            draw.text((caption_x, caption_y), caption, fill=(20, 20, 20), font=caption_font)

            # Choose a window + stones + style, scale to fit cell.
            window = _pick_window(r)
            stones = random_stones(window, density=r.uniform(0.15, 0.35),
                                   mark_prob=r.uniform(0.1, 0.3), rng=r)
            style = _fit_board_in_cell(cell_w, cell_h, window, _random_style(r))
            rb = render_board(stones, window=window, style=style)

            # Paste centered horizontally within cell, below the caption,
            # jittered independently so the board isn't pinned to the caption.
            inner_jx = r.randint(-10, 10) if page_style["jitter"] else 0
            inner_jy = r.randint(-6, 10) if page_style["jitter"] else 0
            board_x = cx0 + (cell_w - rb.image.width) // 2 + inner_jx
            board_y = caption_y + caption_font.size + 8 + inner_jy
            if board_y + rb.image.height > cy0 + cell_h - 40:
                # Skip captions if they don't fit — keep board visible.
                board_y = cy0 + 4 + inner_jy
            # Keep board inside the page.
            board_x = max(margin, min(W - margin - rb.image.width, board_x))
            board_y = max(grid_top, min(H - margin - rb.image.height, board_y))
            img.paste(rb.image, (board_x, board_y))

            if page_style["box_boards"]:
                pad = r.randint(4, 10)
                draw.rectangle(
                    [(board_x - pad, board_y - pad),
                     (board_x + rb.image.width + pad - 1,
                      board_y + rb.image.height + pad - 1)],
                    outline=rule_ink, width=1,
                )

            # Full rendered extent (used only for debugging).
            loose_bbox = (
                board_x, board_y,
                board_x + rb.image.width - 1,
                board_y + rb.image.height - 1,
            )
            # Tight bbox: from the first visible intersection to the last
            # visible intersection. Stones on boundary intersections are
            # half-clipped by this bbox, and the grid classifier learns the
            # half-stone-at-edge visual pattern as a positive-class signal.
            n_cols = window[1] - window[0] + 1
            n_rows = window[3] - window[2] + 1
            tight_x0 = board_x + style.margin
            tight_y0 = board_y + style.margin
            tight_x1 = tight_x0 + (n_cols - 1) * style.pitch
            tight_y1 = tight_y0 + (n_rows - 1) * style.pitch
            tight_bbox = (int(tight_x0), int(tight_y0), int(tight_x1), int(tight_y1))

            e = rb.edges_on_board
            edge_class = (
                (1 if e["left"] else 0) << 3
                | (1 if e["right"] else 0) << 2
                | (1 if e["top"] else 0) << 1
                | (1 if e["bottom"] else 0)
            )
            stone_centers = [
                (board_x + sx, board_y + sy, color)
                for (sx, sy, _, color) in rb.stones
            ]
            boards.append(BoardAnnotation(
                bbox=tight_bbox,
                loose_bbox=loose_bbox,
                window=window,
                edges_on_board=rb.edges_on_board,
                edge_class=edge_class,
                stone_centers=stone_centers,
            ))

            # Question / answer text under the board.
            text_y = board_y + rb.image.height + 8
            text_h = cy0 + cell_h - text_y - 10
            if text_h > body_font.size:
                question = make_paragraph(lang, r.randint(8, 18), r)
                _draw_wrapped(
                    draw, (cx0 + jitter_x, text_y), cell_w - 12, text_h,
                    question, body_font, (40, 40, 40),
                )
            problem_num += 1

    return Page(image=img, lang_code=lang_code, boards=boards)
