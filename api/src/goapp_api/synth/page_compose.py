"""Compose a full synthetic Go-book page from board renders + random text.

Strategy: pick a page kind — "problems", "chapter_opener", "cover", or
"preface". Problem / chapter pages hold a grid of boards with distractors
around them; cover and preface pages are board-free negative examples
loaded with the kinds of big headlines, decorative banners, and ornament
strips that confused earlier bbox detectors on real PDFs.

Every page emits one JSON label: the list of tight bboxes (one per board)
plus each board's edge flags and stone centers. Decorations are never
labeled — they're the distractors the detector needs to learn to ignore.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from .board_render import (
    BOARD_SIZE, BoardStyle, random_stones, render_board,
)
from .text_sources import LANGUAGES, LATIN_FONT_FALLBACK, Language, make_paragraph


BBOX_DETECTOR_PAD = 8  # pixels of paper-bg margin YOLO should learn to emit
                       # beyond the grid on every side. Keeps the outer lines
                       # comfortably inside the bbox even when YOLO is a few
                       # px off, and downstream grid_detect finds true tight
                       # bounds inside this padded region.


@dataclass(frozen=True)
class BoardAnnotation:
    bbox: tuple[int, int, int, int]       # (x0, y0, x1, y1) on page, inclusive.
                                          # This is the TIGHT bbox: its edges
                                          # lie exactly at the outermost visible
                                          # grid intersections. Stones on
                                          # boundary intersections are partially
                                          # clipped. Consumed by the synth
                                          # extractors that build stone/edge/grid
                                          # training crops.
    bbox_padded: tuple[int, int, int, int]  # bbox grown by BBOX_DETECTOR_PAD on
                                          # each side. The YOLO detector label.
    loose_bbox: tuple[int, int, int, int] # Full rendered image extent (for
                                          # debugging / back-compat).
    window: tuple[int, int, int, int]     # (col_min, col_max, row_min, row_max)
    edges_on_board: dict[str, bool]       # {left,right,top,bottom}
    edge_class: int                       # 4-bit encoding of the above,
                                          # (L<<3|R<<2|T<<1|B), range 0..15.
    stone_centers: list[tuple[int, int, str]]  # (x_px_in_page, y_px_in_page, color)
    hoshi_centers: list[tuple[int, int]]  # (x_px_in_page, y_px_in_page) — visible hoshi
    corner_centers: dict[str, tuple[int, int] | None]  # "tl"/"tr"/"bl"/"br" → pixel or None


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
                    "bbox_padded": list(b.bbox_padded),
                    "loose_bbox": list(b.loose_bbox),
                    "window": list(b.window),
                    "edges_on_board": b.edges_on_board,
                    "edge_class": b.edge_class,
                    "stones": [[x, y, c] for (x, y, c) in b.stone_centers],
                    "hoshi": [[x, y] for (x, y) in b.hoshi_centers],
                    "corners": {
                        k: (list(v) if v is not None else None)
                        for k, v in b.corner_centers.items()
                    },
                }
                for b in self.boards
            ],
        }


LAYOUTS = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

# Page-kind weights. Roughly matches what mixed Go-book PDFs look like:
# mostly problem grids, occasional chapter openers and section pages, a
# handful of pure text pages that the bbox detector must learn to reject.
PAGE_KINDS: list[tuple[str, int]] = [
    ("problems", 70),
    ("chapter_opener", 15),
    ("cover", 8),
    ("preface", 7),
]


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
    # Wide pitch range (14–70 px) so the stone detector sees stones at many
    # pixel scales across the dataset. _fit_board_in_cell() narrows this
    # further per layout, but the starting distribution must span the range
    # we expect from real book PDFs (tight cho-chikun ~15 px through full-
    # page hm2 ~100 px, downsampled at training imgsz=640).
    pitch = rng.randint(14, 70)
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
        # Vary mark size: shape marks can be small dots (0.3·r) through
        # almost-filling badges (0.6·r). Numbers can be modest (0.85·r)
        # through bold nearly-filling (1.35·r). Real books span this
        # range; fixed sizes gave the detector a narrow visual template.
        mark_frac=rng.uniform(0.30, 0.60),
        number_frac=rng.uniform(0.85, 1.35),
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
    line_h = font.size + 4

    def tokenize(t: str) -> list[str]:
        return t.split() if " " in t else list(t)

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


def _fit_board_in_cell(
    cell_w: int, cell_h: int, window: tuple[int, int, int, int],
    style: BoardStyle,
) -> BoardStyle:
    """Reduce pitch if necessary so the rendered board fits the cell."""
    col_min, col_max, row_min, row_max = window
    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
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
        mark_frac=style.mark_frac,
        number_frac=style.number_frac,
    )


# ---------------------------------------------------------------------------
# Distractor helpers. Every one of these draws onto the page but emits no
# BoardAnnotation — the detector must learn they are not boards.
# ---------------------------------------------------------------------------


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    outline=None,
    fill=None,
    width: int = 1,
) -> None:
    # PIL's rounded_rectangle handles this natively on modern versions.
    draw.rounded_rectangle(box, radius=radius, outline=outline, fill=fill, width=width)


def _draw_figure_label(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    ink,
) -> tuple[int, int, int, int]:
    """Draw a small bordered rectangle containing `text` centered at
    `center_xy`. Returns the bbox of the label. Emulates the "1 도",
    "problem N", "Fig 1" labels above each diagram in real books."""
    cx, cy = center_xy
    tw = draw.textlength(text, font=font)
    th = font.size
    pad_x, pad_y = 8, 4
    x0 = int(cx - tw / 2 - pad_x)
    y0 = int(cy - th / 2 - pad_y)
    x1 = int(cx + tw / 2 + pad_x)
    y1 = int(cy + th / 2 + pad_y)
    draw.rectangle([(x0, y0), (x1, y1)], outline=ink, fill=(255, 255, 255), width=1)
    draw.text((x0 + pad_x, y0 + pad_y - 2), text, fill=ink, font=font)
    return (x0, y0, x1, y1)


def _draw_problem_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num: int,
    font: ImageFont.ImageFont,
    ink,
) -> int:
    """Draw a filled-circle numbered badge (e.g. "❶") at `xy`. Returns the
    badge radius used so callers can lay out surrounding text."""
    radius = max(12, font.size)
    cx, cy = xy
    draw.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=ink,
    )
    label = str(num)
    tw = draw.textlength(label, font=font)
    th = font.size
    draw.text(
        (int(cx - tw / 2), int(cy - th / 2 - 1)),
        label, fill=(255, 255, 255), font=font,
    )
    return radius


def _draw_header_ornament_strip(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rng: random.Random,
    title: str,
    title_font: ImageFont.ImageFont,
    ink,
) -> None:
    """Draw a full-width strip with a rounded-rect title banner in the
    middle flanked by 2-3 alternating ●○ "stone" circles on each end.
    Mimics the chapter-header strips in hm2 — a notorious false-positive
    source because the circles look exactly like stones."""
    x0, y0, x1, y1 = box
    strip_h = y1 - y0
    radius = strip_h // 2 - 2
    banner_w = int((x1 - x0) * rng.uniform(0.45, 0.6))
    banner_x0 = (x0 + x1 - banner_w) // 2
    banner_x1 = banner_x0 + banner_w
    # Banner (rounded rect containing title).
    _draw_rounded_rect(
        draw, (banner_x0, y0, banner_x1, y1),
        radius=max(4, strip_h // 3),
        outline=ink, width=1,
    )
    tw = draw.textlength(title, font=title_font)
    draw.text(
        (banner_x0 + (banner_w - tw) / 2, y0 + (strip_h - title_font.size) / 2 - 2),
        title, fill=ink, font=title_font,
    )
    # Alternating stone circles on each side.
    n = rng.randint(2, 3)
    gap = radius * 2 + 6
    # Left side, working inward from x0 toward banner_x0.
    for i in range(n):
        cx = x0 + radius + 4 + i * gap
        if cx + radius >= banner_x0 - 6:
            break
        cy = (y0 + y1) // 2
        color_fill = ink if (i % 2 == 0) else (255, 255, 255)
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=color_fill, outline=ink, width=1,
        )
    # Right side, mirrored.
    for i in range(n):
        cx = x1 - radius - 4 - i * gap
        if cx - radius <= banner_x1 + 6:
            break
        cy = (y0 + y1) // 2
        color_fill = ink if (i % 2 == 0) else (255, 255, 255)
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=color_fill, outline=ink, width=1,
        )


def _draw_section_banner(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    ink,
    fill=(240, 240, 240),
) -> None:
    """Left-aligned rounded-rect banner with bold section number + title.
    Mimics "1. 벌림에 대하여" style chapter openers."""
    x0, y0, x1, y1 = box
    _draw_rounded_rect(
        draw, box, radius=(y1 - y0) // 4,
        outline=ink, fill=fill, width=1,
    )
    draw.text(
        (x0 + 16, y0 + ((y1 - y0) - font.size) // 2 - 2),
        text, fill=ink, font=font,
    )


def _draw_date_stamp_table(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lang: Language,
    rng: random.Random,
    ink,
) -> tuple[int, int, int, int]:
    """Small 2-column × 3-row labeled grid in the upper-right, like the
    공부한 날 / 월 / 일 / 검인 boxes on hm2 exercise pages. Looks like a
    tiny board to a naive detector."""
    col_w = rng.randint(42, 56)
    row_h = rng.randint(18, 24)
    x0, y0 = xy
    cols = 2
    rows = 3
    font = _load_font(lang.font_path, row_h - 8)
    headers = lang.sample(rows, rng)
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + c * col_w
            cy0 = y0 + r * row_h
            draw.rectangle(
                [(cx0, cy0), (cx0 + col_w, cy0 + row_h)],
                outline=ink, width=1,
            )
            if c == 0:
                label = headers[r][:3] if r < len(headers) else ""
                draw.text((cx0 + 3, cy0 + 2), label, fill=ink, font=font)
    return (x0, y0, x0 + cols * col_w, y0 + rows * row_h)


def _draw_footer_ornament(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    page_num: int,
    font: ImageFont.ImageFont,
    ink,
) -> None:
    """'···▼···' ornament above a centered page number."""
    cx, cy = xy
    ornament = "· · · ▼ · · ·"
    tw = draw.textlength(ornament, font=font)
    draw.text((int(cx - tw / 2), cy), ornament, fill=ink, font=font)
    num_str = str(page_num)
    nw = draw.textlength(num_str, font=font)
    draw.text(
        (int(cx - nw / 2), cy + font.size + 4),
        num_str, fill=ink, font=font,
    )


def _draw_attribution_block(
    draw: ImageDraw.ImageDraw,
    xy_right: tuple[int, int],
    lang: Language,
    rng: random.Random,
    font: ImageFont.ImageFont,
    ink,
) -> None:
    """Right-aligned 2-line block: '— Some Name' / 'Month Year'."""
    rx, ry = xy_right
    line1 = "— " + " ".join(lang.sample(rng.randint(1, 3), rng))
    line2 = " ".join(lang.sample(2, rng))
    w1 = draw.textlength(line1, font=font)
    w2 = draw.textlength(line2, font=font)
    draw.text((int(rx - w1), ry), line1, fill=ink, font=font)
    draw.text((int(rx - w2), ry + font.size + 4), line2, fill=ink, font=font)


def _draw_display_title(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int], max_w: int,
    lang: Language, rng: random.Random, ink,
) -> int:
    """Huge-font multi-line title. Returns the y offset after the title."""
    size = rng.randint(56, 92)
    font = _load_font(lang.font_path, size)
    n_lines = rng.randint(2, 4)
    x0, y = xy
    for _ in range(n_lines):
        # Aim for a word-group whose width is within max_w.
        for _try in range(6):
            words = lang.sample(rng.randint(1, 3), rng)
            text = (" " if lang.code in ("ko", "en", "fr", "de", "es") else "").join(words)
            if draw.textlength(text, font=font) <= max_w:
                break
        draw.text((x0, y), text, fill=ink, font=font)
        y += size + 10
    return y


# ---------------------------------------------------------------------------
# Page-kind renderers. Each returns a list of BoardAnnotations (possibly
# empty) after drawing on `img`/`draw` in place.
# ---------------------------------------------------------------------------


def _render_cover(
    img: Image.Image, draw: ImageDraw.ImageDraw,
    lang: Language, rng: random.Random,
) -> list[BoardAnnotation]:
    """No boards; just big display text and optional subtitle banner."""
    W, H = img.size
    margin = 80
    # Leave plenty of whitespace above the title, as real covers do.
    y = margin + rng.randint(120, 260)
    ink = (rng.randint(10, 40),) * 3
    y = _draw_display_title(draw, (margin, y), W - 2 * margin, lang, rng, ink)
    # Optional subtitle banner (rounded-rect with smaller heading inside).
    if rng.random() < 0.7:
        sub_size = rng.randint(28, 40)
        sub_font = _load_font(lang.font_path, sub_size)
        subtitle = " ".join(lang.sample(rng.randint(2, 5), rng))
        tw = draw.textlength(subtitle, font=sub_font)
        pad_x, pad_y = 24, 12
        x0 = margin
        y0 = y + 40
        x1 = int(x0 + tw + 2 * pad_x)
        y1 = int(y0 + sub_size + 2 * pad_y)
        _draw_rounded_rect(draw, (x0, y0, x1, y1), radius=14, outline=ink, width=2)
        draw.text((x0 + pad_x, y0 + pad_y - 2), subtitle, fill=ink, font=sub_font)
    # Occasional small decorative stone pair near the bottom (looks like a
    # cover ornament, not a board) — another false-positive magnet to train
    # against.
    if rng.random() < 0.4:
        r_circle = rng.randint(20, 32)
        cx = W // 2
        cy = H - margin - rng.randint(40, 120)
        draw.ellipse(
            [(cx - 2 * r_circle - 4, cy - r_circle), (cx - 4, cy + r_circle)],
            fill=ink,
        )
        draw.ellipse(
            [(cx + 4, cy - r_circle), (cx + 2 * r_circle + 4, cy + r_circle)],
            fill=(255, 255, 255), outline=ink, width=2,
        )
    return []


def _render_preface(
    img: Image.Image, draw: ImageDraw.ImageDraw,
    lang: Language, rng: random.Random,
) -> list[BoardAnnotation]:
    """No boards; multi-section text page with varied heading sizes and
    an optional right-aligned attribution block."""
    W, H = img.size
    margin = 70
    body_font = _load_font(lang.font_path, rng.randint(15, 19))
    ink = (rng.randint(10, 40),) * 3
    y = margin
    n_sections = rng.randint(2, 4)
    for _ in range(n_sections):
        if y > H - margin - 180:
            break
        heading_size = rng.randint(26, 40)
        heading = _load_font(lang.font_path, heading_size)
        heading_text = " ".join(lang.sample(rng.randint(1, 3), rng))
        draw.text((margin, y), heading_text, fill=ink, font=heading)
        y += heading_size + 14
        # Body paragraph.
        para = make_paragraph(lang, rng.randint(40, 90), rng)
        para_box_h = rng.randint(110, 180)
        _draw_wrapped(
            draw, (margin, y), W - 2 * margin, para_box_h,
            para, body_font, ink,
        )
        y += para_box_h + 20
        # Attribution block after occasional section.
        if rng.random() < 0.35 and y < H - margin - 80:
            _draw_attribution_block(
                draw, (W - margin, y),
                lang, rng, body_font, ink,
            )
            y += body_font.size * 2 + 24
    return []


def _draw_page_header_text(
    draw: ImageDraw.ImageDraw,
    W: int, margin: int, lang: Language, rng: random.Random,
    font: ImageFont.ImageFont, ink,
) -> None:
    draw.text(
        (margin, margin // 2),
        make_paragraph(lang, rng.randint(3, 6), rng),
        fill=ink, font=font,
    )


def _render_problems(
    img: Image.Image, draw: ImageDraw.ImageDraw,
    lang: Language, rng: random.Random,
    chapter_opener: bool = False,
) -> list[BoardAnnotation]:
    """Grid of boards + per-cell captions + optional inline distractors.
    If chapter_opener is True, reserve the top strip for an ornamental
    header (●○ row + banner) and limit to 1-2 rows of boards."""
    W, H = img.size
    margin = 60

    body_font = _load_font(lang.font_path, rng.randint(14, 18))
    header_font = _load_font(lang.font_path, rng.randint(22, 28))
    caption_font = _load_font(lang.font_path, rng.randint(16, 20))
    small_font = _load_font(lang.font_path, rng.randint(12, 14))
    badge_font = _load_font(lang.font_path, 14)

    ink = (20, 20, 20)

    # Top of grid depends on header style.
    if chapter_opener:
        # Ornament strip at the very top (●○ flanking a title banner).
        strip_top = margin // 2
        strip_bot = strip_top + rng.randint(36, 52)
        _draw_header_ornament_strip(
            draw, (margin, strip_top, W - margin, strip_bot),
            rng, make_paragraph(lang, rng.randint(2, 4), rng),
            header_font, ink,
        )
        # Then a filled section banner beneath it.
        banner_top = strip_bot + 24
        banner_bot = banner_top + rng.randint(44, 56)
        section_font = _load_font(lang.font_path, rng.randint(26, 32))
        _draw_section_banner(
            draw, (margin, banner_top, W - margin, banner_bot),
            f"{rng.randint(1, 20)}. " + make_paragraph(lang, rng.randint(2, 4), rng),
            section_font, ink,
        )
        grid_top = banner_bot + 32
    else:
        _draw_page_header_text(draw, W, margin, lang, rng, header_font, ink)
        grid_top = margin + header_font.size + 20

    # Date-stamp table in upper-right on some non-chapter-opener pages.
    if not chapter_opener and rng.random() < 0.25:
        _draw_date_stamp_table(
            draw, (W - margin - 110, margin // 2), lang, rng, ink,
        )

    # Layout — chapter openers tend to use a single column and few rows.
    if chapter_opener:
        rows, cols = rng.choice([(1, 1), (1, 2), (2, 2)])
    else:
        rows, cols = rng.choice(LAYOUTS)

    grid_bottom = H - margin - rng.randint(20, 60)
    grid_h = grid_bottom - grid_top
    grid_w = W - 2 * margin
    cell_w = grid_w // cols
    cell_h = grid_h // rows

    page_style = {
        "box_boards": rng.random() < 0.35,
        "row_rules": rng.random() < 0.4,
        "col_rules": rng.random() < 0.2,
        "outer_border": rng.random() < 0.25,
        "jitter": rng.random() < 0.75,
        "figure_labels": rng.random() < 0.5,
        "problem_badges": rng.random() < 0.4,
        "footer_ornament": rng.random() < 0.4,
    }
    rule_ink = (60, 60, 60)

    if page_style["outer_border"]:
        pad = rng.randint(18, 34)
        draw.rectangle(
            [(margin - pad, margin - pad), (W - margin + pad, H - margin + pad)],
            outline=rule_ink, width=1,
        )
    if page_style["row_rules"] and rows > 1:
        for ri in range(1, rows):
            y = grid_top + ri * cell_h - rng.randint(2, 8)
            draw.line([(margin, y), (W - margin, y)], fill=rule_ink, width=1)
    if page_style["col_rules"] and cols > 1:
        for ci in range(1, cols):
            x = margin + ci * cell_w - rng.randint(2, 8)
            draw.line([(x, grid_top), (x, grid_bottom)], fill=rule_ink, width=1)

    boards: list[BoardAnnotation] = []
    problem_num = 1
    for row_i in range(rows):
        for col_i in range(cols):
            cx0 = margin + col_i * cell_w
            cy0 = grid_top + row_i * cell_h

            jitter_x = rng.randint(-18, 18) if page_style["jitter"] else 0
            jitter_y = rng.randint(-12, 12) if page_style["jitter"] else 0
            caption_x = cx0 + jitter_x
            caption_y = cy0 + jitter_y

            # Caption (and optional numbered badge to its left).
            caption_cursor_x = caption_x
            if page_style["problem_badges"]:
                badge_r = _draw_problem_badge(
                    draw,
                    (caption_x + 12, caption_y + caption_font.size // 2 + 2),
                    problem_num, badge_font, rule_ink,
                )
                caption_cursor_x = caption_x + badge_r * 2 + 8
            caption = f"{problem_num}"
            draw.text(
                (caption_cursor_x, caption_y),
                caption, fill=ink, font=caption_font,
            )

            # Board.
            window = _pick_window(rng)
            stones = random_stones(
                window,
                density=rng.uniform(0.15, 0.35),
                # Most pages have few marks; ~15% are "move sequence"
                # diagrams where almost every stone is numbered/marked.
                mark_prob=(
                    rng.uniform(0.6, 0.95) if rng.random() < 0.15
                    else rng.uniform(0.05, 0.3)
                ),
                rng=rng,
            )
            style = _fit_board_in_cell(cell_w, cell_h, window, _random_style(rng))
            rb = render_board(stones, window=window, style=style)

            inner_jx = rng.randint(-10, 10) if page_style["jitter"] else 0
            inner_jy = rng.randint(-6, 10) if page_style["jitter"] else 0
            board_x = cx0 + (cell_w - rb.image.width) // 2 + inner_jx
            board_y = caption_y + caption_font.size + 8 + inner_jy
            # Reserve space above the board for an optional figure label.
            if page_style["figure_labels"]:
                board_y += small_font.size + 14
            if board_y + rb.image.height > cy0 + cell_h - 40:
                board_y = cy0 + 4 + inner_jy
                if page_style["figure_labels"]:
                    board_y += small_font.size + 14
            board_x = max(margin, min(W - margin - rb.image.width, board_x))
            board_y = max(grid_top, min(H - margin - rb.image.height, board_y))
            img.paste(rb.image, (board_x, board_y))

            if page_style["box_boards"]:
                pad = rng.randint(4, 10)
                draw.rectangle(
                    [(board_x - pad, board_y - pad),
                     (board_x + rb.image.width + pad - 1,
                      board_y + rb.image.height + pad - 1)],
                    outline=rule_ink, width=1,
                )

            # Figure label above the board, drawn AFTER the board so the
            # label never overlaps the board's tight bbox.
            if page_style["figure_labels"]:
                label_text = rng.choice([
                    f"{problem_num} 도", f"Fig {problem_num}",
                    f"problem {problem_num}", f"Dia {problem_num}",
                ])
                _draw_figure_label(
                    draw,
                    (board_x + rb.image.width // 2, board_y - small_font.size),
                    label_text, small_font, rule_ink,
                )

            loose_bbox = (
                board_x, board_y,
                board_x + rb.image.width - 1,
                board_y + rb.image.height - 1,
            )
            n_cols = window[1] - window[0] + 1
            n_rows = window[3] - window[2] + 1
            tight_x0 = board_x + style.margin
            tight_y0 = board_y + style.margin
            tight_x1 = tight_x0 + (n_cols - 1) * style.pitch
            tight_y1 = tight_y0 + (n_rows - 1) * style.pitch
            tight_bbox = (int(tight_x0), int(tight_y0), int(tight_x1), int(tight_y1))
            P = BBOX_DETECTOR_PAD
            padded_bbox = (
                max(0, int(tight_x0 - P)),
                max(0, int(tight_y0 - P)),
                min(W - 1, int(tight_x1 + P)),
                min(H - 1, int(tight_y1 + P)),
            )

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
            hoshi_centers = [
                (board_x + hx, board_y + hy) for (hx, hy) in rb.hoshi_pixels
            ]
            corner_centers = {
                k: (board_x + v[0], board_y + v[1]) if v is not None else None
                for k, v in rb.corner_pixels.items()
            }
            boards.append(BoardAnnotation(
                bbox=tight_bbox,
                bbox_padded=padded_bbox,
                loose_bbox=loose_bbox,
                window=window,
                edges_on_board=rb.edges_on_board,
                edge_class=edge_class,
                stone_centers=stone_centers,
                hoshi_centers=hoshi_centers,
                corner_centers=corner_centers,
            ))

            # Question / answer text under the board.
            text_y = board_y + rb.image.height + 8
            text_h = cy0 + cell_h - text_y - 10
            if text_h > body_font.size:
                question = make_paragraph(lang, rng.randint(8, 18), rng)
                _draw_wrapped(
                    draw, (cx0 + jitter_x, text_y), cell_w - 12, text_h,
                    question, body_font, (40, 40, 40),
                )
            problem_num += 1

    # Footer ornament at page bottom.
    if page_style["footer_ornament"]:
        _draw_footer_ornament(
            draw, (W // 2, H - margin + 6),
            rng.randint(1, 200), small_font, rule_ink,
        )

    return boards


def compose_page(
    lang_code: str | None = None,
    rng: random.Random | None = None,
    page_size: tuple[int, int] = (1000, 1400),
    kind: str | None = None,
) -> Page:
    r = rng or random.Random()
    if lang_code is None:
        lang_code = r.choice(list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_code]

    W, H = page_size
    img = Image.new("RGB", (W, H), (252, 248, 236))
    draw = ImageDraw.Draw(img)

    if kind is None:
        kinds, weights = zip(*PAGE_KINDS)
        kind = r.choices(kinds, weights=weights)[0]

    if kind == "cover":
        boards = _render_cover(img, draw, lang, r)
    elif kind == "preface":
        boards = _render_preface(img, draw, lang, r)
    elif kind == "chapter_opener":
        boards = _render_problems(img, draw, lang, r, chapter_opener=True)
    else:
        boards = _render_problems(img, draw, lang, r, chapter_opener=False)

    return Page(image=img, lang_code=lang_code, boards=boards)
