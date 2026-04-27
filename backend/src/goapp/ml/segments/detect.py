"""Line-segment extraction via Canny + Probabilistic Hough Lines.

This is the perception step of the segment-based grid detector: instead of
collapsing the crop into 1D projections (which is what `pitch.measure` does
and what fails on ragged scans), we extract straight-line primitives
directly so a downstream reasoning step can cluster them by orientation
and infer pitch from the modal spacing.

We use Probabilistic Hough rather than LSD because LSD is region-growing
— it requires a minimum density of aligned-gradient pixels in a 2D
neighborhood to accept a line, which 1-px-thin sharp grid lines on
clean prints fail (the "region" is one pixel wide). Hough is parameter
voting — each edge pixel votes independently for the (rho, theta) line
it could lie on — so long thin uniform lines are its easiest case.

Output is a list of (x1, y1, x2, y2) tuples in crop coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Drop segments shorter than this fraction of min(W, H). A 19-line board's
# grid lines span the full crop; even partial-board crops have grid
# segments well above this floor. Filters short noise segments from text
# and stone artifacts.
MIN_LENGTH_FRAC = 0.05

# Hough vote threshold as a fraction of min(W, H). A grid line of length L
# produces ~L Canny edge pixels; threshold = 0.15 * min(W, H) means a
# line needs to span at least ~15% of the crop to register. Lower values
# pick up faint columns at the cost of more text-stroke noise.
HOUGH_THRESHOLD_FRAC = 0.15

# Maximum gap between collinear pixels still counted as one segment. Lets
# Hough span paint-out voids and minor print breaks without splitting.
HOUGH_MAX_GAP_FRAC = 0.08


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return float(np.hypot(dx, dy))


def detect_segments(crop_bgr: np.ndarray) -> list[Segment]:
    """Canny → Probabilistic Hough → length-filtered segments."""
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_dim = min(w, h)
    min_len_px = int(MIN_LENGTH_FRAC * min_dim)
    threshold = int(HOUGH_THRESHOLD_FRAC * min_dim)
    max_gap = int(HOUGH_MAX_GAP_FRAC * min_dim)

    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len_px,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    out: list[Segment] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        out.append(Segment(float(x1), float(y1), float(x2), float(y2)))
    return out
