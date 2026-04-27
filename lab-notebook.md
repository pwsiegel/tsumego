# Tsumego pipeline — lab notebook

What we have tried, in roughly chronological order. Sourced from git history
(commit dates) and recent conversation history. Status legend:

- **In use** — currently in the pipeline.
- **Superseded** — replaced by something better.
- **Scrapped** — tried, didn't work, removed.
- **Deferred** — known issue, parked.

---

## Project goal

Extract Go problems from PDF books and turn each detected board into
something the user can review, edit, and train against. Pipeline at the
ML level is roughly: **PDF page → YOLO board bbox → board crop → grid
geometry (pitch + origin + edges) → stone detection → discretized 19×19
position → SGF**.

---

## 2026-04-18 — initial scaffold

- Vite + React + TypeScript frontend, Go board renderer, FastAPI backend
  with PDF ingest + per-board labeling UIs.
- **Hand-labeled board bboxes (~30 boards from one PDF)** — used to
  fine-tune YOLOv8n. Status: **superseded** (deprecated 04-20 in favor of
  synth, see below). Crops were too tight and clipped edge lines, hurting
  downstream geometry.
- **Hand-labeled stone-point crops (~30)** — used as seed training set
  for the eventual stone CNN.
- **Classical stone-circle detection** (HoughCircles + connected
  components + corner rejection + glyph-fill check + interior luminance).
  Status: **superseded** by the trained CNN.

## 2026-04-20 — synth page generator + specialized CNNs

Pivot away from real-data labeling (too slow, too few examples). Build a
synthetic Go-book page generator with full ground-truth annotations,
train one specialized model per task.

- **Synth page generator** (`goapp/synth/`) — renders 1–6 boards per
  page, full/corner/side/mid-board windows, stones + marks + hoshi,
  multi-script captions, distractors (figure labels, badges, ornament
  strips, footers), per-board jitter, rotation/blur/noise/JPEG
  degradation. Annotations: tight bbox, loose bbox, visible window,
  edges_on_board, edge_class, stone centers. **In use.**
- **YOLO board detector retrained on synth** with tight bboxes + single
  "board" class, conf threshold 0.5, aspect-ratio reject 0.25..4.0 to
  filter banners. **In use** (later retrained, see 04-26).
- **Stone CNN (UNet, 2-channel heatmap)** — synth-trained per-board crop
  detector. Cross-channel dedup for numbered glyphs in stones. Status:
  **scrapped** — near-zero white-stone recall on real PDFs.
- **Stone YOLOv8n detector** (replaces UNet) — 2-class B/W, peaks via NMS.
  **In use.**
- **Edge classifier (4-bit)** — small CNN, ~99% all-correct on synth val.
  Status: **scrapped 04-23** (replaced by the classical edge detector
  brought back in for reasons noted below).
- **Snap regressor (pitch + origin)** — small CNN+MLP, 4 fractional outputs.
  Synth val pitch error ~0.86%, origin ~1.4%. Status: **superseded** by
  classical pitch+origin (04-23).
- **19×19×3 grid classifier** (end-to-end image → grid). Cell accuracy
  ~87%, **below the all-empty baseline (~93%)**. Architecture has fixed
  pixel-to-cell mapping but crops have variable margin/pitch. Status:
  **scrapped**, kept for reference.
- **Joint stone+snap fine-tune** with bilinear sampling at predicted
  grid intersections + CE on 19×19×3. First pass: no improvement —
  gradient signal to snap drowned out. Status: **scrapped**.

End-of-day end-to-end accuracy (synth val): ~75% cell accuracy / ~44%
stone recall. Each component near-perfect alone; pitch/origin error
compounds over 10+ cells.

## 2026-04-21 — YOLO-only refactor

- **Retire classical, snap, joint stacks**, drop CompareDetectors UI.
  Pipeline is reduced to: YOLO crop → stone CNN → discretize via
  classical pitch + image-anchored coordinates. Status: ~stable~.
- **Image-anchored discretize** — stones get coords from the image, not
  from a per-board snap regression. **In use** (this is what
  `discretize.discretize` does today).
- **Synth distractor-rich pages + bbox_padded label** — board detector
  gets trained against figure labels, ●○ ornament strips, banners,
  footers, date stamps. bbox_padded gives YOLO an 8-px safety margin.
  **In use.**

## 2026-04-22 — pipeline polish + persistence + corner detector

- **Corner detector** (per-board-crop training, MPS) — separate model to
  refine bbox corners. Status: **scrapped 04-23** (replaced by classical
  corner+edge geometry fix).
- **Reading-order bbox sort** + **stone dedupe per cell**. **In use.**
- **Tsumego persistence + review flow + new routed web UI** — accepted
  problems saved as SGF + JSON + crop PNG, per-user library on disk.
  **In use.**
- **Validation dataset export + old-vs-new comparison + triage UI** —
  freeze a hand-reviewed set as ground truth, run new pipelines against
  it, inspect diffs in the dev tools. **In use** (`/testing/validate/hm2`).

## 2026-04-23 — pipeline reset

- **`9950f28` inference: corner-based geometry + imgsz=640 rescaling
  fix.**
- **`8303ab5` pipeline: revert to classical geometry + add validation
  endpoint** — the corner detector regressed real PDFs and was reverted.
  Classical edge + pitch detection becomes the canonical path. **In use
  (still the default in `_resolve_geometry`).**
- **`5376188` cleanup: remove dead ML modules and unused dependencies.**
- Validation report page added to dev tools.

## 2026-04-25 — deployment + Modal training + restructure

- **Cloud Run + IAP** deployment with per-user library and signed-URL
  PDF upload (Cloud Run's 32 MiB cap). **In use.**
- **Repo restructure**: `api/` → `backend/`, split monolithic main.py
  into domain routers + ml/ subtree + cli/.
- **Vertex AI L4-spot training** setup (Dockerfile + Cloud Build + job
  YAMLs). Status: **scrapped** — spot capacity unreliable in
  us-central1, jobs queued indefinitely or failed mid-run, 10–15 min
  cold starts made iteration painful.
- **Modal training** — replaces Vertex. Per-second L4 billing, seconds
  to spin up, $30/mo free tier covers fine-tuning. **In use**
  (`make modal-train-*`).
- **Models become repo assets** under `backend/data/models/` — clone and
  go, no GCS pull step.
- **Refactor: lift inference magic numbers to named module constants** —
  `pipeline.py`, `stone_detect`, `edge_detect`, `pitch`, `discretize`
  all get explicit named constants with comments. Pure readability.

## 2026-04-26 — grid geometry regressor experiment

Real PDFs whose frame and grid lines are uniformly thin break the
classical edge detector: `max(thick_score, dark_score)` clusters at the
threshold and pitch detection cascades to wrong cell sizes. Two
attempts to learn the geometry instead:

- **`4893255` — ResNet18 grid-geometry regressor** that predicts
  (gx1, gy1, gx2, gy2, px, py) normalized to crop dims. Trained on
  synth crops with per-side jitter and small rotation/shear. Status:
  **scrapped** — pitch/crop_width target is non-stationary under
  jitter, so the model regressed toward the training mean. No jitter
  setting reconciled training distribution with what real YOLO crops
  look like.
- **`91f4301` — bumped jitter from 25 → 200 px** to try to break the
  anchoring on crop boundaries. Did not help, same calibration failure.
  Status: **scrapped**, kept for reference.

## 2026-04-26 — intersection detector + lattice fitter (current line of work)

Pivot from "predict the geometry directly" to "detect lots of local
features + reconstruct the lattice from them."

- **Intersection detector (YOLOv8n)** — single class "X" for a visible
  perpendicular intersection. Bbox half-frac 0.25·pitch (smaller than
  stones' 0.4) so adjacent intersections don't collide under NMS.
  flipud/fliplr augmentation on. Synth val: **P=0.9998 R=1.0
  mAP@50=0.995 mAP@50-95=0.99** at epoch 25 — saturated synthetic val,
  real PDFs are the real test. **In use as a model**, not yet wired
  into `_discretize_board`.
- **`intersection_detector_no_edges.pt`** variant — same architecture,
  T/L corner-style labels dropped. Diagnostic; available behind the
  `model=no_edges` query param on `/api/pdf/board-intersections`.
- **Lattice fitter** (`fit_lattice` in `intersection_detect/lattice.py`)
  — fits a 2D lattice (pitch_x, pitch_y, origin_x, origin_y, n_cols,
  n_rows, edges) to detected intersections + stone centers. RANSAC-ish
  inlier filter at 0.3·pitch. Output includes per-side "real vs
  window-cut" classification.
- **`/testing/intersections` dev tool** — overlays intersections,
  stones, fitted lattice, and edge rectangle on the board crop. **In
  use.**

## Current session (2026-04-26) — bbox-snap detour and lattice/classical comparison

Started from the observation that on real PDFs the YOLO board detector
sometimes engulfs caption text under boards, polluting downstream stages.

- **Lattice-snap bbox postprocessing** in `pipeline.py`
  (`_snap_bbox_to_lattice`, `_snapped_page_bboxes`, used by
  `_board_crop`). Tried multiple iterations:
  1. Edge-flag-driven snap. Failed: edges flag flipped to "window-cut"
     under loose YOLO crops because lattice extent reached past the real
     board edge.
  2. Clamp-to-raw semantics with cushion. Result: green snapped box was
     identical to red raw bbox most of the time (no signal getting
     through).
  3. Density-based outer-row/col trim. Result: over-aggressive on real
     edge cols with only 1–2 stones; clipped legitimate stones.
  4. Conditional trim — only trim a side where the un-trimmed lattice
     extends past the raw bbox on that side. Result: still losing
     edges of the board on some examples.
  5. Cushion bumped 0.5 → 1.0 pitch. Helped a bit but the underlying
     signal was decorative.
  - Status: **scaffold still in `pipeline.py`**, ready to revert. The
    user has explicitly called this over-engineered and pointed out
    that "shit pipeline results" was the real motivation, not bbox
    cosmetics.
- **Lattice vs classical geometry comparison view** in `IntersectionTest`
  — extends the `/api/pdf/board-intersections` response with the
  classical `_resolve_geometry` output. Frontend overlays both grids on
  the same crop, with independent show/hide toggles for each.
  - First sample browsed (cho-chikun page 3, bbox 2): both fits got the
    pitch ≈ right (~17 px), edge classifications were opposite and
    ambiguous, **YOLO bbox itself was wrong** (n_cols=20 from lattice
    means it extended one column past the real right edge into caption
    space), and "Board 3 of 858" suggests YOLO is firing many false
    positives across the cho-chikun PDF.
  - Second sample (hm2 page 7, bbox 1): **lattice fitter completely
    failed** — pitch_x came back as 21.5 (5× too small) when classical
    correctly said 108. Lattice reported a 93×21 grid spanning the
    image. RANSAC apparently found a sub-pitch that passed the inlier
    threshold. **This is a serious failure mode for fit_lattice that
    would silently produce garbage stone coordinates if wired into
    `_discretize_board`.**
- **Truncated test PDFs** (`~/Downloads/hm2-first20.pdf`,
  `~/Downloads/cho-chikun-2-first20.pdf`) created so we can iterate
  faster on the real-PDF browse loop without 100+ pages of YOLO per
  upload.

---

## 2026-04-26 — paint-out-stones before classical geometry (in progress)

Hypothesis: classical line detection (`edge_detect` + `pitch.measure_grid`)
is being polluted by stone edges. Long horizontal/vertical runs of
stones produce intensity peaks that look like extra grid lines, and
dense clusters pull pitch detection off. The stone detector itself is
trustworthy.

Approach:
1. Run the stone detector on the crop.
2. For each stone, sample board color from a thin annulus just outside
   the stone's radius (median of those pixels). Paint the stone disc
   with that color.
3. Run classical `_resolve_geometry` on the cleaned crop instead of
   the original.

Variant skipped: placing a center dot at each painted-out stone. The
dot would either match or contradict the underlying grid line — and
since line detection is what we're trying to make work, adding a
synthetic feature there risks dominating the peak fit. Going with
no-dot first; revisit if line detection still struggles.

**Risk**: only as good as the stone detector. Missed stones stay in
the cleaned crop and continue contaminating line detection.

Dev surface: a toggle in `/testing/intersections` that swaps the crop
image for the cleaned version and re-runs classical on it. Lets us A/B
the geometry result with one click.

Status: dev surface wired up 2026-04-26. `paint_out_stones` lives at
`backend/src/goapp/ml/stone_detect/clean.py`; `/api/pdf/board-cleaned/...`
serves the cleaned crop; `BoardIntersections.classical_cleaned` carries
the geometry from the cleaned crop.

Outcome 2026-04-26: cho-chikun samples lock onto the real grid cleanly
once stones are painted out. Initial pad of 1 px left stone outlines
intact on hm2 (large dense clusters), producing pitch_x sub-pitch
errors. Bumped `_FILL_PADDING` 1 → 4 (and slid the annulus inner past
the new paint zone) — cho-chikun now looks "amazing"; hm2 is mostly
fine apart from real page-scan curvature on some boards which classical
can't correct without going to a homography fit. Approach validated.
Next: wire `_resolve_geometry` in `_discretize_board` to operate on the
cleaned crop instead of the raw one.

---

---

## 2026-04-26 — homography / border-warp for page-curvature (scrapped)

Hypothesis: hm2 boards have keystone / page-curl distortion that
1D-projection pitch detection cannot correct. Approach: detect the four
outer-border lines via Hough, intersect to get corners, perspective-warp
the quad to a rectangle, then derive grid geometry directly from the
warp output (corners ARE the 19th-line corners, so
`pitch = (side - 1) / 18`).

Code: `backend/src/goapp/ml/border_warp.py`,
`/api/pdf/board-warped/{...}.png`,
`BoardIntersections.classical_warped`, plus a "warped" toggle in
`/testing/intersections`.

Failure modes encountered:
1. **Window-cropped boards (most cho-chikun)** have no outer border in
   the bbox. Hough latches onto arbitrary lines, producing a
   near-degenerate quad. Added validity gates (opposite-side ratio,
   area fraction, edge tilt). Gates correctly reject most of these,
   but warp returning None means the dev view shows nothing — and
   wiring a fallback to `classical_cleaned` defeats the point.
2. **Tilted parallelograms slip past the gates** — Hough picks
   *interior* grid lines on rotated scans; the resulting quad has equal
   opposite sides and big area, but corners spill outside the crop and
   the warp output has black wedges with the actual board content
   skewed diagonally.
3. **Even when warp succeeds on hm2**, the warped output is only a
   marginal improvement over `classical_cleaned`, and the bbox doesn't
   reliably contain the *full* outer border — so the geometric
   pitch=(side-1)/18 derivation is off by a row/column on those.

User verdict 2026-04-26: "warping is making everything worse" — net
regression vs `classical_cleaned`-only. Status: **scrapped.** Don't
re-propose without first solving the "is the bbox guaranteed to contain
the full 19-line border?" question, since the geometric derivation
collapses without that guarantee.

---

## 2026-04-26 — drop lattice-snap; rewrite edge detector

Two related cleanups motivated by tesuji + hm2 failures:

**Lattice-snap removed.** `_snap_bbox_to_lattice` was a second YOLO-bbox
post-processor that trimmed bboxes to the intersection-only `fit_lattice`
extent. It was added back when the only defense against caption-engulfing
loose YOLO bboxes was a separate snap pass. The fused lattice
(`fit_lattice_fused`) handles caption defense downstream by being driven
by segments, stones, and intersections together — so the snap step now
only causes harm: in hm2 it actively trimmed off real edges (visible as
"green-box-cut-into-board" in the dev tool). Removed all `_snapped_*`
plumbing — routes, schemas, and the BboxTest UI now show only the raw
YOLO+pad bbox.

**Edge detector rewritten — first attempt: lattice-probe (scrapped).**
The thickness/darkness detector worked on cho-chikun's heavy frames but
silently failed on tesuji's thin frames (outer line drawn the same
weight as interior). First replacement was a lattice-based "predict next
line" probe: take the outermost detected lattice line per side, step one
pitch outward into the *page* image (using `crop_offset`), threshold a
strip there, declare "real edge" if no ink, "window cut" if ink found.
Failure mode: page layout spacing (gaps between adjacent diagrams,
captions, headers) is comparable to lattice pitch. Whitespace gaps gave
false real-edges; nearby text gave false window-cuts. Saw 5 distinct
failure cases across cho-chikun and hm2 in one test session.
**Status: scrapped.**

**Edge detector — second attempt: corner topology (insufficient
alone).** Don't peek outside the crop; reason about segments inside
it. At a real corner the outermost perpendicular lines terminate
(L-junction); at a window cut they continue past (T-junction). Per
side, count distinct perpendicular Hough segments whose endpoints
extend past the outermost detected lattice line by more than
0.4×pitch. 0–1 distinct extenders → real edge; 2+ → window cut.
Failure modes seen on validation set:
1. Cho-chikun top-strip diagrams: pure whitespace below outer row, no
   segments to register against → false real-edge call on bottom.
2. hm2 boards where the lattice itself missed the outermost frame line
   (faint/curled scan): topology correctly says "this is a window
   cut" relative to the lattice, but the lattice is wrong upstream.
3. Validation report after rollout: hm2 went 22.6% exact, 77% changed
   vs. baseline. Big regression.
Status: **kept as one of two stacked signals** — see next entry.

**Edge detector — third attempt: thickness + topology stacked (in
use).** Add back the thickness/darkness signal that the original
classical detector used, but as a *positive vote layered on top of
topology*, not as the sole signal. Per side, sample a perpendicular
strip at the outermost lattice line and at the first interior line;
measure thickness (run length above adaptive-threshold density) and
darkness (median grayscale of the dark pixels). If outer is at least
1.5× thicker OR 15+ gray levels darker than inner → vote real edge.
Combine: `real_edge = topology_real OR thickness_real`. Topology
catches thin-frame books (tesuji) where every line is the same weight;
thickness catches hm2 / cho-chikun heavy-frame books where the outer
frame is unmistakably bolder than interior. 19-line sanity rule still
runs after to enforce that opposite sides can't both be "real" when
the lattice fits fewer than 19 lines on that axis. Constants:
`THICKNESS_STRIP_FRAC=0.4`, `THICKNESS_LENGTH_FRAC=0.6`,
`THICKNESS_DENSITY_FRAC=0.5`, `THICKNESS_DENSITY_FLOOR=0.25`,
`THICKNESS_RATIO_GATE=1.5`, `DARKNESS_DELTA_GATE=15.0`.

Note: thickness vote presumes the lattice has correctly identified
the outermost board line. When the lattice itself is short of the
real frame (hm2 image #11 case: leftmost detected col was col 1, real
frame at col 0 was missed by `fit_lattice_fused`), thickness measures
an interior line and votes "no", same way topology does. Fixing that
class of failures requires extending the lattice toward faint-but-
present outer frames upstream — open thread, not addressed here.

---

## 2026-04-26 — T-junction-only edge detector (dev tool)

After the topology+thickness stack regressed validation, scrapped that
approach. Reasoning a human uses: an edge is unambiguously visible
because intersections on it are T-junctions (or L at corners). Window
cuts produce + junctions instead. So if we can reliably detect stubs
from existing line segments, we get a single signal that's hard to
confuse.

New module `backend/src/goapp/ml/edge_detect/tjunction.py`. For each
lattice intersection in the crop, group horizontal segments by snapped
row and vertical segments by snapped column, then check whether any
segment in the matching row/column extends ≥ MIN_EXTENT_FRAC * pitch
(0.4) past the intersection in each cardinal direction. Stubs are a
4-bit mask (N/E/S/W). Classify as +/T/L/endpoint/isolated. A side is
"real" iff at least one T or L on its outer lattice line is missing
the outward stub AND no + junctions on that line outvote them.

New endpoint `/api/pdf/board-tjunctions/{page_idx}/{bbox_idx}` and
new dev UI at `/testing/edges`. UI shows per-junction stubs colored by
kind, per-side T/L vs + tally, and a "Run on all boards" button that
aggregates how many edges/sides we recover across an entire upload.

Goal of this iteration: gather data — does this signal alone get us
enough edges? If yes, scrap the existing `edge_detect/detect.py`.
The current `_discretize_board` still calls the old detector; this
module is dev-tool-only until we see how it performs.

**2026-04-26 update — segment-based variants scrapped.** Tried two
versions: lattice-stub (above) and pure segment-endpoint topology
(endpoint-on-perpendicular = T, endpoints-meet = L), with a broken-line
continuation check to merge Hough's split halves. Both failed because
Hough segments don't preserve topology: voting peaks snap segment
endpoints to nearby intersections, so a column that visually crosses a
horizontal often comes back as two segments terminating at the
horizontal — phantom T's. On clean tesuji prints with obvious +
junctions, the detector kept reporting T's and firing edges where
there shouldn't be any.

**Replaced with skeletonization.** New approach in
`edge_detect/tjunction.py`: binarize → `skimage.morphology.skeletonize`
→ prune endpoints `PRUNE_ITERS` times → 8-neighbor count per skeleton
pixel → cluster pixels with ≥ 3 neighbors via 8-connected components →
recover arm directions by walking the skeleton outward
`ARM_WALK_STEPS` from each cluster boundary and snapping the resulting
displacement to N/E/S/W (`ARM_AXIS_RATIO` ≥ 2.5 = within ~22° of axis-
aligned). Pixel skeletons preserve topology directly: a line crossing
another stays connected through the crossing, a line terminating
against another stays a 3-arm junction. L corners (2-neighbor pixels)
are not detected — they don't show up as `nbr ≥ 3` — but T's along
each side are sufficient to fire edge detection on tsumego diagrams
since real edges have many interior columns crossing the outer line.
Added `scikit-image>=0.24` to backend deps.

---

## 2026-04-26 — skeleton T-junction edge detector, current session

Refining the skeletonization detector against real PDFs (tesuji,
cho-chikun-2). Goal: edge detection from purely **local** shape signals
— no crop heuristics, no periphery filters.

**Constraint repeatedly emphasized by user**: do not introduce any
"near-the-bbox-boundary" or "extremum perpendicular position" tests. If
the human eye can see the edge from local shape, the algorithm should
too.

**Changes this session:**
- **Pruning removed.** `PRUNE_ITERS=6` was eating the short axis-aligned
  stubs that artists draw past an interior row to indicate "no edge
  here, board continues." Pruning was the cause of a phantom bottom
  edge in chochikun where a row of `+` junctions had small stubs that
  the algorithm was reading as `T` after pruning. User directive:
  "premature optimization fucked this up — yes remove it."
- **Co-linear cluster filter** on T's: cluster voting T/L's by
  perpendicular position, require ≥ MIN_EDGE_CLUSTER (=2) co-linear
  votes within EDGE_COLINEAR_FRAC (=0.04) of crop dim. Suppresses
  phantom T's from stone paint-out severing arms of interior +'s.
- **Caption removal via bbox-overlap CC filter.** Adaptive-threshold
  binarize → 8-connectivity CCs → keep only CCs whose bbox overlaps
  the largest CC's bbox. Removes "problem 14" caption text outside the
  board frame without removing the left edge in cases where stones
  sever the leftmost column from the main grid CC. (First version was
  largest-CC-only; that dropped real edges when the leftmost column got
  separated by edge stones — bbox-overlap fixes that.)
- **EdgeTest UI** now lists each voting T/L's `(x,y)` per side, useful
  for diagnosing why a side fires/doesn't fire.

**Known remaining failure mode:** edges where stones obscure most of
the outer-row intersections, leaving only 1 T detected. With
MIN_EDGE_CLUSTER=2 those edges go unreported. Cluster augmentation by
"stones-on-this-row" was discussed and rejected as too coarse: a single
phantom T in the same row as 4 mid-board stones would fire a false
edge.

**Open thread the session ended on:** is there a **local** test for
"this stone is sitting on the board edge"? User wants a per-stone
visual check (look at the immediate neighborhood of one stone) that is
independent of where other stones are or of bbox position. Possible
local signals to consider next:
- Outer frame line passes through one side of the stone — grid
  continues in only 3 cardinal directions past the paint-out radius,
  not 4.
- After paint-out, the strip just outside the stone on one side is
  whitespace while the other three sides have grid pixels nearby.
- Skeleton junctions adjacent to the painted-out stone disc only form
  in 3 of 4 directions.

If a reliable per-stone local edge test exists, augment cluster votes
on each side with stones that locally test as edge-touching.

Status: skeleton detector lives in `edge_detect/tjunction.py` (still
dev-tool-only via `/api/pdf/board-tjunctions` + `/testing/edges`). Old
`edge_detect/detect.py` still wired into `_discretize_board`. Will swap
once skeleton detector handles the stone-occluded edge cases.

---

## 2026-04-26 — per-stone local edge test (dev tool)

Motivation: cho-chikun problems where stones sit directly on the left
edge cause paint-out to erase the leftmost frame line, so the skeleton
detector finds 0–1 T's on the left and refuses to fire. We need a local
per-stone signal that doesn't depend on the painted-out grid.

Signal (per stone, per N/E/S/W): "stone is on edge D" iff
  1. no other stone within ~1.5·pitch in direction D (perp window
     0.5·pitch), AND
  2. no grid ink in the **half-plane** extending past the stone in
     direction D, clipped to the main-grid CC bbox so captions don't
     count. Strip starts 1.4·r past stone center, perpendicular
     half-width 1.2·r.
First version checked a thin strip at the predicted next intersection
(1.1·pitch out, ~0.5·pitch wide); too brittle — false-fired all 4
sides on isolated stones in regions with faint grid. Half-plane
formulation is much stronger: a true edge stone has nothing past it,
an interior stone has many grid lines past it that swamp the gate
(`INK_PIXEL_GATE=30` black pixels) even when individual lines are
faint. Pitch (used for neighbor proximity only) estimated as `2.5 ×
median stone radius`.

Stone false positives (e.g. "o" in "problem 14" caption) are filtered
out before classification by main_grid_bbox: any YOLO detection whose
center sits more than 1·median-radius outside the largest binary-CC
bbox is dropped. `main_grid_bbox` lifted from `_skeletonize` into a
public helper for sharing.

Code: `backend/src/goapp/ml/stone_detect/edge_test.py`. Wired into
`/api/pdf/board-tjunctions` (extra `stone_edges` field in
`BoardTJunctionEdges`); EdgeTest UI overlays an orange outward tick on
each stone for each side that fires, plus an orange ring to mark
classified stones.

In the route, `edges[side] = T-cluster vote OR ≥2 stones classify
that side as edge`. Single stone is too lonely — a real edge with
a stone on it should also produce T-junctions along that line, so a
solo stone vote with zero T's is more often a misfire than an edge.

Two cleanup filters run before the merge:
1. **Junction-inside-painted-disc filter.** Skeletonization on the
   painted-out crop sometimes leaves 2-arm corner pixels where the
   paint disc edge severs an interior grid line — they look like
   real L-junctions but are paint-boundary artifacts. Drop any
   junction whose centroid sits within `paint_radius(r)` of any
   stone center, then re-tally. `paint_radius` lifted out of
   `clean.py` as a public helper.
2. **Stone-beyond-edge sanity.** After merging, each fired edge is
   anchored to a position (median y/x of voting T/L's, fallback to
   the outermost edge-stone). If any stone center lies more than
   ~1 median-stone-radius past that position in the outward
   direction, the edge call contradicts the stone's existence on
   the board and is rejected.

**Wired into `_discretize_board` 2026-04-26.** Orchestration lives in
`edge_detect/skeleton.py` (`decide_edges` returns merged edges plus
filtered junctions / sides / stone classifications for the dev tool).
Both the dev-tool route and the main discretization pipeline call it.
Caption-glyph filter (`filter_to_grid_bbox` in `stone_detect/clean.py`)
runs on stones before everything downstream — so YOLO's false
positives on caption text don't pollute lattice fitting, paint-out,
or the discretized output. Old `_resolve_geometry` + classical
`detect_edges` kept as the diagnostic compare path on
`/api/pdf/board-intersections` and `cli/compare_on_val.py`.

---

## 2026-04-27 — drop CNN intersection detector from lattice fit, replace with skeleton junctions

**Problem.** Cho-chikun-2 problems 9, 11, 12 rendered with off-by-one
column / row shifts despite the correct edges firing. Three different
failure modes, all from the CNN intersection detector poisoning
`fit_lattice_fused`:

1. **One pitch outside the board (problem 11).** Spurious ix at `x≈2.6
   px` outside `main_grid_bbox.x0=11`. First fix attempt: re-use
   `filter_to_grid_bbox` with `margin = median(stone_r) ≈ 7`. Caught
   problem 11 but missed problem 12 (where `gbb.x0=8` and the ix at
   `x=2.6` was within the 7-px margin).
2. **Top-of-stone-rim arcs (problem 9).** Ix model fires on stone-arc
   pixels at ~one stone-radius above each row of stones. These form a
   parallel rhythm at half-pitch offset, which the fitter latched onto
   (pitch_y=18.89 vs real ≈17, origin_y=31.79 vs real ≈19), shifting
   everything one row up.

**Why these slipped past the dev tool.** The
`/testing/intersections` view rendered raw CNN ix points but no
lattice consequences — ix on stone-rims looks visually plausible
(near each stone), so the off-by-half-pitch rhythm was invisible
until you compared the rendered SGF to the original.

**Fix.** Drop the CNN ix detector from `_discretize_board` entirely
and use the **skeleton T/L/+ junctions** (already computed in
`decide_edges` for edge classification, with paint-disc artifacts
filtered out) as the intersection signal for `fit_lattice_fused`.
Skeleton junctions can't fire on stone rims by construction —
they're computed on a binary skeleton of the painted-out crop, and
junctions inside any painted disc are filtered. Verified on
cho-chikun-2 boards 8, 10, 11: pitch and origin now lock to the
real grid (oy=19.26 vs real 19, ox=16.25 vs leftmost stone 16.6).

CNN ix model is still used by `/testing/intersections` (raw signal
view), but is no longer wired into the production discretization
pipeline. Skeleton junctions are now overlaid in that dev tool too,
colored by kind (T magenta, L orange, + green). Status: **in use**.

---

## 2026-04-27 — three hm2 fixes (pitch floor, stub-arm pruning, edge cluster)

**Three failures all on the same hm2 PDF, three independent root causes.**

1. **Half-pitch in `fit_lattice_fused` (board 9, page_idx=2 bbox 3).**
   Crop 1557×917, real pitch ≈ 80, fitter chose 39.9. Half-pitch
   ambiguity: every position on an N-pitch grid sits on N/2-pitch too,
   so inlier counts tie within noise. Earlier docstring claimed
   "doubling halves the inlier count, so its score never wins" — that
   only holds when there are positions OFF the doubled grid, which
   isn't true for a regular Go board. Fix: pass stone radii into
   `fit_lattice_fused`, set `pitch_floor = 1.7 × median(r)` on each
   axis, and filter candidate pitches below it. Stones physically
   can't overlap so this is a hard physical bound. Also fixed p3b6
   on the same PDF as a side effect (was 33.7 px, now 67.3 matching
   peers).

2. **Top-edge T's misclassified as `+` (board 13, page_idx=3 bbox 3).**
   *Attempted and reverted.* Vertical grid lines extend 20-30 px past
   the topmost grid row in the skeleton; my first attempt assumed
   these were always ornamental and pruned them in `_recover_arms`,
   which made + → T and made the edge fire. **User flagged a Go-puzzle
   convention I missed**: stubs that don't lead to another intersection
   are often the author's deliberate signal that *the board continues
   here, this is NOT an edge*. So pruning stubs would falsely fire
   edges on windowed-view diagrams (common in problem books). Reverted
   the change. Board 13's actual situation is ambiguous — there does
   appear to be a printed frame band (binary count peaks at y=64-72,
   over 700 px wide) — but disambiguating "frame band + decorative
   ticks" from "tick marks alone" needs scanning the binary for a
   thick horizontal band, not just walking the skeleton. Deferred.

3. **Bottom edge sanity check rejected by spurious interior T's
   (board 20, page_idx=5 bbox 0).** Stone paint-out severs interior
   `+` junctions into phantom T's whose outward direction matches
   the side they got severed on. 24 T/L's voted for the bottom on
   board 20, scattered from y=286 (top of crop!) to y=2068 (real
   bottom). `_edge_position` took the median of *all* of them →
   1897, miles above the real bottom at ~2050. The
   "stone-beyond-edge" sanity check then saw the BR cluster's stones
   as past the asserted edge and rejected the whole bottom. Fix:
   `_edge_position` now returns the median of the largest co-linear
   cluster (matches `tally_edges`'s notion of an edge) instead of
   the median of all voters. Bottom now fires; all 34 detected
   stones make it into the SGF.

Status: **all in use**. None of these fixes touched the lattice
floor / stub probe / edge cluster logic in `_resolve_geometry` or
`detect_edges` (the legacy classical path), which remain as the
diagnostic compare-against on `/api/pdf/board-intersections`.

---

## 2026-04-27 — filter lattice-fit signals to main_grid_bbox

**Problem.** Tesuji-book board 7 (page 4 bbox 0) rendered shifted
one column right despite `edges.left=True`. Diagnosis:
`fit_lattice_fused` returned `origin_x=-3.44, pitch_x=30.11`. Real
left grid line lives at `x≈26` (column of T-junctions), but five
short vertical Hough segments at `x∈{0.5,1,2,3,4}` — page-binding
or scan-edge artifact outside `main_grid_bbox.x0=24` — anchored a
phantom column 0 one pitch to the left of the real grid. Stone
detector + skeleton junctions were correct; lattice anchor was
wrong by exactly one pitch.

**Fix.** Filter both segments (by midpoint) and skeleton-junction
ix centers to `main_grid_bbox` before feeding `fit_lattice_fused`,
matching the existing `filter_to_grid_bbox` step on stones. Verified
tesuji bbox 0: `origin_x=25.73`, leftmost stone now col 1 (was col
2). Cho-chikun and other tesuji-page-3 boards unaffected (origins
in the 20–27 range, all stones inside expected windows). Status:
**in use**.

---

## 2026-04-27 — edge-anchored origin/pitch (curvature-driven phase shifts)

**Problem.** On the hm2 val set, 27 of 124 problems were "changed" (pred
≠ GT). Inspection showed a dominant pattern: every detected stone in a
problem shifted by ±1 (sometimes ±2) columns, all the same direction.
Cases: hm2_0140 (left −1), hm2_0142 (right +1), hm2_0134 (left −2),
hm2_0021 (left −1), hm2_0061 (X+Y both shifted), several more. Lattice
geometry diagnostics: `edges` fired correctly for left+right+top+bottom
in most of these, but `visible_cols` came back 17 or 18 — inconsistent
with both edges being detected (must be 19). All affected images had
visible page-spine bow / barrel curvature.

**Root cause.** `fit_lattice_fused`'s 1D origin search fits a uniform-
pitch grid by minimizing snap residual against segments + stones +
junctions. On a curved board, the residual is smeared enough that
multiple phase offsets (off by whole pitches) score similarly, and the
search lands on a wrong phase. `discretize._place_window` then honors
`left=True` by setting `col_min=0`, but the lattice it received was
already misaligned, so the wrong physical column gets called col 0.
Edge bits alone weren't enough to disambiguate — what was missing was
the *position* of each edge.

**Fix.** `decide_edges` already computes the pixel position of each
accepted edge internally (via `_edge_position`, used to validate the
edge against ink-past-the-line). Expose it on `SkeletonEdgeResult.
edge_positions: dict[str, float | None]`. In `discretize_crop`, use as
hard constraints: when both edges of an axis are detected,
`pitch = (far − near) / 18` and `origin = near`; when only one edge is
detected, anchor origin to it and keep the fitted pitch. Removes the
phase ambiguity entirely on boards with ≥1 detected edge per axis.

**Result.** hm2 val: 97/124 → 120/124 exact (78.2% → 96.8%). Of the 4
remaining changed cases, 3 are extras-only (annotation-glyph false
positives — known issue, deferred) and 1 is a single missed stone
(stone-detector recall, orthogonal). No regressions on the 97
previously-exact problems. Status: **in use**.

---

## Open questions / unresolved threads

1. **What specific failure does "shit pipeline results" mean?** We have
   not pinned down a small set of concrete reproducible failures (e.g.
   "this problem in this PDF discretizes wrong, stones at coords X
   instead of Y"). Without that anchor, every fix is a theory chasing
   another theory.
2. **Is `fit_lattice` salvageable?** The hm2-page-7 failure suggests a
   scoring bug — RANSAC should prefer the larger-pitch fit when both
   are consistent with the data, or reject n_cols >> 19 outright.
   Unread.
3. **Cho-chikun board detector quality.** "858 boards on a 74-page PDF"
   is too many; suggests the synth data does not cover cho-chikun's
   visual style well enough. Worth a separate look.
4. **Edge classification on loose bboxes.** Both classical and lattice
   methods disagree on which sides are "real" vs "window-cut" when the
   YOLO bbox is loose. Probably the wrong layer to be making this call;
   it should follow from the bbox being right.

## Deferred (do not propose unprompted)

- **Background ingest (Cloud Tasks / Pub/Sub).** PDF rendering + YOLO
  is slow on Cloud Run's 2-vCPU runtime — minutes per book. User
  explicitly deferred 2026-04-25.
