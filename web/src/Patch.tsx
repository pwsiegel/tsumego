import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  api,
  type PatchAddBBox,
  type PatchPage,
  type PatchSession,
  type PatchSkipBBox,
} from './api';
import './Patch.css';

type Edits = {
  // (page_idx, bbox_idx) of detected bboxes the user wants to drop
  skipsByPage: Map<number, Set<number>>;
  // user-drawn bboxes keyed by page_idx
  addsByPage: Map<number, PatchAddBBox[]>;
};

function emptyEdits(): Edits {
  return { skipsByPage: new Map(), addsByPage: new Map() };
}

function totalEdits(edits: Edits): { skipped: number; added: number } {
  let skipped = 0;
  for (const v of edits.skipsByPage.values()) skipped += v.size;
  let added = 0;
  for (const v of edits.addsByPage.values()) added += v.length;
  return { skipped, added };
}

export function Patch() {
  const { sessionId = '' } = useParams();
  const navigate = useNavigate();

  const [stage, setStage] = useState<'walk' | 'confirm' | 'applying'>('walk');
  const [session, setSession] = useState<PatchSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pageCursor, setPageCursor] = useState(0);
  const [edits, setEdits] = useState<Edits>(emptyEdits);

  // ---- poll session state until ready ----

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await api.pdf.getPatchSession(sessionId);
        if (cancelled) return;
        setSession(s);
        if (s.phase === 'error') setError(s.error || 'session failed');
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    tick();
    const interval = setInterval(() => {
      if (session && (session.phase === 'ready' || session.phase === 'done' || session.phase === 'error')) {
        return;
      }
      tick();
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, session?.phase]);

  // ---- keyboard nav in walker ----

  const pages = session?.pages ?? [];
  const currentPage = pages[pageCursor];

  useEffect(() => {
    if (stage !== 'walk') return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (e.key === 'Enter' || e.key === 'ArrowRight') {
        e.preventDefault();
        setPageCursor((c) => Math.min(pages.length - 1, c + 1));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setPageCursor((c) => Math.max(0, c - 1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stage, pages.length]);

  // ---- per-page edit handlers ----

  const toggleSkip = (page_idx: number, bbox_idx: number) => {
    setEdits((prev) => {
      const next = { ...prev, skipsByPage: new Map(prev.skipsByPage) };
      const set = new Set(next.skipsByPage.get(page_idx) ?? []);
      if (set.has(bbox_idx)) set.delete(bbox_idx);
      else set.add(bbox_idx);
      if (set.size === 0) next.skipsByPage.delete(page_idx);
      else next.skipsByPage.set(page_idx, set);
      return next;
    });
  };

  const addBbox = (page_idx: number, b: PatchAddBBox) => {
    setEdits((prev) => {
      const next = { ...prev, addsByPage: new Map(prev.addsByPage) };
      const list = next.addsByPage.get(page_idx)?.slice() ?? [];
      list.push(b);
      next.addsByPage.set(page_idx, list);
      return next;
    });
  };

  const removeAdd = (page_idx: number, addIdx: number) => {
    setEdits((prev) => {
      const next = { ...prev, addsByPage: new Map(prev.addsByPage) };
      const list = next.addsByPage.get(page_idx)?.slice() ?? [];
      list.splice(addIdx, 1);
      if (list.length === 0) next.addsByPage.delete(page_idx);
      else next.addsByPage.set(page_idx, list);
      return next;
    });
  };

  const apply = async () => {
    if (!sessionId) return;
    setStage('applying');
    setError(null);
    try {
      const skip: PatchSkipBBox[] = [];
      for (const [page_idx, set] of edits.skipsByPage.entries()) {
        for (const bbox_idx of set) skip.push({ page_idx, bbox_idx });
      }
      const adds: PatchAddBBox[] = [];
      for (const list of edits.addsByPage.values()) adds.push(...list);
      // Apply runs in the background — bounce home, where the progress
      // card picks up via patch-sessions polling.
      await api.pdf.applyPatchSession(sessionId, skip, adds);
      navigate('/');
    } catch (e) {
      setError(String(e));
      setStage('confirm');
    }
  };

  // ---- render ----

  if (!sessionId) {
    return (
      <div className="patch-root">
        <p>Missing session id.</p>
        <Link to="/upload">Upload a PDF</Link>
      </div>
    );
  }

  if (!session || session.phase !== 'ready') {
    return (
      <div className="patch-root">
        <div className="patch-header">
          <h1>Detecting boards{session?.source ? `: ${session.source}` : ''}</h1>
        </div>
        <p>
          {session
            ? `${session.phase}: ${session.pages_rendered}/${session.total_pages ?? '?'} pages`
            : 'starting…'}
        </p>
        {error && <p className="patch-error">{error}</p>}
      </div>
    );
  }

  if (stage === 'walk' && currentPage) {
    return (
      <PatchWalker
        sessionId={sessionId}
        source={session.source}
        page={currentPage}
        pageCursor={pageCursor}
        totalPages={pages.length}
        edits={edits}
        onToggleSkip={(bbox_idx) => toggleSkip(currentPage.page_idx, bbox_idx)}
        onAddBbox={(b) => addBbox(currentPage.page_idx, b)}
        onRemoveAdd={(i) => removeAdd(currentPage.page_idx, i)}
        onPrev={() => setPageCursor((c) => Math.max(0, c - 1))}
        onNext={() => setPageCursor((c) => Math.min(pages.length - 1, c + 1))}
        onSkipRemaining={() => setStage('confirm')}
      />
    );
  }

  if (stage === 'confirm') {
    const counts = totalEdits(edits);
    const totalDetected = session.pages.reduce((n, p) => n + p.bboxes.length, 0);
    const kept = totalDetected - counts.skipped;
    return (
      <div className="patch-root">
        <div className="patch-header">
          <h1>Confirm ingest: {session.source}</h1>
        </div>
        <ul className="patch-counts">
          <li>{kept} detected boards to ingest</li>
          <li className="patch-deleted">{counts.skipped} detected boards skipped</li>
          <li className="patch-added">{counts.added} added boards to ingest</li>
        </ul>
        <div className="patch-actions">
          <button onClick={() => setStage('walk')}>Back to walker</button>
          <button onClick={apply} className="patch-apply-btn">
            Ingest {kept + counts.added}
          </button>
        </div>
        {error && <p className="patch-error">{error}</p>}
      </div>
    );
  }

  if (stage === 'applying') {
    return (
      <div className="patch-root">
        <h1>Starting ingest…</h1>
      </div>
    );
  }

  return <div className="patch-root">…</div>;
}

// --- per-page walker view ---

type WalkerProps = {
  sessionId: string;
  source: string;
  page: PatchPage;
  pageCursor: number;
  totalPages: number;
  edits: Edits;
  onToggleSkip: (bbox_idx: number) => void;
  onAddBbox: (b: PatchAddBBox) => void;
  onRemoveAdd: (addIdx: number) => void;
  onPrev: () => void;
  onNext: () => void;
  onSkipRemaining: () => void;
};

function PatchWalker(props: WalkerProps) {
  const { sessionId, source, page, edits, pageCursor, totalPages } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  // `moved` distinguishes a click from a drag at mouseup time. Click and
  // drag both start with mousedown — we don't know which until the mouse
  // either stays put (click → toggle/remove) or moves (drag → add bbox).
  const [drag, setDrag] = useState<{
    x0: number; y0: number; x1: number; y1: number;
    moved: boolean;
    cxStart: number; cyStart: number;
    onClickBboxIdx?: number;     // detected bbox under cursor at mousedown
    onClickAddIdx?: number;      // added bbox under cursor at mousedown
  } | null>(null);
  const adds = edits.addsByPage.get(page.page_idx) ?? [];
  const skips = edits.skipsByPage.get(page.page_idx) ?? new Set<number>();

  // Map between rendered DOM coords and image-pixel coords.
  const imageRef = useRef<HTMLImageElement>(null);
  const scale = (): { sx: number; sy: number; ox: number; oy: number } => {
    const img = imageRef.current;
    if (!img || !page.image_w || !page.image_h) return { sx: 1, sy: 1, ox: 0, oy: 0 };
    const rect = img.getBoundingClientRect();
    return {
      sx: page.image_w / rect.width,
      sy: page.image_h / rect.height,
      ox: rect.left,
      oy: rect.top,
    };
  };
  const toImageCoords = (e: { clientX: number; clientY: number }) => {
    const { sx, sy, ox, oy } = scale();
    return { x: (e.clientX - ox) * sx, y: (e.clientY - oy) * sy };
  };

  // 4 DOM pixels = a click; anything beyond is a drag.
  const DRAG_THRESHOLD_PX = 4;

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    const bboxIdxAttr = target.dataset.bboxIdx;
    const addIdxAttr = target.dataset.bboxAddIdx;
    const p = toImageCoords(e);
    setDrag({
      x0: p.x, y0: p.y, x1: p.x, y1: p.y,
      moved: false,
      cxStart: e.clientX, cyStart: e.clientY,
      onClickBboxIdx: bboxIdxAttr ? parseInt(bboxIdxAttr, 10) : undefined,
      onClickAddIdx: addIdxAttr ? parseInt(addIdxAttr, 10) : undefined,
    });
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!drag) return;
    const p = toImageCoords(e);
    const moved = drag.moved
      || Math.abs(e.clientX - drag.cxStart) > DRAG_THRESHOLD_PX
      || Math.abs(e.clientY - drag.cyStart) > DRAG_THRESHOLD_PX;
    setDrag({ ...drag, x1: p.x, y1: p.y, moved });
  };
  const onMouseUp = () => {
    if (!drag) return;
    if (!drag.moved) {
      if (drag.onClickBboxIdx !== undefined) props.onToggleSkip(drag.onClickBboxIdx);
      else if (drag.onClickAddIdx !== undefined) props.onRemoveAdd(drag.onClickAddIdx);
      setDrag(null);
      return;
    }
    const x0 = Math.min(drag.x0, drag.x1);
    const y0 = Math.min(drag.y0, drag.y1);
    const x1 = Math.max(drag.x0, drag.x1);
    const y1 = Math.max(drag.y0, drag.y1);
    if (x1 - x0 > 20 && y1 - y0 > 20) {
      props.onAddBbox({
        page_idx: page.page_idx,
        x0: Math.round(x0), y0: Math.round(y0),
        x1: Math.round(x1), y1: Math.round(y1),
      });
    }
    setDrag(null);
  };

  // Project image-pixel coords into rendered DOM-pixel coords.
  const project = (b: { x0: number; y0: number; x1: number; y1: number }) => {
    const { sx, sy } = scale();
    if (!sx || !sy) return null;
    return {
      left: b.x0 / sx,
      top: b.y0 / sy,
      width: (b.x1 - b.x0) / sx,
      height: (b.y1 - b.y0) / sy,
    };
  };

  // Force a re-render after the image loads so projection uses real dimensions.
  const [imgLoaded, setImgLoaded] = useState(false);
  useMemo(() => setImgLoaded(false), [page.page_idx]);

  return (
    <div className="patch-root">
      <div className="patch-header">
        <span>
          {source} · page {page.page_idx + 1} ({pageCursor + 1}/{totalPages})
        </span>
        <div className="patch-walker-actions">
          <button onClick={props.onPrev} disabled={pageCursor === 0}>← Prev</button>
          <button onClick={props.onNext} disabled={pageCursor === totalPages - 1}>Next →</button>
          <button onClick={props.onSkipRemaining} className="patch-skip-btn">
            Done → Confirm
          </button>
        </div>
      </div>
      <div
        className="patch-canvas"
        ref={containerRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => setDrag(null)}
      >
        <img
          ref={imageRef}
          src={api.pdf.patchPageImageUrl(sessionId, page.page_idx)}
          alt={`page ${page.page_idx + 1}`}
          onLoad={() => setImgLoaded(true)}
          draggable={false}
        />
        {imgLoaded && page.bboxes.map((b) => {
          const proj = project(b);
          if (!proj) return null;
          const skipped = skips.has(b.bbox_idx);
          const cls = skipped
            ? 'patch-bbox patch-bbox-deleted'
            : 'patch-bbox patch-bbox-existing';
          return (
            <div
              key={`b-${b.bbox_idx}`}
              className={cls}
              data-bbox-idx={b.bbox_idx}
              style={proj}
              title={skipped ? 'Click to keep · drag to add' : 'Click to skip · drag to add'}
            />
          );
        })}
        {imgLoaded && adds.map((a, i) => {
          const proj = project(a);
          if (!proj) return null;
          return (
            <div
              key={`a-${i}`}
              className="patch-bbox patch-bbox-added"
              data-bbox-add-idx={i}
              style={proj}
              title="Click to remove this added bbox · drag to add another"
            />
          );
        })}
        {drag && imgLoaded && (() => {
          const proj = project({
            x0: Math.min(drag.x0, drag.x1),
            y0: Math.min(drag.y0, drag.y1),
            x1: Math.max(drag.x0, drag.x1),
            y1: Math.max(drag.y0, drag.y1),
          });
          if (!proj) return null;
          return (
            <div
              className="patch-bbox patch-bbox-drag"
              data-bbox-kind="drag"
              style={proj}
            />
          );
        })()}
      </div>
      <p className="patch-help">
        Click a green bbox to skip it (red). Drag to add a new bbox (blue).
        Enter or → next page, ← previous page.
      </p>
    </div>
  );
}
