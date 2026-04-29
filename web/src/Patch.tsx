import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  api,
  type PatchAddBBox,
  type PatchPage,
  type PatchSession,
} from './api';
import './Patch.css';

type Edits = {
  // existing_problem_id values to delete
  deletes: Set<string>;
  // new bboxes keyed by page_idx
  addsByPage: Map<number, PatchAddBBox[]>;
};

function emptyEdits(): Edits {
  return { deletes: new Set(), addsByPage: new Map() };
}

function totalEdits(edits: Edits): { kept: number; deleted: number; added: number } {
  let added = 0;
  for (const v of edits.addsByPage.values()) added += v.length;
  return { kept: 0, deleted: edits.deletes.size, added };
}

export function Patch() {
  const { source: encSource = '' } = useParams();
  const source = decodeURIComponent(encSource);
  const [params] = useSearchParams();
  const sessionIdFromUrl = params.get('session');
  const navigate = useNavigate();

  const [stage, setStage] = useState<'pick' | 'starting' | 'walk' | 'confirm' | 'applying' | 'done'>(
    sessionIdFromUrl ? 'walk' : 'pick',
  );
  const [sessionId, setSessionId] = useState<string | null>(sessionIdFromUrl);
  const [session, setSession] = useState<PatchSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pageCursor, setPageCursor] = useState(0);
  const [edits, setEdits] = useState<Edits>(emptyEdits);
  const [applyResult, setApplyResult] = useState<
    { deleted: number; added: number; reindexed: number } | null
  >(null);

  // ---- start: upload PDF ----

  const onFileChosen = async (file: File) => {
    setStage('starting');
    setError(null);
    try {
      const sid = await api.pdf.startPatchSession(source, file);
      setSessionId(sid);
      navigate(`?session=${sid}`, { replace: true });
    } catch (e) {
      setError(String(e));
      setStage('pick');
    }
  };

  // ---- poll session state until ready (or done) ----

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await api.pdf.getPatchSession(sessionId);
        if (cancelled) return;
        setSession(s);
        if (s.phase === 'ready' && stage === 'starting') setStage('walk');
        if (s.phase === 'error') {
          setError(s.error || 'session failed');
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    tick();
    const interval = setInterval(() => {
      // Stop polling once we have a terminal-ish state with everything we need.
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
      // Don't fight inputs.
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

  const toggleDelete = (pid: string) => {
    setEdits((prev) => {
      const next = { ...prev, deletes: new Set(prev.deletes) };
      if (next.deletes.has(pid)) next.deletes.delete(pid);
      else next.deletes.add(pid);
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
      const adds: PatchAddBBox[] = [];
      for (const list of edits.addsByPage.values()) adds.push(...list);
      const result = await api.pdf.applyPatchSession(
        sessionId,
        Array.from(edits.deletes),
        adds,
      );
      setApplyResult(result);
      setStage('done');
    } catch (e) {
      setError(String(e));
      setStage('confirm');
    }
  };

  // ---- render ----

  if (stage === 'pick') {
    return (
      <div className="patch-root">
        <div className="patch-header">
          <Link to={`/collections/${encodeURIComponent(source)}/edit`}>← Back</Link>
          <h1>Patch detection: {source}</h1>
        </div>
        <p>
          Upload the original source PDF. We'll re-run board detection,
          then walk you through every page so you can add missed bboxes
          or delete bad ones. Existing problems' IDs are preserved.
        </p>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFileChosen(f);
          }}
        />
        {error && <p className="patch-error">{error}</p>}
      </div>
    );
  }

  if (stage === 'starting' || (stage === 'walk' && (!session || session.phase !== 'ready'))) {
    return (
      <div className="patch-root">
        <div className="patch-header">
          <h1>Patch detection: {source}</h1>
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

  if (stage === 'walk' && session && currentPage && sessionId) {
    return (
      <PatchWalker
        sessionId={sessionId}
        page={currentPage}
        pageCursor={pageCursor}
        totalPages={pages.length}
        edits={edits}
        onToggleDelete={toggleDelete}
        onAddBbox={(b) => addBbox(currentPage.page_idx, b)}
        onRemoveAdd={(i) => removeAdd(currentPage.page_idx, i)}
        onPrev={() => setPageCursor((c) => Math.max(0, c - 1))}
        onNext={() => setPageCursor((c) => Math.min(pages.length - 1, c + 1))}
        onJump={(i) => setPageCursor(i)}
        onSkipRemaining={() => setStage('confirm')}
      />
    );
  }

  if (stage === 'confirm' && session) {
    const counts = totalEdits(edits);
    const totalProblems = session.pages.reduce(
      (n, p) => n + p.bboxes.filter((b) => b.existing_problem_id).length,
      0,
    );
    const kept = totalProblems - counts.deleted;
    return (
      <div className="patch-root">
        <div className="patch-header">
          <h1>Confirm patch: {source}</h1>
        </div>
        <ul className="patch-counts">
          <li>{kept} problems kept</li>
          <li className="patch-deleted">{counts.deleted} problems to delete</li>
          <li className="patch-added">{counts.added} new bboxes to ingest</li>
        </ul>
        {session.align_warnings.length > 0 && (
          <div className="patch-warnings">
            <h3>Alignment warnings</h3>
            <ul>
              {session.align_warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="patch-actions">
          <button onClick={() => setStage('walk')}>Back to walker</button>
          <button onClick={apply} className="patch-apply-btn">Apply</button>
        </div>
        {error && <p className="patch-error">{error}</p>}
      </div>
    );
  }

  if (stage === 'applying') {
    return (
      <div className="patch-root">
        <h1>Applying…</h1>
      </div>
    );
  }

  if (stage === 'done' && applyResult) {
    return (
      <div className="patch-root">
        <h1>Patch complete</h1>
        <ul className="patch-counts">
          <li>{applyResult.deleted} problems deleted</li>
          <li>{applyResult.added} new problems ingested</li>
          <li>{applyResult.reindexed} problems reindexed</li>
        </ul>
        <p>
          Local collection updated. To push the result to prod, run:
          <code style={{ display: 'block', marginTop: '0.5rem' }}>
            python -m goapp.cli.push_collection --source "{source}"
          </code>
        </p>
        <Link to={`/collections/${encodeURIComponent(source)}/edit`}>
          Back to collection
        </Link>
      </div>
    );
  }

  return <div className="patch-root">…</div>;
}

// --- per-page walker view ---

type WalkerProps = {
  sessionId: string;
  page: PatchPage;
  pageCursor: number;
  totalPages: number;
  edits: Edits;
  onToggleDelete: (pid: string) => void;
  onAddBbox: (b: PatchAddBBox) => void;
  onRemoveAdd: (addIdx: number) => void;
  onPrev: () => void;
  onNext: () => void;
  onJump: (i: number) => void;
  onSkipRemaining: () => void;
};

function PatchWalker(props: WalkerProps) {
  const { sessionId, page, edits, pageCursor, totalPages } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  // `moved` distinguishes a click from a drag at mouseup time. Click and
  // drag both start with mousedown — we don't know which until the mouse
  // either stays put (click → toggle/remove) or moves (drag → add bbox).
  const [drag, setDrag] = useState<{
    x0: number; y0: number; x1: number; y1: number;
    moved: boolean;
    cxStart: number; cyStart: number;
    onClickPid?: string;          // existing bbox under cursor at mousedown
    onClickAddIdx?: number;       // added bbox under cursor at mousedown
  } | null>(null);
  const adds = edits.addsByPage.get(page.page_idx) ?? [];

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
    const pidAttr = target.dataset.bboxPid;
    const addIdxAttr = target.dataset.bboxAddIdx;
    const p = toImageCoords(e);
    setDrag({
      x0: p.x, y0: p.y, x1: p.x, y1: p.y,
      moved: false,
      cxStart: e.clientX, cyStart: e.clientY,
      onClickPid: pidAttr,
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
      // Treated as a click on whatever was under the cursor at mousedown.
      if (drag.onClickPid) props.onToggleDelete(drag.onClickPid);
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
        <span>Page {page.page_idx + 1} ({pageCursor + 1}/{totalPages})</span>
        <div className="patch-walker-actions">
          <button onClick={props.onPrev} disabled={pageCursor === 0}>← Prev</button>
          <button onClick={props.onNext} disabled={pageCursor === totalPages - 1}>Next →</button>
          <button onClick={props.onSkipRemaining} className="patch-skip-btn">
            Skip remaining → Confirm
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
          const pid = b.existing_problem_id;
          const isDeleted = pid ? edits.deletes.has(pid) : false;
          const cls = pid
            ? (isDeleted ? 'patch-bbox patch-bbox-deleted' : 'patch-bbox patch-bbox-existing')
            : 'patch-bbox patch-bbox-orphan';
          return (
            <div
              key={`b-${b.bbox_idx}`}
              className={cls}
              data-bbox-pid={pid ?? undefined}
              style={proj}
              title={pid ? (isDeleted ? 'Click to keep · drag to add' : 'Click to delete · drag to add') : 'No matching saved problem'}
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
        Click an existing bbox to mark it for deletion (red).
        Drag to add a new bbox (blue) — works anywhere, including over a deleted bbox.
        Enter or → next page, ← previous page.
      </p>
    </div>
  );
}
