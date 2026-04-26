// Centralised client for the FastAPI backend. Response and request shapes are
// mirrored from backend/src/goapp/api/<group>/schemas.py. Stones are exchanged
// with the server in board-coordinate form {col, row, color}; the rendering
// Stone in ./types uses {x, y, color} and is converted at the call site.

// ---------- Stones ----------

export type ServerStone = {
  col: number;
  row: number;
  color: string;
};

// ---------- /api/tsumego ----------

export type Collection = {
  source: string;
  count: number;
  accepted: number;
  accepted_edited: number;
  rejected: number;
  unreviewed: number;
  last_uploaded_at: string;
};

export type TsumegoProblem = {
  id: string;
  source: string;
  uploaded_at: string;
  source_board_idx: number;
  page_idx: number | null;
  bbox_idx: number | null;
  status: string; // "unreviewed" | "accepted" | "accepted_edited" | "rejected"
  image: string | null;
  black_to_play: boolean;
  stones: ServerStone[];
};

export type UpdateProblemRequest = {
  status?: string;
  stones?: ServerStone[];
  black_to_play?: boolean;
};

export type UpdateProblemResponse = {
  id: string;
  status: string;
};

// ---------- /api/pdf ----------

export type BoardBox = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  confidence: number;
};

export type BboxDetectResponse = {
  page_index: number;
  page_width: number;
  page_height: number;
  boards: BoardBox[];
};

export type BboxUploadResponse = {
  page_count: number;
};

export type BoardListItem = {
  page_idx: number;
  bbox_idx: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  confidence: number;
};

export type DiscretizedStone = {
  x: number;
  y: number;
  color: string;
  conf: number;
  col_local: number;
  row_local: number;
  col: number;
  row: number;
};

export type BoardDiscretize = {
  page_idx: number;
  bbox_idx: number;
  crop_width: number;
  crop_height: number;
  cell_size: number;
  origin_x: number;
  origin_y: number;
  visible_cols: number;
  visible_rows: number;
  col_min: number;
  row_min: number;
  edges: { left: boolean; right: boolean; top: boolean; bottom: boolean };
  stones: DiscretizedStone[];
};

export type Intersection = { x: number; y: number; conf: number };

export type BoardIntersections = {
  page_idx: number;
  bbox_idx: number;
  crop_width: number;
  crop_height: number;
  intersections: Intersection[];
};

// Streaming ingest events from `/api/pdf/ingest` (NDJSON body).
export type IngestEvent =
  | { event: 'start'; source: string; uploaded_at: string; total_pages: number }
  | { event: 'page_rendered'; page: number; total_pages: number }
  | { event: 'board_saved'; source_board_idx: number; page_idx: number; bbox_idx: number; total_saved: number }
  | { event: 'done'; source: string; total_saved: number; skipped: number }
  | { event: 'error'; detail: string };

// ---------- /api/val ----------

export type ComparisonProblem = {
  stem: string;
  source_board_idx: number;
  crop_width: number;
  crop_height: number;
  gt: ServerStone[];
  old: ServerStone[];
  new: ServerStone[];
  old_matches_gt: boolean;
  new_matches_gt: boolean;
};

export type ComparisonData = {
  val_dir: string;
  old_model: string;
  new_model: string;
  total: number;
  changed_count: number;
  problems: ComparisonProblem[];
};

export type Flip = {
  col: number;
  row: number;
  gt_color: string;
  pred_color: string;
};

export type ProblemResult = {
  stem: string;
  status: 'exact' | 'changed' | 'error';
  error?: string;
  gt_count?: number;
  pred_count?: number;
  missed?: ServerStone[];
  extra?: ServerStone[];
  flips?: Flip[];
  gt_stones?: ServerStone[];
  pred_stones?: ServerStone[];
};

export type RunResult = {
  dataset: string;
  filter_status: string;
  total: number;
  exact: number;
  changed: number;
  errors: number;
  problems: ProblemResult[];
};

// One entry in the gt-edits log. Only `length` is consumed today, but the
// shape is documented here so future readers know what the endpoint returns.
export type GtEdit = {
  timestamp: string;
  stem: string;
  source?: string;
  source_board_idx?: number;
  original_tsumego_id?: string;
  before: ServerStone[];
  after: ServerStone[];
  added: ServerStone[];
  removed: ServerStone[];
  color_flips: { col: number; row: number; from: string; to: string }[];
};

// ---------- /api/health ----------

export type Health = {
  status: 'warming' | 'ready' | 'degraded';
  version: string;
  models_ready: boolean;
  error: string | null;
};

// ---------- Internals ----------

/**
 * Throws an Error whose message is the server's `detail` field (when present),
 * otherwise the HTTP status text. Callers can `String(e)` for display.
 */
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const body = (await r.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? r.statusText);
  }
  return r.json() as Promise<T>;
}

function postJson<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

const NO_STORE: RequestInit = { cache: 'no-store' };

/** PUT a Blob with a progress callback. Uses XHR because fetch lacks
 *  request-body progress events. */
function putWithProgress(
  url: string,
  body: Blob,
  onProgress: (frac: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', url);
    xhr.setRequestHeader('Content-Type', body.type || 'application/pdf');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(xhr.statusText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(body);
  });
}

/** POST a body and parse newline-delimited JSON events as they arrive.
 *  XHR is used here too so we can both stream the response and (for the
 *  multipart case) report request-body upload progress. */
function streamNdjson<E>(
  url: string,
  headers: Record<string, string> | undefined,
  body: XMLHttpRequestBodyInit,
  onEvent: (ev: E) => void,
  onProgress?: (frac: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    if (headers) {
      for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    }
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
    }
    let offset = 0;
    xhr.onprogress = () => {
      const text = xhr.responseText ?? '';
      while (true) {
        const nl = text.indexOf('\n', offset);
        if (nl === -1) break;
        const line = text.slice(offset, nl).trim();
        offset = nl + 1;
        if (!line) continue;
        try {
          onEvent(JSON.parse(line) as E);
        } catch {
          // ignore unparseable lines
        }
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(xhr.statusText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(body);
  });
}

// Cache-busting query param for image/JSON URLs the server may overwrite
// in place (e.g. after a fresh PDF upload).
function bust(): string {
  return `_t=${Date.now()}`;
}

// ---------- Endpoints ----------

export const api = {
  health: {
    get(): Promise<Health> {
      return request<Health>('/api/health', NO_STORE);
    },
  },

  tsumego: {
    listCollections(): Promise<Collection[]> {
      return request<{ collections: Collection[] }>('/api/tsumego/collections', NO_STORE)
        .then((r) => r.collections);
    },
    deleteCollection(source: string): Promise<unknown> {
      return request(`/api/tsumego/collections/${encodeURIComponent(source)}`, { method: 'DELETE' });
    },
    listProblems(source: string): Promise<TsumegoProblem[]> {
      return request<{ problems: TsumegoProblem[] }>(
        `/api/tsumego/collections/${encodeURIComponent(source)}/problems`,
        NO_STORE,
      ).then((r) => r.problems);
    },
    getProblem(id: string): Promise<TsumegoProblem> {
      return request<TsumegoProblem>(`/api/tsumego/${id}`, NO_STORE);
    },
    deleteProblem(id: string): Promise<unknown> {
      return request(`/api/tsumego/${id}`, { method: 'DELETE' });
    },
    updateProblem(id: string, req: UpdateProblemRequest): Promise<UpdateProblemResponse> {
      return postJson<UpdateProblemResponse>(`/api/tsumego/${id}/status`, req);
    },
    imageUrl(id: string): string {
      return `/api/tsumego/${id}/image.png`;
    },
  },

  pdf: {
    uploadPdf(file: File): Promise<BboxUploadResponse> {
      const form = new FormData();
      form.append('file', file, file.name);
      return request<BboxUploadResponse>('/api/pdf/bbox-upload', { method: 'POST', body: form });
    },
    detectBboxes(pageIdx: number): Promise<BboxDetectResponse> {
      return request<BboxDetectResponse>(`/api/pdf/bbox-detect/${pageIdx}?${bust()}`, NO_STORE);
    },
    listBoards(): Promise<BoardListItem[]> {
      return request<{ total: number; boards: BoardListItem[] }>('/api/pdf/boards', NO_STORE)
        .then((r) => r.boards);
    },
    discretizeBoard(pageIdx: number, bboxIdx: number, peakThresh: number): Promise<BoardDiscretize> {
      return request<BoardDiscretize>(
        `/api/pdf/board-discretize/${pageIdx}/${bboxIdx}?peak_thresh=${peakThresh}&${bust()}`,
        NO_STORE,
      );
    },
    detectIntersections(pageIdx: number, bboxIdx: number, peakThresh: number): Promise<BoardIntersections> {
      return request<BoardIntersections>(
        `/api/pdf/board-intersections/${pageIdx}/${bboxIdx}?peak_thresh=${peakThresh}&${bust()}`,
        NO_STORE,
      );
    },
    pageImageUrl(pageIdx: number): string {
      return `/api/pdf/bbox-page/${pageIdx}.png?${bust()}`;
    },
    boardCropUrl(pageIdx: number, bboxIdx: number): string {
      return `/api/pdf/board-crop/${pageIdx}/${bboxIdx}.png?${bust()}`;
    },
    /**
     * Stream PDF ingest progress over NDJSON. Calls `onProgress` with the
     * upload fraction during the request body, and `onEvent` for each parsed
     * server event. Resolves once the response completes successfully.
     *
     * Two transports depending on what the server tells us:
     * - `signed-url`: PUT the file directly to GCS (Cloud Run's 32 MiB
     *   request-body cap doesn't apply), then POST a small JSON request to
     *   trigger the streaming ingest.
     * - `multipart`: legacy path used locally; one XHR carries both the
     *   upload and the streaming response.
     */
    async streamIngest(
      file: File,
      onProgress: (frac: number) => void,
      onEvent: (ev: IngestEvent) => void,
    ): Promise<void> {
      const plan = await postJson<{ mode: 'signed-url' | 'multipart'; upload_id?: string; url?: string }>(
        '/api/pdf/upload-url',
        { filename: file.name },
      );
      if (plan.mode === 'signed-url' && plan.url && plan.upload_id) {
        await putWithProgress(plan.url, file, onProgress);
        onProgress(1);
        await streamNdjson(
          '/api/pdf/ingest-from-upload',
          { 'Content-Type': 'application/json' },
          JSON.stringify({ upload_id: plan.upload_id, filename: file.name }),
          onEvent,
        );
        return;
      }
      const form = new FormData();
      form.append('file', file, file.name);
      await streamNdjson('/api/pdf/ingest', undefined, form, onEvent, onProgress);
    },
  },

  val: {
    getComparison(dataset: string): Promise<ComparisonData> {
      return request<ComparisonData>(`/api/val/${dataset}/comparison`, NO_STORE);
    },
    getGtEdits(dataset: string): Promise<GtEdit[]> {
      return request<GtEdit[]>(`/api/val/${dataset}/gt-edits`, NO_STORE);
    },
    updateGtStones(
      dataset: string,
      stem: string,
      stones: ServerStone[],
    ): Promise<{ ok: boolean; stem: string; stones: ServerStone[] }> {
      return postJson(`/api/val/${dataset}/problems/${stem}/stones`, { stones });
    },
    runValidation(dataset: string, status: string = 'accepted'): Promise<RunResult> {
      return request<RunResult>(`/api/val/${dataset}/run?status=${status}`);
    },
    imageUrl(dataset: string, stem: string): string {
      return `/api/val/${dataset}/images/${stem}.png`;
    },
    gtEditsUrl(dataset: string): string {
      return `/api/val/${dataset}/gt-edits`;
    },
  },
};
