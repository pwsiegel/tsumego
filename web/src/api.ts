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

export type IngestStreamEvent =
  | { event: 'start'; stage: 'render' | 'detect'; total: number }
  | { event: 'page-rendered'; index: number }
  | { event: 'page-detected'; index: number; boards: BoardListItem[] }
  | { event: 'done'; page_count: number; board_count: number }
  | { event: 'error'; message: string };

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

export type StoneCenter = { x: number; y: number; color: string };

export type FusedLattice = {
  pitch_x: number | null;
  pitch_y: number | null;
  origin_x: number | null;
  origin_y: number | null;
  edges: { left: boolean; right: boolean; top: boolean; bottom: boolean };
};

export type Segment = { x1: number; y1: number; x2: number; y2: number };

export type BoardIntersections = {
  page_idx: number;
  bbox_idx: number;
  crop_width: number;
  crop_height: number;
  stones: StoneCenter[];
  segments: Segment[];
  fused_lattice: FusedLattice | null;
  skeleton_junctions: Junction[];
};

export type Junction = {
  x: number;
  y: number;
  kind: 'T' | 'L' | '+' | 'I' | '?';
  arms: number;
  outward: string[];
};

export type SideTally = { t: number; l: number; total: number };

export type StoneEdgeClass = {
  x: number;
  y: number;
  r: number;
  color: 'B' | 'W';
  sides: { N: boolean; E: boolean; S: boolean; W: boolean };
};

export type BoardTJunctionEdges = {
  page_idx: number;
  bbox_idx: number;
  crop_width: number;
  crop_height: number;
  segments: Segment[];
  junctions: Junction[];
  sides: { left: SideTally; right: SideTally; top: SideTally; bottom: SideTally };
  edges: { left: boolean; right: boolean; top: boolean; bottom: boolean };
  stone_edges: StoneEdgeClass[];
};

export type IngestJobPhase = 'rendering' | 'detecting' | 'done' | 'error';

export type IngestJob = {
  job_id: string;
  source: string;
  phase: IngestJobPhase;
  started_at: string;
  updated_at: string;
  total_pages: number | null;
  pages_rendered: number;
  pages_detected: number;
  total_saved: number;
  skipped: number;
  error: string | null;
  stalled: boolean;
};

// ---------- /api/study + /api/teacher ----------

export type Move = { col: number; row: number };

export type Review = { verdict: 'correct' | 'incorrect'; reviewed_at: string };

/** Student-facing attempt: full per-teacher reviews map. */
export type Attempt = {
  id: string;
  problem_id: string;
  moves: Move[];
  submitted_at: string;
  sent_to: string[];
  sent_at: string | null;
  reviews: Record<string, Review>;
  acked_at: string | null;
};

export type SubmissionState = 'pending' | 'returned' | 'acked';

export type Submission = {
  sent_at: string;
  teacher_id: string;
  state: SubmissionState;
  items: AttemptWithProblem[];
};

/** Teacher-facing attempt: only this teacher's review is exposed. */
export type TeacherAttempt = {
  id: string;
  problem_id: string;
  moves: Move[];
  submitted_at: string;
  sent_at: string | null;
  review: Review | null;
};

export type ProblemSummary = {
  id: string;
  source: string;
  source_board_idx: number;
  black_to_play: boolean;
  stones: ServerStone[];
  has_image: boolean;
};

export type AttemptWithProblem = {
  attempt: Attempt;
  problem: ProblemSummary;
};

export type TeacherAttemptWithProblem = {
  attempt: TeacherAttempt;
  problem: ProblemSummary;
};

export type Teacher = {
  id: string;
  label: string;
  created_at: string;
  token: string;
  url: string;
};

export type TeacherMe = {
  id: string;
  label: string;
  student: string;       // raw uid
  student_name: string;  // configured display name (falls back to student)
};

export type Profile = { display_name: string | null };

export type ProblemStatus = {
  last_verdict: 'correct' | 'incorrect' | null;
};

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

export type ValStreamEvent =
  | { event: 'start'; total: number; dataset: string; filter_status: string }
  | { event: 'problem'; result: ProblemResult }
  | { event: 'done'; exact: number; changed: number; errors: number };

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

/** POST a body with upload-progress reporting; resolves with parsed JSON. */
function postWithProgress<T>(
  url: string,
  body: XMLHttpRequestBodyInit,
  onProgress: (frac: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch (e) {
          reject(e);
        }
      } else {
        let detail: string | undefined;
        try {
          detail = (JSON.parse(xhr.responseText) as { detail?: string }).detail;
        } catch {
          // ignore
        }
        reject(new Error(detail ?? xhr.statusText ?? `HTTP ${xhr.status}`));
      }
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
    renameCollection(source: string, new_source: string): Promise<{ old_source: string; new_source: string; renamed: number }> {
      return postJson(
        `/api/tsumego/collections/${encodeURIComponent(source)}/rename`,
        { new_source },
      );
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
    /** Streaming upload + render + board detection. Calls `onEvent` for
     * each NDJSON event (one per rendered/detected page plus phase
     * markers). Resolves on stream close. */
    async uploadPdfStream(
      file: File,
      onEvent: (event: IngestStreamEvent) => void,
      opts: { signal?: AbortSignal } = {},
    ): Promise<void> {
      const form = new FormData();
      form.append('file', file, file.name);
      const res = await fetch('/api/pdf/bbox-ingest-stream', {
        method: 'POST', body: form, signal: opts.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`upload stream failed: ${res.status} ${res.statusText}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl = buf.indexOf('\n');
        while (nl !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) onEvent(JSON.parse(line) as IngestStreamEvent);
          nl = buf.indexOf('\n');
        }
      }
      const tail = buf.trim();
      if (tail) onEvent(JSON.parse(tail) as IngestStreamEvent);
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
    detectIntersections(
      pageIdx: number, bboxIdx: number, peakThresh: number,
    ): Promise<BoardIntersections> {
      return request<BoardIntersections>(
        `/api/pdf/board-intersections/${pageIdx}/${bboxIdx}?peak_thresh=${peakThresh}&${bust()}`,
        NO_STORE,
      );
    },
    detectTJunctionEdges(
      pageIdx: number, bboxIdx: number, peakThresh: number,
    ): Promise<BoardTJunctionEdges> {
      return request<BoardTJunctionEdges>(
        `/api/pdf/board-tjunctions/${pageIdx}/${bboxIdx}?peak_thresh=${peakThresh}&${bust()}`,
        NO_STORE,
      );
    },
    pageImageUrl(pageIdx: number): string {
      return `/api/pdf/bbox-page/${pageIdx}.png?${bust()}`;
    },
    boardCropUrl(pageIdx: number, bboxIdx: number): string {
      return `/api/pdf/board-crop/${pageIdx}/${bboxIdx}.png?${bust()}`;
    },
    boardCleanedUrl(pageIdx: number, bboxIdx: number): string {
      return `/api/pdf/board-cleaned/${pageIdx}/${bboxIdx}.png?${bust()}`;
    },
    boardSkeletonUrl(pageIdx: number, bboxIdx: number): string {
      return `/api/pdf/board-skeleton/${pageIdx}/${bboxIdx}.png?${bust()}`;
    },
    /** Upload the PDF and kick off a background ingest job. Resolves
     * with the new job_id once the server has staged the file and
     * scheduled the work; progress is then visible via `listJobs`.
     *
     * `onProgress` receives the upload fraction (0..1) so the UI can
     * show a progress bar during the slow part on cloud (the upload
     * itself); the server-side rendering/detection happens in the
     * background after this resolves. */
    async startIngest(
      file: File,
      onProgress: (frac: number) => void,
    ): Promise<string> {
      const plan = await postJson<{ mode: 'signed-url' | 'multipart'; upload_id?: string; url?: string }>(
        '/api/pdf/upload-url',
        { filename: file.name },
      );
      if (plan.mode === 'signed-url' && plan.url && plan.upload_id) {
        await putWithProgress(plan.url, file, onProgress);
        onProgress(1);
        const r = await postJson<{ job_id: string }>(
          '/api/pdf/ingest-from-upload',
          { upload_id: plan.upload_id, filename: file.name },
        );
        return r.job_id;
      }
      const form = new FormData();
      form.append('file', file, file.name);
      const r = await postWithProgress<{ job_id: string }>(
        '/api/pdf/ingest', form, onProgress,
      );
      return r.job_id;
    },
    listJobs(): Promise<IngestJob[]> {
      return request<{ jobs: IngestJob[] }>('/api/pdf/jobs', NO_STORE)
        .then((r) => r.jobs);
    },
    restartJob(job_id: string): Promise<{ job_id: string }> {
      return postJson<{ job_id: string }>(
        `/api/pdf/jobs/${encodeURIComponent(job_id)}/restart`, {},
      );
    },
    async dismissJob(job_id: string): Promise<void> {
      const res = await fetch(
        `/api/pdf/jobs/${encodeURIComponent(job_id)}`,
        { method: 'DELETE' },
      );
      if (!res.ok) throw new Error(`dismiss failed: ${res.status}`);
    },
  },

  study: {
    submitAttempt(problem_id: string, moves: Move[]): Promise<Attempt> {
      return postJson<Attempt>('/api/study/attempts', { problem_id, moves });
    },
    listAttempts(problem_id: string): Promise<Attempt[]> {
      return request<{ attempts: Attempt[] }>(
        `/api/study/problems/${problem_id}/attempts`, NO_STORE,
      ).then((r) => r.attempts);
    },
    listReviewed(): Promise<AttemptWithProblem[]> {
      return request<{ items: AttemptWithProblem[] }>('/api/study/reviewed', NO_STORE)
        .then((r) => r.items);
    },
    problemStatuses(): Promise<Record<string, ProblemStatus>> {
      return request<{ statuses: Record<string, ProblemStatus> }>(
        '/api/study/problem-status', NO_STORE,
      ).then((r) => r.statuses);
    },
    getBatch(): Promise<AttemptWithProblem[]> {
      return request<{ items: AttemptWithProblem[] }>('/api/study/batch', NO_STORE)
        .then((r) => r.items);
    },
    sendBatch(teacher_id: string): Promise<{ sent_count: number; teacher_id: string; sent_at: string }> {
      return postJson<{ sent_count: number; teacher_id: string; sent_at: string }>(
        '/api/study/batch/send', { teacher_id },
      );
    },
    listSubmissions(): Promise<Submission[]> {
      return request<{ submissions: Submission[] }>('/api/study/submissions', NO_STORE)
        .then((r) => r.submissions);
    },
    getSubmission(sent_at: string): Promise<Submission> {
      return request<Submission>(
        `/api/study/submissions/${encodeURIComponent(sent_at)}`, NO_STORE,
      );
    },
    ackSubmission(sent_at: string): Promise<{ sent_at: string; acked_count: number }> {
      return postJson<{ sent_at: string; acked_count: number }>(
        `/api/study/submissions/${encodeURIComponent(sent_at)}/ack`, {},
      );
    },
    listTeachers(): Promise<Teacher[]> {
      return request<{ teachers: Teacher[] }>('/api/study/teachers', NO_STORE)
        .then((r) => r.teachers);
    },
    createTeacher(label: string): Promise<Teacher> {
      return postJson<Teacher>('/api/study/teachers', { label });
    },
    updateTeacher(id: string, label: string): Promise<Teacher> {
      return request<Teacher>(`/api/study/teachers/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
    },
    deleteTeacher(id: string): Promise<void> {
      return request<unknown>(`/api/study/teachers/${id}`, { method: 'DELETE' })
        .then(() => undefined);
    },
    getProfile(): Promise<Profile> {
      return request<Profile>('/api/study/profile', NO_STORE);
    },
    updateProfile(p: Partial<Profile>): Promise<Profile> {
      return request<Profile>('/api/study/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(p),
      });
    },
  },

  teacher: {
    me(token: string): Promise<TeacherMe> {
      return request<TeacherMe>(`/api/teacher/${token}/me`, NO_STORE);
    },
    queue(token: string): Promise<TeacherAttemptWithProblem[]> {
      return request<{ items: TeacherAttemptWithProblem[] }>(
        `/api/teacher/${token}/queue`, NO_STORE,
      ).then((r) => r.items);
    },
    reviewed(token: string): Promise<TeacherAttemptWithProblem[]> {
      return request<{ items: TeacherAttemptWithProblem[] }>(
        `/api/teacher/${token}/reviewed`, NO_STORE,
      ).then((r) => r.items);
    },
    getAttempt(token: string, attempt_id: string): Promise<TeacherAttemptWithProblem> {
      return request<TeacherAttemptWithProblem>(
        `/api/teacher/${token}/attempts/${attempt_id}`, NO_STORE,
      );
    },
    review(token: string, attempt_id: string, verdict: 'correct' | 'incorrect'): Promise<TeacherAttempt> {
      return postJson<TeacherAttempt>(
        `/api/teacher/${token}/attempts/${attempt_id}/review`,
        { verdict },
      );
    },
    problemImageUrl(token: string, problem_id: string): string {
      return `/api/teacher/${token}/problems/${problem_id}/image.png`;
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
    /** Stream validation results as NDJSON, one event per line. The
     * caller's `onEvent` is called as each line lands so a progress bar
     * can tick per problem; the returned promise resolves on stream
     * close. Pass an AbortSignal to cancel mid-flight (e.g. when the
     * user navigates away). */
    async runValidationStream(
      dataset: string,
      onEvent: (event: ValStreamEvent) => void,
      opts: { status?: string; signal?: AbortSignal } = {},
    ): Promise<void> {
      const status = opts.status ?? 'accepted';
      const res = await fetch(
        `/api/val/${dataset}/run?status=${status}`,
        { signal: opts.signal },
      );
      if (!res.ok || !res.body) {
        throw new Error(`validation stream failed: ${res.status} ${res.statusText}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl = buf.indexOf('\n');
        while (nl !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) onEvent(JSON.parse(line) as ValStreamEvent);
          nl = buf.indexOf('\n');
        }
      }
      const tail = buf.trim();
      if (tail) onEvent(JSON.parse(tail) as ValStreamEvent);
    },
    imageUrl(dataset: string, stem: string): string {
      return `/api/val/${dataset}/images/${stem}.png`;
    },
    gtEditsUrl(dataset: string): string {
      return `/api/val/${dataset}/gt-edits`;
    },
  },
};
