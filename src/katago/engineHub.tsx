// Site-wide AI engine state: one selected analysis model, one settings modal,
// one lease, one warm-up. The provider holds the browser-engine lease for the
// whole tab lifetime (so the loaded net survives navigation instead of being
// disposed per-view) and warms the selected model as soon as the site loads.
// Views consume model/visits/batch and readiness from here; the sidebar shows
// the model name + health and opens the shared settings modal.

import {
  createContext, useContext, useEffect, useMemo, useRef, useState,
  useSyncExternalStore, type ReactNode,
} from 'react';
import {
  BROWSER_MODELS, LOCAL_MODEL, webgpuAvailable, analyzePosition,
  type AnalysisModel,
} from './webEngine';
import { katagoBackendAvailable } from '../data/katago';
import {
  getModelStatus, subscribeModelStatus, type KataGoModelStatus,
} from './engine/katago/client';
import { setEnginePrefs, setPlayDefaults } from '../data/profile';
import type { PlayDefaults } from '../data/model';
import { useAuth } from '../auth';
import { useEngineLease, useBrowserEngineHeldElsewhere, type LeaseStatus } from './engineLease';
import { Modal } from '../Modal';
import { AnalysisSettings, PlaySettings } from '../EngineSettings';
import './engineHub.css';

const DEFAULT_MODEL_ID = BROWSER_MODELS[0].id;   // b18
const isBrowserModel = (id?: string) => !!id && BROWSER_MODELS.some((m) => m.id === id);

/** Opponent settings for human-like play. Mirrors the persisted `playDefaults`,
 * which is written back (debounced) whenever any of them changes. */
const DEFAULT_PLAY: PlayDefaults = {
  rank: 'rank_9k',
  temperature: 1.0,
  moveDelay: 1,
  engine: 'browser',
  scoreMode: 'show',
  alertKind: 'behind',
  alertThreshold: 5,
  dropPoints: 5,
  dropMoves: 10,
};

export type EngineHealth = 'warming' | 'ready' | 'down' | 'blocked';

const WEBGPU_NOTE =
  'This browser has no WebGPU adapter, so the analysis net cannot run. Chrome or Edge 113+ on a machine with a supported GPU will run it; Safari and Firefox need WebGPU enabled in their settings.';

type Hub = {
  models: AnalysisModel[];
  model: AnalysisModel;
  loadedModelName: string;         // the net actually resident in the worker
  modelId: string;
  pickModel: (id: string) => void;
  visitsByModel: Record<string, number>;
  setVisits: (id: string, v: number) => void;
  visits: number;                  // selected model's playouts
  batchOverride: number | null;
  setBatchOverride: (b: number | null) => void;
  health: EngineHealth;
  healthNote: string | null;       // why the engine is down, when it is
  leaseStatus: LeaseStatus;
  engineReady: boolean;            // safe to issue analysis calls now
  play: PlayDefaults;              // human-like opponent settings, live
  setPlay: (patch: Partial<PlayDefaults>) => void;
  openSettings: () => void;
};

const HubContext = createContext<Hub | null>(null);

// eslint-disable-next-line react-refresh/only-export-components
export function useEngineHub(): Hub {
  const hub = useContext(HubContext);
  if (!hub) throw new Error('useEngineHub outside EngineHubProvider');
  return hub;
}

export function EngineHubProvider({ children }: { children: ReactNode }) {
  const { user, profile, updateProfile } = useAuth();
  const [localAvailable, setLocalAvailable] = useState<boolean | null>(null);
  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const userPicked = useRef(false);
  const [visitsByModel, setVisitsByModel] = useState<Record<string, number>>(
    () => Object.fromEntries([...BROWSER_MODELS, LOCAL_MODEL].map((m) => [m.id, m.defaultVisits])),
  );
  const [batchOverride, setBatchOverride] = useState<number | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [warmState, setWarmState] = useState<'pending' | 'ok' | 'failed'>('pending');
  const [playRaw, setPlayRaw] = useState<PlayDefaults>(DEFAULT_PLAY);
  const seededPlay = useRef(false);
  const playSaveTimer = useRef<number | null>(null);

  const localOk = localAvailable === true;
  const models = useMemo(
    () => (localOk ? [...BROWSER_MODELS, LOCAL_MODEL] : BROWSER_MODELS),
    [localOk],
  );
  const model = models.find((m) => m.id === modelId) ?? models[0];

  useEffect(() => {
    let on = true;
    katagoBackendAvailable().then((ok) => { if (on) setLocalAvailable(ok); });
    webgpuAvailable().then((ok) => { if (on) setWebgpuOk(ok); });
    return () => { on = false; };
  }, []);

  // Seed from the saved preference once backend availability is known.
  useEffect(() => {
    if (localAvailable === null || userPicked.current) return;
    const saved = localOk ? profile?.enginePrefs?.localModelId : profile?.enginePrefs?.browserModelId;
    const valid = saved === 'local' ? localOk : isBrowserModel(saved);
    setModelId(valid ? saved! : DEFAULT_MODEL_ID);
  }, [localAvailable, localOk, profile]);

  // Seed the opponent settings from the saved profile once, then local state is
  // authoritative (and is written back on every change).
  useEffect(() => {
    if (seededPlay.current || !profile?.playDefaults) return;
    seededPlay.current = true;
    setPlayRaw({ ...DEFAULT_PLAY, ...profile.playDefaults });
  }, [profile]);

  // 'local' can only apply once the native backend has answered; the saved
  // preference is kept either way, so it takes effect when the backend appears.
  const play = useMemo<PlayDefaults>(
    () => (playRaw.engine === 'local' && !localOk ? { ...playRaw, engine: 'browser' } : playRaw),
    [playRaw, localOk],
  );

  const setPlay = (patch: Partial<PlayDefaults>) => {
    setPlayRaw((prev) => {
      const next = { ...prev, ...patch };
      if (user) {
        if (playSaveTimer.current != null) clearTimeout(playSaveTimer.current);
        playSaveTimer.current = window.setTimeout(() => {
          void setPlayDefaults(user.uid, next).catch(() => {});   // best-effort
        }, 800);
      }
      return next;
    });
  };

  const pickModel = (id: string) => {
    userPicked.current = true;
    setModelId(id);
    if (!user) return;
    const next = localOk
      ? { ...profile?.enginePrefs, localModelId: id }
      : { ...profile?.enginePrefs, browserModelId: id };
    updateProfile({ enginePrefs: next });
    void setEnginePrefs(user.uid, next).catch(() => {});   // best-effort persist
  };

  // Hold the lease for the tab's lifetime: the worker keeps the net resident
  // across navigation, and a second tab shows "blocked" instead of stealing it.
  const leaseStatus = useEngineLease(model.kind === 'browser');
  const heldElsewhere = useBrowserEngineHeldElsewhere();

  // Warm the selected model as soon as it's usable: a 1-visit background query
  // forces the net fetch + GPU warm-up so the first real analysis is instant.
  const warmedFor = useRef('');
  useEffect(() => {
    let on = true;
    if (model.kind === 'local') {
      if (warmedFor.current === model.id) return;
      warmedFor.current = model.id;
      setWarmState('pending');
      katagoBackendAvailable().then((ok) => { if (on) setWarmState(ok ? 'ok' : 'failed'); });
      return () => { on = false; };
    }
    if (leaseStatus !== 'active' || warmedFor.current === model.id) return;
    warmedFor.current = model.id;
    setWarmState('pending');
    analyzePosition({
      model, stones: [], moves: [], toPlay: 'B',
      positionId: `hub:warm:${model.id}`, visits: 1, background: true,
    })
      .then(() => { if (on) setWarmState('ok'); })
      .catch(() => { if (on) { setWarmState('failed'); warmedFor.current = ''; } });
    return () => { on = false; };
  }, [model, leaseStatus]);

  // The worker's own account of its resident model outranks our bookkeeping:
  // it also covers nets other surfaces load (e.g. Play's human net).
  const worker: KataGoModelStatus = useSyncExternalStore(subscribeModelStatus, getModelStatus, getModelStatus);
  // No WebGPU adapter means the browser engine cannot run at all — a hard
  // failure that outranks every other state, since nothing else can fix it.
  const noWebgpu = model.kind === 'browser' && webgpuOk === false;
  // Blocked only when the SELECTED model needs the browser engine and can't
  // have it; a native selection stays green while another tab uses the GPU.
  const health: EngineHealth =
    noWebgpu ? 'down'
      : model.kind === 'browser' && (leaseStatus === 'waiting' || heldElsewhere) ? 'blocked'
        : worker.status === 'loading' ? 'warming'
          : worker.status === 'error' ? 'down'
            : worker.status === 'ready' ? 'ready'
              : warmState === 'pending' ? 'warming'
                : warmState === 'failed' ? 'down'
                  : 'ready';
  const healthNote = noWebgpu ? WEBGPU_NOTE
    : health === 'down' ? 'The analysis engine failed to start. Reloading the page usually clears it.'
      : null;
  const loadedModelName = worker.modelName ?? model.name;
  const engineReady = model.kind !== 'browser' || leaseStatus === 'active';

  const hub: Hub = {
    models,
    model,
    loadedModelName,
    healthNote,
    modelId: model.id,
    pickModel,
    visitsByModel,
    setVisits: (id, v) => setVisitsByModel((prev) => ({ ...prev, [id]: v })),
    visits: visitsByModel[model.id] ?? model.defaultVisits,
    batchOverride,
    setBatchOverride,
    health,
    leaseStatus,
    engineReady,
    play,
    setPlay,
    openSettings: () => setSettingsOpen(true),
  };

  return (
    <HubContext.Provider value={hub}>
      {children}
      <Modal open={settingsOpen} onClose={() => setSettingsOpen(false)} title="AI engine">
        <p className="eh-status-line">
          <HealthDot health={health} /> {loadedModelName} — {HEALTH_LABEL[health]}
          {health !== 'blocked' && heldElsewhere && (
            <span className="eh-status-note"> · browser engine in use by another tab</span>
          )}
        </p>
        {healthNote && <p className="eh-status-error">{healthNote}</p>}
        <AnalysisSettings
          models={models}
          modelId={model.id}
          onModelId={pickModel}
          visitsByModel={visitsByModel}
          onVisitsChange={hub.setVisits}
          batchOverride={batchOverride}
          onBatchOverride={setBatchOverride}
        />
        <PlaySettings play={play} onChange={setPlay} localAvailable={localOk} />
      </Modal>
    </HubContext.Provider>
  );
}

const HEALTH_LABEL: Record<EngineHealth, string> = {
  ready: 'ready',
  warming: 'warming up…',
  down: 'down',
  blocked: 'in use by another tab',
};

function HealthDot({ health }: { health: EngineHealth }) {
  return <span className={`eh-dot eh-dot-${health}`} aria-hidden="true" />;
}

/** Sidebar button: current model + health; opens the shared settings modal. */
export function EngineStatusButton() {
  const { loadedModelName, health, healthNote, openSettings } = useEngineHub();
  return (
    <button
      type="button"
      className="eh-button"
      onClick={openSettings}
      title={healthNote
        ? `AI engine: ${loadedModelName} — ${HEALTH_LABEL[health]}. ${healthNote}`
        : `AI engine: ${loadedModelName} — ${HEALTH_LABEL[health]}`}
    >
      <HealthDot health={health} />
      <span className="eh-button-name">{loadedModelName}</span>
    </button>
  );
}
