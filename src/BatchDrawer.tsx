import { useEffect, useState } from 'react';
import { useAuth } from './auth';
import { useBatch } from './batch';
import { ProblemCard } from './ProblemCard';
import { toStones } from './stones';
import { problemIndex, type ProblemIndex } from './data/library';
import { removeFromBatch, sendBatch } from './data/study';
import { listTeachers } from './data/links';
import type { AttemptDoc, UserDoc } from './data/model';
import './BatchDrawer.css';

export function BatchDrawer() {
  const { user } = useAuth();
  const uid = user!.uid;
  const { batch, refresh } = useBatch();
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState<ProblemIndex | null>(null);
  const [teachers, setTeachers] = useState<UserDoc[] | null>(null);
  const [picked, setPicked] = useState('');
  const [sending, setSending] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => { problemIndex().then(setIndex); }, []);
  useEffect(() => {
    listTeachers(uid).then((ts) => { setTeachers(ts); setPicked((p) => p || ts[0]?.uid || ''); });
  }, [uid]);
  // Reserve content space + clear the modal over the drawer while open.
  useEffect(() => {
    document.body.classList.toggle('drawer-open', open);
    return () => document.body.classList.remove('drawer-open');
  }, [open]);

  const send = async () => {
    if (!picked || batch.length === 0) return;
    setSending(true);
    try {
      const n = batch.length;
      await sendBatch(uid, picked);
      refresh();
      setFlash(`Submitted ${n} problem${n === 1 ? '' : 's'}.`);
      setTimeout(() => setFlash(null), 2500);
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <button type="button" className={`batch-toggle${open ? ' open' : ''}`} onClick={() => setOpen((o) => !o)}>
        {open ? '›' : '‹'} Current submission ({batch.length})
      </button>
      <aside className={`batch-drawer${open ? ' open' : ''}`} aria-hidden={!open}>
        <div className="batch-drawer-head">
          <h3>Current submission</h3>
          <button type="button" className="batch-close" onClick={() => setOpen(false)} aria-label="Collapse">×</button>
        </div>
        <div className="batch-drawer-body">
          {batch.length === 0
            ? <p className="dim">Nothing here yet. Solve a problem and hit “Add to submission” to collect it.</p>
            : <ul className="problem-card-grid sm">
                {batch.map((a) => <BatchRow key={a.id} attempt={a} index={index} onRemove={() => removeFromBatch(uid, a.problemId).then(refresh)} />)}
              </ul>}
        </div>
        {batch.length > 0 && (
          <div className="batch-drawer-foot">
            {teachers && teachers.length > 0 ? (
              <>
                <select value={picked} onChange={(e) => setPicked(e.target.value)} disabled={sending}>
                  {teachers.map((t) => <option key={t.uid} value={t.uid}>{t.displayName}</option>)}
                </select>
                <button className="batch-send" onClick={send} disabled={sending || !picked}>
                  {sending ? 'Sending…' : 'Send'}
                </button>
              </>
            ) : <p className="dim">Link a teacher to submit.</p>}
            {flash && <p className="batch-flash">{flash}</p>}
          </div>
        )}
      </aside>
    </>
  );
}

function BatchRow({ attempt, index, onRemove }: { attempt: AttemptDoc; index: ProblemIndex | null; onRemove: () => void }) {
  const problem = index?.byId.get(attempt.problemId) ?? null;
  return (
    <li className="problem-card-cell">
      <button type="button" className="problem-card-remove" onClick={onRemove} aria-label="Remove">×</button>
      <ProblemCard
        stones={problem ? toStones(problem.stones) : []}
        moves={attempt.moves.map((m) => ({ x: m.col, y: m.row }))}
        collection={problem?.collection ?? attempt.collection}
        number={problem ? problem.source_board_idx + 1 : undefined}
      />
    </li>
  );
}
