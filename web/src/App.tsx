import { useState } from 'react';
import { BboxTest } from './BboxTest';
import { Board } from './Board';
import { StoneTest } from './StoneTest';
import type { Color, Stone } from './types';
import { other } from './types';
import './App.css';

type Mode = 'setup' | 'play';
type View = 'board' | 'bboxTest' | 'stoneTest';

function App() {
  const [stones, setStones] = useState<Stone[]>([]);
  const [mode, setMode] = useState<Mode>('setup');
  const [view, setView] = useState<View>('board');
  const [firstPlayer, setFirstPlayer] = useState<Color>('B');
  const [setupColor, setSetupColor] = useState<Color>('B');

  const numberedMoves = stones.filter((s) => s.number !== undefined);
  const nextNumber = numberedMoves.length + 1;
  const nextColor: Color =
    numberedMoves.length === 0
      ? firstPlayer
      : other(numberedMoves[numberedMoves.length - 1].color);

  const handlePlay = (x: number, y: number) => {
    if (mode === 'setup') {
      setStones((prev) => [...prev, { x, y, color: setupColor }]);
    } else {
      setStones((prev) => [
        ...prev,
        { x, y, color: nextColor, number: nextNumber },
      ]);
    }
  };

  const handleUndo = () => {
    setStones((prev) => prev.slice(0, -1));
  };

  const handleReset = () => {
    if (confirm('Clear the board?')) setStones([]);
  };

  const handleClearMoves = () => {
    setStones((prev) => prev.filter((s) => s.number === undefined));
  };

  return (
    <div className="app">
      <aside className="controls">
        <h1>Go problem</h1>

        <section>
          <h2>Mode</h2>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                checked={mode === 'setup'}
                onChange={() => setMode('setup')}
              />
              Setup (place initial stones)
            </label>
            <label>
              <input
                type="radio"
                checked={mode === 'play'}
                onChange={() => setMode('play')}
              />
              Play (numbered moves)
            </label>
          </div>
        </section>

        {mode === 'setup' && (
          <section>
            <h2>Setup color</h2>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  checked={setupColor === 'B'}
                  onChange={() => setSetupColor('B')}
                />
                Black
              </label>
              <label>
                <input
                  type="radio"
                  checked={setupColor === 'W'}
                  onChange={() => setSetupColor('W')}
                />
                White
              </label>
            </div>
          </section>
        )}

        {mode === 'play' && (
          <section>
            <h2>First to move</h2>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  checked={firstPlayer === 'B'}
                  onChange={() => setFirstPlayer('B')}
                  disabled={numberedMoves.length > 0}
                />
                Black
              </label>
              <label>
                <input
                  type="radio"
                  checked={firstPlayer === 'W'}
                  onChange={() => setFirstPlayer('W')}
                  disabled={numberedMoves.length > 0}
                />
                White
              </label>
            </div>
            <p className="hint">
              Next: <strong>{nextColor === 'B' ? 'Black' : 'White'} {nextNumber}</strong>
            </p>
          </section>
        )}

        <section>
          <h2>Actions</h2>
          <div className="button-row">
            <button onClick={handleUndo} disabled={stones.length === 0}>
              Undo
            </button>
            <button onClick={handleClearMoves} disabled={numberedMoves.length === 0}>
              Clear moves
            </button>
            <button onClick={handleReset} disabled={stones.length === 0}>
              Reset all
            </button>
          </div>
        </section>

        <section>
          <h2>Tools</h2>
          <div className="button-row">
            <button onClick={() => setView('bboxTest')}>
              Test bounding boxes…
            </button>
            <button onClick={() => setView('stoneTest')}>
              Test stone detection…
            </button>
          </div>
        </section>
      </aside>

      <main className="board-pane">
        {view === 'board' && (
          <Board stones={stones} onPlay={handlePlay} />
        )}
        {view === 'bboxTest' && (
          <BboxTest onExit={() => setView('board')} />
        )}
        {view === 'stoneTest' && (
          <StoneTest onExit={() => setView('board')} />
        )}
      </main>
    </div>
  );
}

export default App;
