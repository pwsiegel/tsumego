import { Link } from 'react-router-dom';
import './Home.css';

export function Home() {
  return (
    <div className="home">
      <h1>Welcome</h1>

      <section className="home-guide">
        <h2>Solving tsumego</h2>
        <p>
          Pick a collection under <Link to="/library">Solve tsumego</Link>, play your answer on the
          board, then hit <strong>Add to submission</strong>. Answers pile up in the drawer on the
          right and don't go anywhere until you open it and press <strong>Send</strong>. Your
          teacher marks them by hand, so expect a wait. <Link to="/submissions">Submissions</Link>
          {' '}shows what's outstanding, <Link to="/history">History</Link> shows everything.
        </p>
        <p>
          <strong>Explore</strong>, the tab beside <strong>Solve</strong>, lets you push moves
          around with AI help. Nothing there is recorded. If a problem beats you, hit{' '}
          <strong>Mark stuck</strong> rather than guessing — your teacher sees it right away.
        </p>
      </section>

      <section className="home-guide">
        <h2>Playing and reviewing</h2>
        <p>
          <Link to="/play">Play AI</Link> is a game against a human-like opponent at a rank you
          choose. <Link to="/review">Review games</Link> holds your own games and analyses them;
          click any point on the board to branch a variation.
        </p>
        <p>
          The engine runs in your browser rather than on a server, so it takes a moment to load and
          it's weaker than a desktop one: 50 playouts a position by default. It's steady on shape
          but can misread a deep variation. Raise the playouts from the sidebar button if you want
          a second opinion.
        </p>
      </section>
    </div>
  );
}
