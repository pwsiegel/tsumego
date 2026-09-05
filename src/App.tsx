import { useEffect, useState } from 'react';
import { Routes, Route, Navigate, Link, NavLink, useLocation, useNavigate, type Location } from 'react-router-dom';
import { useAuth } from './auth';
import { BatchProvider } from './batch';
import { Spinner } from './Spinner';
import { ThemeToggle } from './ThemeToggle';
import { ProblemModalShell } from './ProblemModal';
import { BatchDrawer } from './BatchDrawer';
import { listStudents } from './data/links';
import { Login } from './views/Login';
import './Sidebar.css';
import { Home } from './views/Home';
import { Library } from './views/Library';
import { CollectionView } from './views/Collection';
import { Solve } from './views/Solve';
import { Submissions } from './views/Submissions';
import { SubmissionDetail } from './views/SubmissionDetail';
import { History } from './views/History';
import { ProfileModal } from './views/Profile';
import { Review } from './views/Review';
import { GameReview } from './views/GameReview';
import { ProGames } from './views/ProGames';
import { EngineHubProvider, EngineStatusButton } from './katago/engineHub';

function navClass({ isActive }: { isActive: boolean }) {
  return isActive ? 'sidebar-link active' : 'sidebar-link';
}

function subNavClass({ isActive }: { isActive: boolean }) {
  return isActive ? 'sidebar-link sidebar-sublink active' : 'sidebar-link sidebar-sublink';
}

function Sidebar({ teacherMode, canToggle, onToggle }: {
  teacherMode: boolean;
  canToggle: boolean;
  onToggle: () => void;
}) {
  const { profile, signOutUser } = useAuth();
  const [showProfile, setShowProfile] = useState(false);
  return (
    <aside className="sidebar">
      <Link to={teacherMode ? '/submissions' : '/'} className="sidebar-brand">Go training</Link>
      {/* One nav definition for both roles: teacher view just hides the
          player-only items. Submissions/History route to the role's own pages. */}
      <nav className="sidebar-links" aria-label="Primary">
        {!teacherMode && <NavLink to="/" end className={navClass}>Home</NavLink>}
        <div className="sidebar-group">
          {teacherMode
            ? <span className="sidebar-group-label">Solve tsumego</span>
            : <NavLink to="/library" className={navClass}>Solve tsumego</NavLink>}
          <NavLink to="/submissions" end className={subNavClass}>Submissions</NavLink>
          <NavLink to="/history" className={subNavClass}>History</NavLink>
        </div>
        {!teacherMode && <NavLink to="/play" className={navClass}>Play AI</NavLink>}
        <div className="sidebar-group">
          <NavLink to="/review" end className={navClass}>Review games</NavLink>
          <NavLink to="/review/online" className={subNavClass}>Online</NavLink>
          <NavLink to="/review/over-the-board" className={subNavClass}>Over the board</NavLink>
          <NavLink to="/review/ai" className={subNavClass}>vs AI</NavLink>
        </div>
        <NavLink to="/pro-games" className={navClass}>Pro games</NavLink>
      </nav>
      <div className="sidebar-foot">
        <EngineStatusButton />
        {profile?.displayName && (
          <button type="button" className="sidebar-name" onClick={() => setShowProfile(true)}>
            {profile.displayName}
          </button>
        )}
        {canToggle && (
          <button type="button" className="sidebar-btn" onClick={onToggle}>
            {teacherMode ? 'Player view' : 'Teacher view'}
          </button>
        )}
        <button type="button" className="sidebar-btn" onClick={signOutUser}>Sign out</button>
        <ThemeToggle />
      </div>
      {showProfile && <ProfileModal onClose={() => setShowProfile(false)} />}
    </aside>
  );
}

export default function App() {
  const { user, profile, loading, signOutUser } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [override, setOverride] = useState<boolean | null>(null);
  const [canTeach, setCanTeach] = useState(false);
  useEffect(() => {
    if (user) listStudents(user.uid).then((s) => setCanTeach(s.length > 0)).catch(() => {});
  }, [user]);
  const backgroundLocation = (location.state as { backgroundLocation?: Location } | null)?.backgroundLocation;

  if (loading) return <div className="center-screen"><Spinner /></div>;
  if (!user) return <Login />;
  if (!profile) {
    return (
      <div className="center-screen">
        <p>Signed in as {user.email}, but this account isn’t authorized.</p>
        <button onClick={signOutUser}>Sign out</button>
      </div>
    );
  }

  // Teacher mode follows the account role; a dual-role user (a teacher who also
  // studies) can flip views, and `override` holds that manual choice.
  const teacherMode = override ?? (profile.role === 'teacher');
  const toggleMode = () => {
    const next = !teacherMode;
    setOverride(next);
    navigate(next ? '/submissions' : '/');
  };

  // The batch drawer ("current submission") belongs to the tsumego workflow only.
  const effectivePath = (backgroundLocation ?? location).pathname;
  const showBatch = !teacherMode && /^\/(library|submissions|history|solve)(\/|$)/.test(effectivePath);

  return (
    <BatchProvider>
      <EngineHubProvider>
      <div className="app-shell">
        <Sidebar teacherMode={teacherMode} canToggle={canTeach || profile.role === 'teacher'} onToggle={toggleMode} />
        <main className="app-main">
          <Routes location={backgroundLocation ?? location}>
            <Route path="/" element={teacherMode ? <Navigate to="/submissions" replace /> : <Home />} />
            <Route path="/library" element={<Library />} />
            <Route path="/library/:slug" element={<CollectionView />} />
            <Route path="/solve/:slug/:id" element={<Solve />} />
            <Route path="/submissions" element={<Submissions teacherMode={teacherMode} />} />
            <Route path="/submissions/:id" element={<SubmissionDetail />} />
            <Route path="/history" element={<History teacherMode={teacherMode} />} />
            <Route path="/play" element={<GameReview key="play" fresh />} />
            <Route path="/review" element={<Review teacherMode={teacherMode} />} />
            <Route
              path="/review/online"
              element={<Review key="online" teacherMode={teacherMode} scope="online" />}
            />
            <Route
              path="/review/over-the-board"
              element={<Review key="otb" teacherMode={teacherMode} scope="otb" />}
            />
            <Route
              path="/review/ai"
              element={<Review key="ai" teacherMode={teacherMode} scope="ai" />}
            />
            <Route path="/review/:id" element={<GameReview key="review" />} />
            <Route path="/pro-games" element={<ProGames />} />
            <Route path="/teacher" element={<Navigate to="/submissions" replace />} />
            <Route path="/teacher/submissions" element={<Navigate to="/submissions" replace />} />
            <Route path="/teacher/history" element={<Navigate to="/history" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          {backgroundLocation && (
            <Routes>
              <Route path="/solve/:slug/:id" element={<ProblemModalShell><Solve /></ProblemModalShell>} />
            </Routes>
          )}
        </main>
        {showBatch && <BatchDrawer />}
      </div>
      </EngineHubProvider>
    </BatchProvider>
  );
}
