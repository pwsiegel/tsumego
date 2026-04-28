import { Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { ThemeToggle } from './ThemeToggle';
import { BboxTest } from './BboxTest';
import { BoardParsing } from './BoardParsing';
import { Collection } from './Collection';
import { Compare } from './Compare';
import { HealthGate } from './HealthGate';
import { Home } from './Home';
import { ProblemDetail } from './ProblemDetail';
import { Profile } from './Profile';
import { Review } from './Review';
import { Reviewed } from './Reviewed';
import { SolveEntry, SolveView } from './SolveView';
import { Submission } from './Submission';
import { TeacherView } from './TeacherView';
import { TestingIndex } from './TestingIndex';
import { Upload } from './Upload';
import { Validate } from './Validate';

function BboxTestRoute() {
  const navigate = useNavigate();
  return <BboxTest onExit={() => navigate('/testing')} />;
}

function BoardParsingRoute() {
  const navigate = useNavigate();
  return <BoardParsing onExit={() => navigate('/testing')} />;
}

function TsumegoPlaceholder() {
  return (
    <div style={{ maxWidth: '36rem', margin: '4rem auto', padding: '0 1.5rem' }}>
      <h1>Tsumego library</h1>
      <p style={{ color: 'var(--text-muted)' }}>Coming soon.</p>
    </div>
  );
}

function App() {
  return (
    <HealthGate>
      <ThemeToggle />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/reviewed" element={<Reviewed />} />
        <Route path="/submissions/:sent_at" element={<Submission />} />
        <Route path="/collections/:source/edit" element={<Collection />} />
        <Route path="/collections/:source/review" element={<Review />} />
        <Route path="/collections/:source/problem/:id" element={<ProblemDetail />} />
        <Route path="/collections/:source/solve" element={<SolveEntry />} />
        <Route path="/collections/:source/solve/:id" element={<SolveView />} />
        <Route path="/tsumego" element={<TsumegoPlaceholder />} />
        <Route path="/teacher" element={<TeacherView />} />
        <Route path="/teacher/students/:student_uid" element={<TeacherView />} />
        <Route path="/compare/:dataset" element={<Compare />} />
        <Route path="/testing" element={<TestingIndex />} />
        <Route path="/testing/bbox" element={<BboxTestRoute />} />
        <Route path="/testing/parsing" element={<BoardParsingRoute />} />
        <Route path="/testing/validate/:dataset" element={<Validate />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HealthGate>
  );
}

export default App;
