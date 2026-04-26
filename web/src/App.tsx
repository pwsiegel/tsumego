import { Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { BboxTest } from './BboxTest';
import { Collection } from './Collection';
import { Compare } from './Compare';
import { GridTest } from './GridTest';
import { HealthGate } from './HealthGate';
import { Home } from './Home';
import { ProblemDetail } from './ProblemDetail';
import { Review } from './Review';
import { StoneTest } from './StoneTest';
import { TestingIndex } from './TestingIndex';
import { Upload } from './Upload';
import { Validate } from './Validate';

function BboxTestRoute() {
  const navigate = useNavigate();
  return <BboxTest onExit={() => navigate('/testing')} />;
}

function StoneTestRoute() {
  const navigate = useNavigate();
  return <StoneTest onExit={() => navigate('/testing')} />;
}

function GridTestRoute() {
  const navigate = useNavigate();
  return <GridTest onExit={() => navigate('/testing')} />;
}

function TsumegoPlaceholder() {
  return (
    <div style={{ maxWidth: '36rem', margin: '4rem auto', padding: '0 1.5rem' }}>
      <h1>Tsumego library</h1>
      <p style={{ color: '#666' }}>Coming soon.</p>
    </div>
  );
}

function App() {
  return (
    <HealthGate>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/collections/:source" element={<Collection />} />
        <Route path="/collections/:source/review" element={<Review />} />
        <Route path="/collections/:source/problem/:id" element={<ProblemDetail />} />
        <Route path="/tsumego" element={<TsumegoPlaceholder />} />
        <Route path="/compare/:dataset" element={<Compare />} />
        <Route path="/testing" element={<TestingIndex />} />
        <Route path="/testing/bbox" element={<BboxTestRoute />} />
        <Route path="/testing/grid" element={<GridTestRoute />} />
        <Route path="/testing/stones" element={<StoneTestRoute />} />
        <Route path="/testing/validate/:dataset" element={<Validate />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HealthGate>
  );
}

export default App;
