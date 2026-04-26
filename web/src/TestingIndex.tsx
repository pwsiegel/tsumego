import { Link } from 'react-router-dom';
import './TestingIndex.css';

export function TestingIndex() {
  return (
    <div className="testing-index">
      <h1>Pipeline testing</h1>
      <p className="intro">
        Internal tools for inspecting each stage of the PDF → SGF pipeline.
      </p>
      <ul className="testing-links">
        <li>
          <Link to="/testing/bbox">Board bbox detection</Link>
          <span className="desc">
            Upload a PDF and see the raw YOLO bounding boxes drawn over
            each page. Useful for diagnosing the board detector.
          </span>
        </li>
        <li>
          <Link to="/testing/stones">Stone detection + discretization</Link>
          <span className="desc">
            Upload a PDF, step through every detected board, and see the
            inferred 19×19 grid alongside the crop.
          </span>
        </li>
        <li>
          <Link to="/testing/validate/hm2">Validation: hm2</Link>
          <span className="desc">
            Run the current pipeline against the hm2 ground-truth dataset
            and inspect problems where the detector disagrees.
          </span>
        </li>
      </ul>
      <p className="home-link">
        <Link to="/">← back to app</Link>
      </p>
    </div>
  );
}
