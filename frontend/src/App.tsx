import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { EditorPage } from "./pages/EditorPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { ComparePage } from "./pages/ComparePage";

function Nav() {
  const cls = ({ isActive }: { isActive: boolean }) =>
    isActive ? "nav-link active" : "nav-link";
  return (
    <nav className="navbar">
      <span className="nav-brand">
        <span className="dot" />
        Flex
      </span>
      <NavLink to="/" className={cls} end>
        Editor
      </NavLink>
      <NavLink to="/workspace" className={cls}>
        Workspace
      </NavLink>
      <NavLink to="/compare" className={cls}>
        Compare
      </NavLink>
    </nav>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<EditorPage />} />
        <Route path="/workspace" element={<WorkspacePage />} />
        <Route path="/compare" element={<ComparePage />} />
      </Routes>
    </BrowserRouter>
  );
}
