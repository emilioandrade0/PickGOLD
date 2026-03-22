import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Header from "./components/Header.jsx";
import NBAPage from "./pages/NBAPage.jsx";
import MLBPage from "./pages/MLBPage.jsx";
import KBOPage from "./pages/KBOPage.jsx";
import NHLPage from "./pages/NHLPage.jsx";
import NCAABaseballPage from "./pages/NCAABaseballPage.jsx";
import EuroLeaguePage from "./pages/EuroLeaguePage.jsx";
import LigaMXPage from "./pages/LigaMXPage.jsx";
import LaLigaPage from "./pages/LaLigaPage.jsx";
import InsightsPage from "./pages/InsightsPage.jsx";
import WeekdayScoringPage from "./pages/WeekdayScoringPage.jsx";
import BestPicksPage from "./pages/BestPicksPage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import { getActiveSession, logoutUser } from "./services/auth.js";

function ProtectedApp({ onLogout, userName }) {
  return (
    <div className="min-h-screen bg-transparent text-white">
      <Header onLogout={onLogout} userName={userName} />

      <Routes>
        <Route path="/" element={<Navigate to="/nba" replace />} />
        <Route path="/nba" element={<NBAPage />} />
        <Route path="/mlb" element={<MLBPage />} />
        <Route path="/kbo" element={<KBOPage />} />
        <Route path="/nhl" element={<NHLPage />} />
        <Route path="/ncaa-baseball" element={<NCAABaseballPage />} />
        <Route path="/euroleague" element={<EuroLeaguePage />} />
        <Route path="/liga-mx" element={<LigaMXPage />} />
        <Route path="/laliga" element={<LaLigaPage />} />
        <Route path="/insights" element={<InsightsPage />} />
        <Route path="/weekday-scoring" element={<WeekdayScoringPage />} />
        <Route path="/best-picks" element={<BestPicksPage />} />
        <Route path="*" element={<Navigate to="/nba" replace />} />
      </Routes>
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState(() => getActiveSession());

  function handleAuthenticated(user) {
    setSession(user || getActiveSession());
  }

  function handleLogout() {
    logoutUser();
    setSession(null);
  }

  return (
    <Routes>
      <Route
        path="/auth"
        element={session ? <Navigate to="/nba" replace /> : <AuthPage onAuthenticated={handleAuthenticated} />}
      />

      <Route
        path="*"
        element={
          session ? (
            <ProtectedApp onLogout={handleLogout} userName={session?.name || "Usuario"} />
          ) : (
            <Navigate to="/auth" replace />
          )
        }
      />
    </Routes>
  );
}