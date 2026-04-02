import { useEffect, useState } from "react";
import { Navigate, Outlet, Route, Routes } from "react-router-dom";
import Header from "./components/Header.jsx";
import LandingPage from "./pages/LandingPage.jsx";
import NBAPage from "./pages/NBAPage.jsx";
import MLBPage from "./pages/MLBPage.jsx";
import TennisPage from "./pages/TennisPage.jsx";
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
import AdminUserApproval from "./pages/AdminUserApproval.jsx";
import { getActiveSession, logoutUser, refreshSession } from "./services/auth.js";

function ProtectedLayout({ onLogout, userName }) {
  return (
    <div className="min-h-screen bg-transparent text-white">
      <Header onLogout={onLogout} userName={userName} />
      <Outlet />
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState(() => getActiveSession());

  useEffect(() => {
    let active = true;

    async function syncSession() {
      const current = getActiveSession();
      if (!current?.token) return;
      const refreshed = await refreshSession();
      if (!active) return;
      if (refreshed?.ok) {
        setSession(refreshed);
      } else {
        setSession(null);
      }
    }

    syncSession();
    return () => {
      active = false;
    };
  }, []);

  function handleAuthenticated(user) {
    setSession(user || getActiveSession());
  }

  function handleLogout() {
    logoutUser();
    setSession(null);
  }

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route
        path="/auth"
        element={session ? <Navigate to="/nba" replace /> : <AuthPage onAuthenticated={handleAuthenticated} />}
      />

      <Route
        element={
          session ? (
            <ProtectedLayout onLogout={handleLogout} userName={session?.name || "Usuario"} session={session} />
          ) : (
            <Navigate to="/auth" replace />
          )
        }
      >
        <Route path="/nba" element={<NBAPage />} />
        <Route path="/mlb" element={<MLBPage />} />
        <Route path="/tennis" element={<TennisPage />} />
        <Route path="/kbo" element={<KBOPage />} />
        <Route path="/nhl" element={<NHLPage />} />
        <Route path="/ncaa-baseball" element={<NCAABaseballPage />} />
        <Route path="/euroleague" element={<EuroLeaguePage />} />
        <Route path="/liga-mx" element={<LigaMXPage />} />
        <Route path="/laliga" element={<LaLigaPage />} />
        <Route path="/insights" element={<InsightsPage />} />
        <Route path="/weekday-scoring" element={<WeekdayScoringPage />} />
        <Route path="/best-picks" element={<BestPicksPage />} />
        {session?.role === "admin" && (
          <Route path="/admin/approve-users" element={<AdminUserApproval />} />
        )}
      </Route>

      <Route path="*" element={<Navigate to={session ? "/nba" : "/"} replace />} />
    </Routes>
  );
}
