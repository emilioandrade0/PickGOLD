import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Header from "./components/Header.jsx";
import NBAPage from "./pages/NBAPage.jsx";
import MLBPage from "./pages/MLBPage.jsx";
import KBOPage from "./pages/KBOPage.jsx";
import NHLPage from "./pages/NHLPage.jsx";
import EuroLeaguePage from "./pages/EuroLeaguePage.jsx";
import LigaMXPage from "./pages/LigaMXPage.jsx";
import LaLigaPage from "./pages/LaLigaPage.jsx";
import InsightsPage from "./pages/InsightsPage.jsx";
import WeekdayScoringPage from "./pages/WeekdayScoringPage.jsx";
import BestPicksPage from "./pages/BestPicksPage.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import AdminUserApproval from "./pages/AdminUserApproval.jsx";
import PendingApprovalPage from "./pages/PendingApprovalPage.jsx";
import { getActiveSession, logoutUser } from "./services/auth.js";

function ProtectedApp({ onLogout, userName, session }) {
  return (
    <div className="min-h-screen bg-transparent text-white">
      <Header onLogout={onLogout} userName={userName} />

      <Routes>
        <Route path="/" element={<Navigate to="/nba" replace />} />
        <Route path="/nba" element={<NBAPage />} />
        <Route path="/mlb" element={<MLBPage />} />
        <Route path="/kbo" element={<KBOPage />} />
        <Route path="/nhl" element={<NHLPage />} />
        <Route path="/euroleague" element={<EuroLeaguePage />} />
        <Route path="/liga-mx" element={<LigaMXPage />} />
        <Route path="/laliga" element={<LaLigaPage />} />
        <Route path="/insights" element={<InsightsPage />} />
        <Route path="/weekday-scoring" element={<WeekdayScoringPage />} />
        <Route path="/best-picks" element={<BestPicksPage />} />
        {session?.role === "admin" && (
          <Route path="/admin/approve-users" element={<AdminUserApproval adminEmail={session.email} />} />
        )}
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

  const isAdmin = session?.role === "admin";
  const isApproved = session?.status === "approved";
  const canAccessPredictions = !!session && (isApproved || isAdmin);

  return (
    <Routes>
      <Route
        path="/auth"
        element={session ? <Navigate to={canAccessPredictions ? "/nba" : "/pending"} replace /> : <AuthPage onAuthenticated={handleAuthenticated} />}
      />

      <Route
        path="/pending"
        element={
          session ? (
            canAccessPredictions ? (
              <Navigate to="/nba" replace />
            ) : (
              <PendingApprovalPage user={session} onLogout={handleLogout} />
            )
          ) : (
            <Navigate to="/auth" replace />
          )
        }
      />

      <Route
        path="*"
        element={
          session ? (
            canAccessPredictions ? (
              <ProtectedApp onLogout={handleLogout} userName={session?.name || "Usuario"} session={session} />
            ) : (
              <Navigate to="/pending" replace />
            )
          ) : (
            <Navigate to="/auth" replace />
          )
        }
      />
    </Routes>
  );
}
