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
import TermsPage from "./pages/TermsPage.jsx";
import PrivacyPage from "./pages/PrivacyPage.jsx";
import DisclaimerPage from "./pages/DisclaimerPage.jsx";
import { getActiveSession, logoutUser, refreshSession } from "./services/auth.js";
import { Link } from "react-router-dom";

function AppFooter() {
  return (
    <footer className="border-t border-white/8 bg-black/18">
      <div className="mx-auto flex max-w-[1780px] flex-wrap items-center justify-center gap-5 px-4 py-4 text-center text-sm text-white/50 xl:px-6 2xl:px-8">
        <Link to="/terms" className="transition hover:text-white/80">
          Términos y condiciones
        </Link>
        <Link to="/privacy" className="transition hover:text-white/80">
          Política de privacidad
        </Link>
        <Link to="/disclaimer" className="transition hover:text-white/80">
          Disclaimer
        </Link>
      </div>
    </footer>
  );
}

function ProtectedLayout({ onLogout, userName }) {
  return (
    <div className="min-h-screen bg-transparent text-white">
      <Header onLogout={onLogout} userName={userName} />
      <div className="flex min-h-[calc(100vh-80px)] flex-col">
        <div className="flex-1">
          <Outlet />
        </div>
        <AppFooter />
      </div>
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
      <Route
        path="/"
        element={(
          <div className="min-h-screen bg-transparent text-white">
            <div className="flex min-h-screen flex-col">
              <div className="flex-1">
                <LandingPage />
              </div>
              <AppFooter />
            </div>
          </div>
        )}
      />
      <Route
        path="/auth"
        element={
          session ? (
            <Navigate to="/nba" replace />
          ) : (
            <div className="min-h-screen bg-transparent text-white">
              <div className="flex min-h-screen flex-col">
                <div className="flex-1">
                  <AuthPage onAuthenticated={handleAuthenticated} />
                </div>
                <AppFooter />
              </div>
            </div>
          )
        }
      />
      <Route path="/terms" element={<TermsPage />} />
      <Route path="/privacy" element={<PrivacyPage />} />
      <Route path="/disclaimer" element={<DisclaimerPage />} />

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
