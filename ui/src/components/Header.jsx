import { useLocation, useNavigate } from "react-router-dom";
import SportTabs from "./SportTabs.jsx";

function getSportTitle(pathname) {
  if (pathname.startsWith("/best-picks")) return "BEST PICKS";
  if (pathname.startsWith("/weekday-scoring")) return "WEEKDAY";
  if (pathname.startsWith("/insights")) return "INSIGHTS";
  if (pathname.startsWith("/kbo")) return "KBO";
  if (pathname.startsWith("/mlb")) return "MLB";
  if (pathname.startsWith("/nhl")) return "NHL";
  if (pathname.startsWith("/ncaa-baseball")) return "NCAA BASEBALL";
  if (pathname.startsWith("/euroleague")) return "EUROLEAGUE";
  if (pathname.startsWith("/liga-mx")) return "LIGA MX";
  if (pathname.startsWith("/laliga")) return "LALIGA";
  return "NBA";
}

export default function Header({ onLogout, userName }) {
  const location = useLocation();
  const navigate = useNavigate();
  const sportTitle = getSportTitle(location.pathname);

  let userRole = null;
  if (typeof window !== "undefined") {
    try {
      const session = JSON.parse(window.localStorage.getItem("nba_gold_session"));
      userRole = session?.role;
    } catch {
      userRole = null;
    }
  }

  return (
    <header className="border-b border-white/10 bg-black/35 shadow-2xl backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-6 py-5">
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h1 className="text-5xl font-light tracking-tight">
                {sportTitle}
                <span className="font-semibold text-amber-300"> GOLD</span>
              </h1>
              <p className="mt-2 text-sm text-white/75">
                Predicciones diarias, historial por fecha y detalle avanzado por evento.
              </p>
            </div>

            <div className="flex items-center gap-3">
              {userName ? (
                <span className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium tracking-wide text-white/90">
                  {userName}
                </span>
              ) : null}

              {userRole === "admin" && (
                <button
                  type="button"
                  onClick={() => navigate("/admin/approve-users")}
                  className="rounded-lg border border-green-400/35 bg-green-400/10 px-3 py-2 text-sm font-semibold text-green-200 transition hover:bg-green-400/20"
                >
                  Admin: Autorizar usuarios
                </button>
              )}

              {onLogout ? (
                <button
                  type="button"
                  onClick={onLogout}
                  className="rounded-lg border border-amber-300/35 bg-amber-300/10 px-3 py-2 text-sm font-semibold text-amber-200 transition hover:bg-amber-300/20"
                >
                  Cerrar sesión
                </button>
              ) : null}
            </div>
          </div>

          <SportTabs />
        </div>
      </div>
    </header>
  );
}
