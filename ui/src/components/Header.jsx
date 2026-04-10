import { useLocation, useNavigate } from "react-router-dom";
import SportTabs from "./SportTabs.jsx";
import { getActiveSession } from "../services/auth.js";
import { useAppSettings } from "../context/AppSettingsContext.jsx";

function getSportTitle(pathname, socialMode) {
  if (pathname.startsWith("/best-picks")) return socialMode ? "Radar" : "Best Picks";
  if (pathname.startsWith("/weekday-scoring")) return socialMode ? "Ritmo" : "Weekday Scoring";
  if (pathname.startsWith("/insights")) return socialMode ? "Modelo" : "Insights";
  if (pathname.startsWith("/triple-a")) return "Triple-A";
  if (pathname.startsWith("/tennis")) return "Tennis";
  if (pathname.startsWith("/kbo")) return "KBO";
  if (pathname.startsWith("/mlb")) return "MLB";
  if (pathname.startsWith("/nhl")) return "NHL";
  if (pathname.startsWith("/ncaa-baseball")) return "NCAA Baseball";
  if (pathname.startsWith("/euroleague")) return "EuroLeague";
  if (pathname.startsWith("/liga-mx")) return "Liga MX";
  if (pathname.startsWith("/laliga")) return "LaLiga";
  if (pathname.startsWith("/bundesliga")) return "Bundesliga";
  if (pathname.startsWith("/live")) return "En Vivo";
  if (pathname.startsWith("/resumen-dia")) return "Resumen del dia";
  if (pathname.startsWith("/estadisticas")) return "Estadisticas";
  return "NBA";
}

export default function Header({ onLogout, userName }) {
  const { socialMode } = useAppSettings();
  const location = useLocation();
  const navigate = useNavigate();
  const sportTitle = getSportTitle(location.pathname, socialMode);
  const userRole = getActiveSession()?.role || null;

  return (
    <header className="relative border-b border-white/8 bg-[radial-gradient(circle_at_top_left,rgba(255,194,73,0.12),transparent_24%),radial-gradient(circle_at_top_right,rgba(52,211,153,0.08),transparent_18%),linear-gradient(180deg,#0b0e15,#0a0d14)]">
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="absolute left-8 top-8 h-36 w-36 rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-10 top-0 h-32 w-32 rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-[1780px] px-4 py-5 xl:px-6 2xl:px-8">
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="max-w-3xl">
              <div className="">
                
              </div>

              <div className="mt-4 flex flex-wrap items-end gap-x-3 gap-y-2">
                <h1 className="text-4xl font-light tracking-tight text-white sm:text-5xl">
                  {sportTitle}
                  <span className="font-semibold text-amber-300"> GOLD</span>
                </h1>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 xl:max-w-md xl:justify-end">
              {userName ? (
                <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-white/78">
                  {userName}
                </span>
              ) : null}

              {userRole === "admin" && (
                <button
                  type="button"
                  onClick={() => navigate("/admin/approve-users")}
                  className="rounded-xl border border-green-400/35 bg-green-400/10 px-3.5 py-2 text-sm font-semibold text-green-200 transition hover:bg-green-400/16"
                >
                  CONFIGURACION ADMIN
                </button>
              )}

              {onLogout ? (
                <button
                  type="button"
                  onClick={onLogout}
                  className="rounded-xl border border-amber-300/28 bg-amber-300/10 px-3.5 py-2 text-sm font-semibold text-amber-200 transition hover:bg-amber-300/16"
                >
                  Cerrar sesion
                </button>
              ) : null}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-3 shadow-[0_14px_34px_rgba(0,0,0,0.18)]">
            <div className="mb-3 flex items-center justify-between gap-4 px-1">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-white/40">
                  {socialMode ? "Ligas y modulos" : "Deportes y modulos"}
                </p>
              </div>
              <span className="text-xs text-white/42">{socialMode ? "Deportes" : "Deportes"}</span>
            </div>
            <SportTabs />
          </div>
        </div>
      </div>
    </header>
  );
}
