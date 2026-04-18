import { useLocation, useNavigate } from "react-router-dom";
import SportTabs from "./SportTabs.jsx";
import { getActiveSession } from "../services/auth.js";
import { useAppSettings } from "../context/AppSettingsContext.jsx";

function getSportTitle(pathname, socialMode) {
  if (pathname.startsWith("/admin/")) return "Admin";
  if (pathname.startsWith("/best-picks")) return socialMode ? "Radar" : "Best Picks";
  if (pathname.startsWith("/weekday-scoring")) return socialMode ? "Ritmo" : "Weekday Scoring";
  if (pathname.startsWith("/insights")) return socialMode ? "Modelo" : "Insights";
  if (pathname.startsWith("/triple-a")) return "Triple-A";
  if (pathname.startsWith("/tennis")) return "Tennis";
  if (pathname.startsWith("/kbo")) return "KBO";
  if (pathname.startsWith("/mlb")) return "MLB";
  if (pathname.startsWith("/lmb")) return "LMB";
  if (pathname.startsWith("/wnba")) return "WNBA";
  if (pathname.startsWith("/nhl")) return "NHL";
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
  const { socialMode, uiTheme } = useAppSettings();
  const location = useLocation();
  const navigate = useNavigate();
  const sportTitle = getSportTitle(location.pathname, socialMode);
  const userRole = getActiveSession()?.role || null;
  const isDashboardPro = uiTheme === "dashboard_pro";
  const isClassicLight = uiTheme === "classic_light";
  const sectionLabel = sportTitle;

  if (isClassicLight) {
    return (
      <header className="classic-light-surface classic-light-header border-b border-[#aeb0b7] bg-[#b6b7bd] text-[#23262f]">
        <div className="mx-auto max-w-[1780px] px-4 py-3 xl:px-6 2xl:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => navigate("/nba")}
              className="inline-flex items-center gap-2 rounded-xl border border-[#8e919b] bg-[#e8e8ec] px-3 py-1.5 text-[#242730] transition hover:bg-[#f2f2f5]"
              title="Volver al menu principal"
            >
              <span className="text-base">⬢</span>
              <span className="text-3xl font-black uppercase tracking-[0.01em] [font-family:var(--classic-display-font)]">Pick Gold</span>
            </button>

            <div className="flex flex-wrap items-center gap-2">
              {userName ? (
                <span className="rounded-full border border-[#989ba5] bg-[#ececf0] px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-[#4d5260]">
                  {userName}
                </span>
              ) : null}

              <button
                type="button"
                onClick={() => navigate("/live")}
                className="rounded-full border border-[#8f929c] bg-[#ececf0] px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-[#343844] transition hover:bg-[#f5f5f7]"
              >
                Live
              </button>
              <button
                type="button"
                onClick={() => navigate("/resumen-dia")}
                className="rounded-full border border-[#8f929c] bg-[#ececf0] px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-[#343844] transition hover:bg-[#f5f5f7]"
              >
                Eventos del dia
              </button>
              <button
                type="button"
                onClick={() => navigate(userRole === "admin" ? "/admin/approve-users" : "/estadisticas")}
                className="rounded-full border border-[#8f929c] bg-[#ececf0] px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-[#343844] transition hover:bg-[#f5f5f7]"
              >
                Configuracion
              </button>

              {onLogout ? (
                <button
                  type="button"
                  onClick={onLogout}
                  className="rounded-full border border-[#9b7f60] bg-[#efe0c8] px-3 py-1 text-xs font-semibold uppercase tracking-[0.08em] text-[#5b4630] transition hover:bg-[#f6e8d3]"
                >
                  Salir
                </button>
              ) : null}
            </div>
          </div>

          <div className="mt-3 rounded-[14px] border border-[#aeb0b7] bg-[#d0d1d6] px-3 py-2">
            <SportTabs userRole={userRole} />
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className={`relative border-b border-white/8 ${isDashboardPro ? "bg-[radial-gradient(circle_at_top_left,rgba(255,194,73,0.10),transparent_30%),radial-gradient(circle_at_top_right,rgba(52,211,153,0.08),transparent_24%),linear-gradient(180deg,#090d16,#090d14)]" : "bg-[radial-gradient(circle_at_top_left,rgba(255,194,73,0.12),transparent_24%),radial-gradient(circle_at_top_right,rgba(52,211,153,0.08),transparent_18%),linear-gradient(180deg,#0b0e15,#0a0d14)]"}`}>
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="absolute left-8 top-8 h-36 w-36 rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-10 top-0 h-32 w-32 rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <div className={`relative mx-auto max-w-[1780px] px-4 ${isDashboardPro ? "py-3" : "py-5"} xl:px-6 2xl:px-8`}>
        <div className={`flex flex-col ${isDashboardPro ? "gap-4" : "gap-5"}`}>
          <div className={`flex flex-col gap-5 xl:flex-row ${isDashboardPro ? "xl:items-center" : "xl:items-start"} xl:justify-between`}>
            <div className="max-w-3xl">
              {isDashboardPro ? (
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => navigate("/nba")}
                    className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-amber-300/35 bg-amber-300/12 text-amber-200 transition hover:border-amber-200/60 hover:bg-amber-300/18"
                    title="Volver al menu principal"
                  >
                    ⬢
                  </button>
                  <button
                    type="button"
                    onClick={() => navigate("/nba")}
                    className="text-3xl font-light tracking-tight text-white transition hover:text-amber-100"
                    title="Volver al menu principal"
                  >
                    Pick<span className="font-semibold text-amber-300">Gold</span>
                  </button>
                  <span className="text-white/35">/</span>
                  <span className="rounded-full border border-white/12 bg-white/[0.04] px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-white/70">
                    {sectionLabel}
                  </span>
                  <button
                    type="button"
                    onClick={() => navigate("/live")}
                    className="rounded-lg border border-white/12 bg-transparent px-3 py-1.5 text-xs font-semibold text-white/78 transition hover:border-white/22 hover:bg-white/[0.06] hover:text-white"
                  >
                    En vivo
                  </button>
                </div>
              ) : (
                <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
                  <button
                    type="button"
                    onClick={() => navigate("/nba")}
                    className="text-4xl sm:text-5xl font-light tracking-tight text-white transition hover:text-amber-100"
                    title="Volver al menu principal"
                  >
                    {sportTitle}
                    <span className="font-semibold text-amber-300"> GOLD</span>
                  </button>
                </div>
              )}
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
                  className={`rounded-xl border px-3.5 py-2 text-sm font-semibold transition ${isDashboardPro ? "border-cyan-300/35 bg-cyan-400/10 text-cyan-100 hover:bg-cyan-400/16" : "border-green-400/35 bg-green-400/10 text-green-200 hover:bg-green-400/16"}`}
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

          {!isDashboardPro && (
          <div className={`border border-white/8 p-3 shadow-[0_14px_34px_rgba(0,0,0,0.18)] ${isDashboardPro ? "rounded-[24px] bg-[linear-gradient(180deg,rgba(255,255,255,0.025),rgba(255,255,255,0.01))]" : "rounded-[28px] bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))]"}`}>
            <div className="mb-3 flex items-center justify-between gap-4 px-1">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-white/40">
                  {socialMode ? "Ligas y modulos" : "Deportes y modulos"}
                </p>
              </div>
              <span className="text-xs text-white/42">{userRole === "admin" ? "Vista admin" : "Vista usuario"}</span>
            </div>
            <SportTabs userRole={userRole} />
          </div>
          )}
        </div>
      </div>
    </header>
  );
}
