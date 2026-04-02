import { useLocation, useNavigate } from "react-router-dom";
import SportTabs from "./SportTabs.jsx";
import { getActiveSession } from "../services/auth.js";

function getSportTitle(pathname) {
  if (pathname.startsWith("/best-picks")) return "BEST PICKS";
  if (pathname.startsWith("/weekday-scoring")) return "WEEKDAY";
  if (pathname.startsWith("/insights")) return "INSIGHTS";
  if (pathname.startsWith("/tennis")) return "TENNIS";
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
  const isInsightsPage =
    location.pathname.startsWith("/insights")
    || location.pathname.startsWith("/best-picks")
    || location.pathname.startsWith("/weekday-scoring");

  const userRole = getActiveSession()?.role || null;

  return (
    <header className="relative overflow-hidden border-b border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(255,196,61,0.16),transparent_28%),radial-gradient(circle_at_top_right,rgba(38,224,196,0.12),transparent_24%),linear-gradient(180deg,rgba(7,9,14,0.98),rgba(10,12,18,0.94))] shadow-2xl backdrop-blur-sm">
      <div className="pointer-events-none absolute inset-0 opacity-60">
        <div className="absolute left-12 top-12 h-56 w-56 rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-10 top-8 h-48 w-48 rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="grid flex-1 gap-6 xl:grid-cols-[minmax(0,1.3fr)_360px]">
              <div>
                <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/90">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
                  Plataforma premium de picks
                </div>

                <h1 className="max-w-4xl text-5xl font-light leading-none tracking-tight sm:text-6xl lg:text-7xl">
                  {sportTitle}
                  <span className="font-semibold text-amber-300"> GOLD</span>
                </h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-white/72 sm:text-lg">
                  Picks listos para vender mejor: confianza visual, lectura inmediata, historial claro
                  y una experiencia que convierte datos fríos en valor premium.
                </p>

                <div className="mt-6 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => navigate("/best-picks")}
                    className="rounded-2xl bg-amber-300 px-5 py-3 text-sm font-bold text-[#141821] shadow-[0_0_22px_rgba(246,196,83,0.28)] transition hover:-translate-y-0.5 hover:bg-amber-200"
                  >
                    Ver Best Picks
                  </button>
                  <button
                    type="button"
                    onClick={() => navigate("/insights")}
                    className="rounded-2xl border border-white/14 bg-white/6 px-5 py-3 text-sm font-semibold text-white/88 transition hover:border-cyan-300/35 hover:bg-cyan-300/10 hover:text-cyan-100"
                  >
                    Explorar Insights
                  </button>
                </div>

                <div className="mt-6 flex flex-wrap gap-3 text-sm text-white/68">
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2">UX pensada para conversión</span>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2">Jerarquía premium</span>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2">Multi-mercado y multi-deporte</span>
                </div>
              </div>

              <div className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(16,19,27,0.96))] p-5 shadow-[0_22px_60px_rgba(0,0,0,0.28)]">
                <div className="flex items-center justify-between">
                  <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/60">
                    Panel de conversión
                  </span>
                  <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                    + confianza visual
                  </span>
                </div>

                <div className="mt-5 grid grid-cols-3 gap-3">
                  <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">% Hit</p>
                    <p className="mt-3 text-3xl font-semibold text-white">77.8</p>
                  </div>
                  <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mercados</p>
                    <p className="mt-3 text-3xl font-semibold text-white">12</p>
                  </div>
                  <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Premium</p>
                    <p className="mt-3 text-3xl font-semibold text-amber-200">A+</p>
                  </div>
                </div>

                <div className="mt-4 rounded-2xl border border-cyan-400/18 bg-cyan-400/8 p-4">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-cyan-100/70">Propuesta de valor</p>
                  <p className="mt-2 text-sm leading-6 text-white/78">
                    Menos ruido, más claridad. Cada pantalla debe hacer que el usuario sienta
                    ventaja, rapidez y control.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 xl:justify-end">
              {userName ? (
                <span className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium tracking-wide text-white/90">
                  {userName}
                </span>
              ) : null}

              {userRole === "admin" && (
                <button
                  type="button"
                  onClick={() => navigate("/admin/approve-users")}
                  className="rounded-xl border border-green-400/35 bg-green-400/10 px-3 py-2 text-sm font-semibold text-green-200 transition hover:bg-green-400/20"
                >
                  Admin: Autorizar usuarios
                </button>
              )}

              {onLogout ? (
                <button
                  type="button"
                  onClick={onLogout}
                  className="rounded-xl border border-amber-300/35 bg-amber-300/10 px-3 py-2 text-sm font-semibold text-amber-200 transition hover:bg-amber-300/20"
                >
                  Cerrar sesión
                </button>
              ) : null}
            </div>
          </div>

          <div className={isInsightsPage ? "pt-1" : "pt-2"}>
            <SportTabs />
          </div>
        </div>
      </div>
    </header>
  );
}
