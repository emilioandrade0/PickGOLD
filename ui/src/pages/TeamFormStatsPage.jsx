import { useEffect, useMemo, useState } from "react";
import { fetchTeamFormInsights, fetchTeamFormTeamDetail } from "../services/api.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

const SPORTS = [
  { key: "nba", label: "NBA" },
  { key: "mlb", label: "MLB" },
  { key: "triple_a", label: "Triple-A" },
  { key: "tennis", label: "Tennis" },
  { key: "kbo", label: "KBO" },
  { key: "nhl", label: "NHL" },
  { key: "euroleague", label: "EuroLeague" },
  { key: "liga_mx", label: "Liga MX" },
  { key: "laliga", label: "LaLiga" },
];

function toneByLabel(label) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("mala")) return "text-rose-200 border-rose-400/40 bg-rose-500/12";
  if (normalized.includes("media")) return "text-orange-200 border-orange-400/40 bg-orange-500/12";
  if (normalized.includes("normal")) return "text-white/90 border-white/20 bg-white/[0.06]";
  if (normalized.includes("buena")) return "text-cyan-100 border-cyan-400/40 bg-cyan-500/12";
  if (normalized.includes("muy")) return "text-emerald-100 border-emerald-400/45 bg-emerald-500/12";
  return "text-white/90 border-white/20 bg-white/[0.06]";
}

function borderByScore(scorePct) {
  if (scorePct >= 75) return "border-emerald-400/45";
  if (scorePct >= 60) return "border-cyan-400/40";
  if (scorePct >= 45) return "border-white/18";
  if (scorePct >= 30) return "border-orange-400/40";
  return "border-rose-400/45";
}

export default function TeamFormStatsPage() {
  const [rows, setRows] = useState([]);
  const [windowGames, setWindowGames] = useState(8);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [search, setSearch] = useState("");
  const [sportFilter, setSportFilter] = useState("all");
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [teamDetail, setTeamDetail] = useState(null);
  const [teamDetailLoading, setTeamDetailLoading] = useState(false);
  const [teamDetailError, setTeamDetailError] = useState("");

  async function loadStats({ force = false, windowSize = windowGames } = {}) {
    if (force) setRefreshing(true);
    else setLoading(true);
    setError("");

    try {
      const payload = await fetchTeamFormInsights(windowSize, 3, force);
      setRows(Array.isArray(payload?.rows) ? payload.rows : []);
      setLastUpdated(String(payload?.generated_at || ""));
    } catch (err) {
      setRows([]);
      setError(err?.message || "No se pudieron cargar las rachas de equipos.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadStats({ windowSize: windowGames });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowGames]);

  const filteredRows = useMemo(() => {
    const text = String(search || "").trim().toLowerCase();
    return rows.filter((row) => {
      const displayName = getTeamDisplayName(row.sportKey, row.teamCode || row.teamName);
      if (sportFilter !== "all" && row.sportKey !== sportFilter) return false;
      if (!text) return true;
      return `${displayName} ${row.teamCode} ${row.sportLabel}`.toLowerCase().includes(text);
    });
  }, [rows, search, sportFilter]);

  const topRows = filteredRows.slice(0, 5);
  const lowRows = [...filteredRows].sort((a, b) => Number(a.formScorePct) - Number(b.formScorePct)).slice(0, 5);

  async function openTeamModal(row) {
    setSelectedTeam(row);
    setTeamDetail(null);
    setTeamDetailError("");
    setTeamDetailLoading(true);
    try {
      const detail = await fetchTeamFormTeamDetail(
        row.sportKey,
        row.teamCode || row.teamName,
        windowGames,
        30,
        5,
        4000,
        false,
      );
      setTeamDetail(detail);
    } catch (err) {
      setTeamDetailError(err?.message || "No se pudo cargar la curva de racha.");
    } finally {
      setTeamDetailLoading(false);
    }
  }

  function closeTeamModal() {
    setSelectedTeam(null);
    setTeamDetail(null);
    setTeamDetailError("");
    setTeamDetailLoading(false);
  }

  return (
    <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
      <section className="mb-6 rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.10),transparent_28%),linear-gradient(180deg,rgba(15,19,27,0.97),rgba(10,13,20,0.98))] p-6 shadow-[0_20px_56px_rgba(0,0,0,0.24)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-200/85">Estadisticas IA</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Racha actual por equipo</h2>
            <p className="mt-1 text-sm text-white/60">Calculo server-side con cache para carga rapida y estable.</p>
            {lastUpdated && (
              <p className="mt-2 text-xs text-white/45">Actualizado: {new Date(lastUpdated).toLocaleString()}</p>
            )}
          </div>

          <button
            type="button"
            onClick={() => loadStats({ force: true })}
            className="rounded-xl border border-emerald-300/35 bg-emerald-300/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-300/18"
          >
            {refreshing ? "Actualizando..." : "Actualizar"}
          </button>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Equipos analizados</p>
            <p className="mt-2 text-3xl font-semibold text-white">{rows.length}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Ventana</p>
            <div className="mt-2">
              <select
                value={windowGames}
                onChange={(e) => setWindowGames(Number(e.target.value))}
                className="w-full rounded-xl border border-white/15 bg-black/25 px-3 py-2 text-sm text-white outline-none focus:border-emerald-300/55"
              >
                <option value={5}>Ultimos 5 juegos</option>
                <option value={8}>Ultimos 8 juegos</option>
                <option value={12}>Ultimos 12 juegos</option>
              </select>
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mejor score</p>
            <p className="mt-2 text-3xl font-semibold text-emerald-300">
              {filteredRows[0] ? `${Number(filteredRows[0].formScorePct).toFixed(1)}%` : "-"}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Filtro activo</p>
            <p className="mt-2 text-lg font-semibold text-cyan-200">
              {sportFilter === "all" ? "Todos los deportes" : SPORTS.find((s) => s.key === sportFilter)?.label}
            </p>
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-[240px_minmax(0,1fr)]">
          <select
            value={sportFilter}
            onChange={(e) => setSportFilter(e.target.value)}
            className="rounded-2xl border border-white/15 bg-black/25 px-4 py-3 text-sm text-white outline-none focus:border-emerald-300/55"
          >
            <option value="all">Todos los deportes</option>
            {SPORTS.map((sport) => (
              <option key={sport.key} value={sport.key}>
                {sport.label}
              </option>
            ))}
          </select>

          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Buscar equipo..."
            className="rounded-2xl border border-white/15 bg-black/25 px-4 py-3 text-sm text-white outline-none placeholder:text-white/35 focus:border-emerald-300/55"
          />
        </div>
      </section>

      {loading && (
        <div className="rounded-[28px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
          Analizando rachas por equipo...
        </div>
      )}

      {!loading && error && (
        <div className="rounded-[28px] border border-rose-400/45 bg-rose-500/10 p-10 text-center text-rose-100">
          {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <section className="mb-6 grid gap-4 xl:grid-cols-2">
            <div className="rounded-3xl border border-white/10 bg-black/20 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-white/60">Top forma</h3>
              <div className="mt-3 space-y-2">
                {topRows.map((row) => (
                  <div key={`top-${row.sportKey}-${row.teamCode}`} className="flex items-center justify-between rounded-xl border border-emerald-400/25 bg-emerald-500/8 px-3 py-2">
                    <p className="text-sm font-semibold text-white">{getTeamDisplayName(row.sportKey, row.teamCode || row.teamName)} <span className="text-white/50">({row.sportLabel})</span></p>
                    <p className="text-sm font-semibold text-emerald-200">{Number(row.formScorePct).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-black/20 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-white/60">Top baja forma</h3>
              <div className="mt-3 space-y-2">
                {lowRows.map((row) => (
                  <div key={`low-${row.sportKey}-${row.teamCode}`} className="flex items-center justify-between rounded-xl border border-rose-400/25 bg-rose-500/8 px-3 py-2">
                    <p className="text-sm font-semibold text-white">{getTeamDisplayName(row.sportKey, row.teamCode || row.teamName)} <span className="text-white/50">({row.sportLabel})</span></p>
                    <p className="text-sm font-semibold text-rose-200">{Number(row.formScorePct).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {filteredRows.map((row) => (
              <article
                key={`${row.sportKey}-${row.teamCode}`}
                className={`cursor-pointer rounded-[24px] border bg-[linear-gradient(180deg,rgba(20,24,34,0.96),rgba(13,17,25,0.98))] p-4 shadow-[0_14px_34px_rgba(0,0,0,0.2)] transition hover:-translate-y-0.5 hover:border-cyan-300/40 ${borderByScore(Number(row.formScorePct))}`}
                onClick={() => openTeamModal(row)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-base font-semibold text-white">{getTeamDisplayName(row.sportKey, row.teamCode || row.teamName)}</p>
                    <p className="mt-0.5 text-xs text-white/55">{row.sportLabel}</p>
                  </div>
                  <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold ${toneByLabel(row.formLabel)}`}>
                    {row.formLabel}
                  </span>
                </div>

                <div className="mt-4 flex items-end justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Form Score</p>
                    <p className="mt-1 text-3xl font-semibold text-white">{Number(row.formScorePct).toFixed(1)}%</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Racha actual</p>
                    <p className="mt-1 text-lg font-semibold text-cyan-100">{row.streak}</p>
                  </div>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-rose-400 via-amber-300 to-emerald-400"
                    style={{ width: `${Math.max(0, Math.min(100, Number(row.formScorePct) || 0))}%` }}
                  />
                </div>

                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/85">
                    W-D-L: {row.wins}-{row.draws}-{row.losses}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/85">
                    DG: {Number(row.diff) >= 0 ? `+${row.diff}` : row.diff}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/70">
                    GF: {row.gf} | GA: {row.ga}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/70">
                    Muestra: {row.gamesSample}
                  </div>
                </div>

                <p className="mt-3 text-xs text-white/45">Ultimo juego: {row.lastMatchDate}</p>
              </article>
            ))}
          </section>

          {filteredRows.length === 0 && (
            <div className="rounded-[26px] border border-white/10 bg-white/[0.03] p-8 text-center text-white/65">
              No hay equipos para ese filtro.
            </div>
          )}
        </>
      )}

      {selectedTeam && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="relative max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-3xl border border-white/12 bg-[linear-gradient(180deg,rgba(18,24,36,0.98),rgba(10,14,24,0.98))] shadow-[0_30px_70px_rgba(0,0,0,0.38)]">
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-white/10 bg-[#101827]/95 px-6 py-4 backdrop-blur">
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-cyan-200/75">{selectedTeam.sportLabel}</p>
                <h3 className="mt-1 text-2xl font-semibold text-white">
                  {getTeamDisplayName(selectedTeam.sportKey, selectedTeam.teamCode || selectedTeam.teamName)}
                </h3>
              </div>
              <button
                type="button"
                onClick={closeTeamModal}
                className="rounded-xl border border-white/20 bg-white/8 px-4 py-2 text-sm text-white/80 transition hover:bg-white/14 hover:text-white"
              >
                Cerrar
              </button>
            </div>

            <div className="space-y-5 p-6">
              {teamDetailLoading && (
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-8 text-center text-white/75">
                  Construyendo curva de racha...
                </div>
              )}

              {!teamDetailLoading && teamDetailError && (
                <div className="rounded-2xl border border-rose-400/45 bg-rose-500/10 p-8 text-center text-rose-100">
                  {teamDetailError}
                </div>
              )}

              {!teamDetailLoading && !teamDetailError && teamDetail && (
                <>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Racha actual</p>
                      <p className="mt-2 text-2xl font-semibold text-cyan-100">{teamDetail.current?.form_label || "-"}</p>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Form score actual</p>
                      <p className="mt-2 text-2xl font-semibold text-emerald-300">{Number(teamDetail.current?.score_pct || 0).toFixed(1)}%</p>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Cambio vs inicio</p>
                      <p className={`mt-2 text-2xl font-semibold ${Number(teamDetail.change_vs_start_pct || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                        {Number(teamDetail.change_vs_start_pct || 0) >= 0 ? "+" : ""}{Number(teamDetail.change_vs_start_pct || 0).toFixed(1)} pts
                      </p>
                    </div>
                  </div>

                  <div className="rounded-3xl border border-cyan-400/22 bg-[linear-gradient(180deg,rgba(34,211,238,0.08),rgba(8,16,28,0.35))] p-5">
                    <p className="mb-3 text-xs uppercase tracking-[0.16em] text-cyan-100/75">Curva de racha</p>
                    <TeamTrendChart timeline={teamDetail.timeline || []} />
                    <p className="mt-3 text-sm text-white/72">
                      {buildNarrative(teamDetail.timeline || [])}
                    </p>
                  </div>

                  {teamDetail.forecast && (
                    <div className="rounded-3xl border border-emerald-400/22 bg-[linear-gradient(180deg,rgba(16,185,129,0.10),rgba(8,16,28,0.35))] p-5">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <p className="text-xs uppercase tracking-[0.16em] text-emerald-100/80">Prediccion IA por simulacion</p>
                        <span className="rounded-full border border-emerald-300/35 bg-emerald-300/12 px-3 py-1 text-xs font-semibold text-emerald-100">
                          {teamDetail.forecast.trend_label}
                        </span>
                      </div>

                      <div className="mt-4 grid gap-3 sm:grid-cols-3">
                        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mejorar</p>
                          <p className="mt-2 text-2xl font-semibold text-emerald-300">{toPct(teamDetail.forecast.improve_probability)}</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mantenerse</p>
                          <p className="mt-2 text-2xl font-semibold text-cyan-200">{toPct(teamDetail.forecast.stable_probability)}</p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Empeorar</p>
                          <p className="mt-2 text-2xl font-semibold text-rose-300">{toPct(teamDetail.forecast.decline_probability)}</p>
                        </div>
                      </div>

                      <div className="mt-4 grid gap-3 sm:grid-cols-3">
                        <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-white/80">
                          Score esperado: <span className="font-semibold text-white">{Number(teamDetail.forecast.expected_score_pct || 0).toFixed(1)}%</span>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-white/80">
                          Cambio esperado: <span className={`font-semibold ${Number(teamDetail.forecast.expected_change_pct || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                            {Number(teamDetail.forecast.expected_change_pct || 0) >= 0 ? "+" : ""}{Number(teamDetail.forecast.expected_change_pct || 0).toFixed(1)}
                          </span>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-white/80">
                          Rango esperado (P25-P75): <span className="font-semibold text-white">{Number(teamDetail.forecast.score_p25 || 0).toFixed(1)}% - {Number(teamDetail.forecast.score_p75 || 0).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <p className="mb-3 text-xs uppercase tracking-[0.16em] text-white/55">Evolucion reciente</p>
                    <div className="space-y-2">
                      {(teamDetail.timeline || []).slice(-10).reverse().map((point, idx) => (
                        <div key={`${point.date}-${idx}`} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
                          <span className="text-sm font-semibold text-white/90">{point.date}</span>
                          <span className="text-sm text-white/75">Resultado: {point.outcome}</span>
                          <span className="text-sm text-white/75">W-D-L: {point.wins}-{point.draws}-{point.losses}</span>
                          <span className="text-sm text-white/75">Racha: {point.form_label}</span>
                          <span className="text-sm font-semibold text-cyan-100">{Number(point.score_pct).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function TeamTrendChart({ timeline }) {
  const points = Array.isArray(timeline) ? timeline : [];
  if (points.length < 2) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-8 text-center text-sm text-white/60">
        No hay suficiente historial para graficar.
      </div>
    );
  }

  const width = 900;
  const height = 220;
  const padX = 28;
  const padY = 20;
  const chartW = width - (padX * 2);
  const chartH = height - (padY * 2);
  const n = points.length - 1;

  const coords = points.map((p, idx) => {
    const x = padX + ((idx / n) * chartW);
    const y = padY + ((100 - Number(p.score_pct || 0)) / 100) * chartH;
    return { x, y, pct: Number(p.score_pct || 0) };
  });

  const path = coords.map((c, idx) => `${idx === 0 ? "M" : "L"} ${c.x.toFixed(2)} ${c.y.toFixed(2)}`).join(" ");
  const last = coords[coords.length - 1];

  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0b1420]/85 p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-56 w-full">
        <defs>
          <linearGradient id="trendLine" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#fb7185" />
            <stop offset="50%" stopColor="#facc15" />
            <stop offset="100%" stopColor="#34d399" />
          </linearGradient>
        </defs>
        {[0, 25, 50, 75, 100].map((v) => {
          const y = padY + ((100 - v) / 100) * chartH;
          return <line key={v} x1={padX} y1={y} x2={width - padX} y2={y} stroke="rgba(255,255,255,0.12)" strokeWidth="1" />;
        })}
        <path d={path} fill="none" stroke="url(#trendLine)" strokeWidth="4" strokeLinecap="round" />
        {coords.map((c, idx) => (
          <circle key={idx} cx={c.x} cy={c.y} r={idx === coords.length - 1 ? 6 : 3.5} fill={idx === coords.length - 1 ? "#22d3ee" : "#cbd5e1"} />
        ))}
        <text x={last.x + 10} y={Math.max(18, last.y - 8)} fill="#a5f3fc" fontSize="13" fontWeight="700">
          {last.pct.toFixed(1)}%
        </text>
      </svg>
    </div>
  );
}

function buildNarrative(timeline) {
  const points = Array.isArray(timeline) ? timeline : [];
  if (points.length < 2) return "Aun no hay suficiente historial para narrativa detallada.";
  const first = points[0];
  const last = points[points.length - 1];
  const days = estimateDayDiff(first?.date, last?.date);
  const change = Number(last?.score_pct || 0) - Number(first?.score_pct || 0);
  const trend = change >= 0 ? "mejorando" : "empeorando";
  return `Hace ${days} dias estaba en ${String(first?.form_label || "racha normal").toLowerCase()} (${Number(first?.score_pct || 0).toFixed(1)}%) y ahora esta en ${String(last?.form_label || "racha normal").toLowerCase()} (${Number(last?.score_pct || 0).toFixed(1)}%), ${trend} ${Math.abs(change).toFixed(1)} puntos de forma.`;
}

function estimateDayDiff(startDate, endDate) {
  const start = new Date(`${startDate}T00:00:00`);
  const end = new Date(`${endDate}T00:00:00`);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return 0;
  const ms = Math.max(0, end.getTime() - start.getTime());
  return Math.round(ms / (1000 * 60 * 60 * 24));
}

function toPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}
