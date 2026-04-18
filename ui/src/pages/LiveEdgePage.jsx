import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchLiveEdgePerformance, fetchLiveEdgeRecommendations } from "../services/api.js";
import { getTeamLogoUrl } from "../utils/teamLogos.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

const SPORT_PATHS = {
  nba: "/nba",
  wnba: "/wnba",
  mlb: "/mlb",
  lmb: "/lmb",
  triple_a: "/triple-a",
  tennis: "/tennis",
  kbo: "/kbo",
  nhl: "/nhl",
  euroleague: "/euroleague",
  liga_mx: "/liga-mx",
  laliga: "/laliga",
  bundesliga: "/bundesliga",
  ligue1: "/ligue1",
};

function signalTone(signalState) {
  const key = String(signalState || "").toUpperCase();
  if (key === "FOLLOW") return "border-emerald-400/45 bg-emerald-500/15 text-emerald-100";
  if (key === "VALUE") return "border-cyan-300/45 bg-cyan-400/15 text-cyan-100";
  if (key === "HOLD") return "border-amber-300/45 bg-amber-400/15 text-amber-100";
  if (key === "WAIT" || key === "WATCH") return "border-white/20 bg-white/[0.06] text-white/85";
  if (key === "NO_BET" || key === "NO_SIGNAL") return "border-rose-400/45 bg-rose-500/15 text-rose-100";
  return "border-white/20 bg-white/[0.06] text-white/85";
}

function riskTone(riskLevel) {
  const key = String(riskLevel || "").toLowerCase();
  if (key === "low") return "border-emerald-400/45 bg-emerald-500/15 text-emerald-100";
  if (key === "medium") return "border-amber-300/45 bg-amber-400/15 text-amber-100";
  return "border-rose-400/45 bg-rose-500/15 text-rose-100";
}

function actionTone(action) {
  const key = String(action || "").toUpperCase();
  if (key === "BET") return "border-emerald-400/45 bg-emerald-500/15 text-emerald-100";
  if (key === "NO_BET") return "border-rose-400/45 bg-rose-500/15 text-rose-100";
  return "border-amber-300/45 bg-amber-400/15 text-amber-100";
}

function actionLabel(action) {
  const key = String(action || "").toUpperCase();
  if (key === "BET") return "Entrar";
  if (key === "NO_BET") return "No entrar";
  return "Esperar";
}

function TeamBadge({ sportKey, teamCode, sideLabel, score }) {
  const logoUrl = getTeamLogoUrl(sportKey, teamCode);
  const teamName = getTeamDisplayName(sportKey, teamCode);
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2">
      <div className="flex items-center gap-2">
        <img
          src={logoUrl || "/logos/default-team.svg"}
          alt={`Logo ${teamName}`}
          loading="lazy"
          onError={(e) => {
            e.currentTarget.onerror = null;
            e.currentTarget.src = "/logos/default-team.svg";
          }}
          className="h-8 w-8 rounded-full bg-white/95 p-1 object-contain"
        />
        <div>
          <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">{sideLabel}</p>
          <p className="text-sm font-semibold text-white">{teamName}</p>
        </div>
      </div>
      <span className="rounded-lg border border-cyan-300/35 bg-cyan-400/10 px-2.5 py-1 text-base font-black text-cyan-100">
        {score ?? "-"}
      </span>
    </div>
  );
}

function metricOrSample(value, hasSample, digits = 1, suffix = "") {
  if (!hasSample) return "Sin muestra";
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "Sin muestra";
  return `${n.toFixed(digits)}${suffix}`;
}

export default function LiveEdgePage() {
  const navigate = useNavigate();
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [performance, setPerformance] = useState(null);
  const [performanceWindow, setPerformanceWindow] = useState("30d");

  const loadLiveEdge = useCallback(async ({ silent = false } = {}) => {
    if (silent) setRefreshing(true);
    else setLoading(true);
    setError("");
    try {
      const [result, perf] = await Promise.all([
        fetchLiveEdgeRecommendations({ includeNoSignal: false, limit: 180, track: true, autosettle: true }),
        fetchLiveEdgePerformance({ autosettle: true, window: performanceWindow }),
      ]);
      setPayload(result);
      setPerformance(perf);
    } catch (err) {
      setError(err?.message || "No se pudo cargar Live Edge.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [performanceWindow]);

  useEffect(() => {
    loadLiveEdge();
  }, [loadLiveEdge]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      loadLiveEdge({ silent: true });
    }, 25000);
    return () => clearInterval(intervalId);
  }, [loadLiveEdge]);

  const recommendations = useMemo(
    () => (Array.isArray(payload?.recommendations) ? payload.recommendations : []),
    [payload],
  );
  const summary = payload?.summary || {};
  const perfSummary = performance?.summary || {};
  const perfByMarket = Array.isArray(performance?.by_market) ? performance.by_market : [];
  const perfBySport = Array.isArray(performance?.by_sport) ? performance.by_sport : [];
  const perfByPattern = Array.isArray(performance?.by_pattern_support) ? performance.by_pattern_support : [];
  const hasGradedSample = Number(perfSummary?.graded_bets || 0) > 0;

  return (
    <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
      <section className="mb-6 rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(34,211,238,0.16),transparent_34%),linear-gradient(180deg,rgba(16,23,34,0.96),rgba(10,16,26,0.98))] p-5 shadow-[0_20px_56px_rgba(0,0,0,0.24)]">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-200/85">Live Edge</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Recomendaciones live separadas del modelo pregame</h2>
            <p className="mt-1 text-sm text-white/60">
              Este modulo no altera ni mezcla resultados con las predicciones base.
            </p>
          </div>
          <button
            type="button"
            onClick={() => loadLiveEdge({ silent: true })}
            className="rounded-xl border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/18"
          >
            {refreshing ? "Actualizando..." : "Actualizar ahora"}
          </button>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Separation tag</p>
            <p className="mt-2 text-sm font-semibold text-cyan-100">{payload?.separation_tag || "LIVE_EDGE_ONLY"}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Eventos live escaneados</p>
            <p className="mt-2 text-3xl font-semibold text-white">{summary?.events_live_scanned || 0}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Recomendaciones</p>
            <p className="mt-2 text-3xl font-semibold text-emerald-300">{summary?.recommendations || 0}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Deportes activos</p>
            <p className="mt-2 text-3xl font-semibold text-amber-200">{summary?.sports_active || 0}</p>
          </div>
        </div>
        <p className="mt-4 text-xs text-white/65">
          Guia rapida: <span className="text-emerald-200">Entrar</span> = hay jugada live sugerida,
          <span className="mx-1 text-amber-200">Esperar</span> = monitorear,
          <span className="mx-1 text-rose-200">No entrar</span> = evitar entrada por ahora.
        </p>
      </section>

      <section className="mb-6 rounded-[26px] border border-white/10 bg-[linear-gradient(180deg,rgba(14,20,30,0.96),rgba(10,15,24,0.98))] p-4 shadow-[0_16px_40px_rgba(0,0,0,0.22)]">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-cyan-100/85">Performance Live Edge</h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1">
              {["today", "7d", "30d", "all"].map((windowKey) => (
                <button
                  key={windowKey}
                  type="button"
                  onClick={() => setPerformanceWindow(windowKey)}
                  className={`rounded-lg px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.1em] transition ${
                    performanceWindow === windowKey
                      ? "bg-cyan-300/25 text-cyan-100"
                      : "text-white/65 hover:bg-white/[0.06] hover:text-white"
                  }`}
                >
                  {windowKey}
                </button>
              ))}
            </div>
            <span className="text-xs text-white/60">
              Settled ahora: {performance?.settle_info?.settled || 0}
            </span>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-4 xl:grid-cols-8">
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Bets graded</p>
            <p className="mt-1 text-xl font-semibold text-white">{perfSummary?.graded_bets || 0}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Hit rate</p>
            <p className="mt-1 text-xl font-semibold text-emerald-300">
              {metricOrSample(perfSummary?.hit_rate, hasGradedSample, 1, "%")}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">ROI (1u)</p>
            <p className={`mt-1 text-xl font-semibold ${Number(perfSummary?.roi || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
              {metricOrSample(perfSummary?.roi, hasGradedSample, 1, "%")}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Units</p>
            <p className={`mt-1 text-xl font-semibold ${Number(perfSummary?.units || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
              {metricOrSample(perfSummary?.units, hasGradedSample, 2)}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Open tracked</p>
            <p className="mt-1 text-xl font-semibold text-amber-200">{perfSummary?.open_entries || 0}</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">W-L-P</p>
            <p className="mt-1 text-xl font-semibold text-white">
              {hasGradedSample ? `${perfSummary?.won || 0}-${perfSummary?.lost || 0}-${perfSummary?.push || 0}` : "Sin muestra"}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Clear margin</p>
            <p className={`mt-1 text-xl font-semibold ${Number(perfSummary?.avg_clear_margin || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
              {metricOrSample(perfSummary?.avg_clear_margin, hasGradedSample, 2)}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">CLV proxy</p>
            <p className="mt-1 text-xl font-semibold text-cyan-200">
              {metricOrSample(perfSummary?.clv_proxy, hasGradedSample, 1, "%")}
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
            <p className="text-[10px] uppercase tracking-[0.12em] text-white/45">Ventana</p>
            <p className="mt-1 text-xl font-semibold text-white">{String(performance?.window || performanceWindow).toUpperCase()}</p>
          </div>
        </div>
        <p className="mt-2 text-[11px] text-white/55">
          CLV proxy = score de frescura de linea segun fuente (live/reference/model), no reemplaza CLV real de casa.
        </p>
        {!hasGradedSample && (
          <p className="mt-1 text-[11px] text-amber-200/85">
            Aun no hay picks liquidados en esta ventana. Cuando terminen juegos, se activan Hit Rate, ROI, Units y W-L-P.
          </p>
        )}

        {perfByMarket.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-xl border border-white/10">
            <table className="min-w-full text-left text-xs text-white/80">
              <thead className="bg-white/[0.04] text-[10px] uppercase tracking-[0.12em] text-white/55">
                <tr>
                  <th className="px-3 py-2">Mercado</th>
                  <th className="px-3 py-2">Bets</th>
                  <th className="px-3 py-2">W-L-P</th>
                  <th className="px-3 py-2">Hit %</th>
                  <th className="px-3 py-2">ROI %</th>
                  <th className="px-3 py-2">Units</th>
                  <th className="px-3 py-2">Clear margin</th>
                </tr>
              </thead>
              <tbody>
                {perfByMarket.slice(0, 12).map((row) => (
                  <tr key={row.key} className="border-t border-white/10">
                    <td className="px-3 py-2 font-semibold text-cyan-100">{row.key}</td>
                    <td className="px-3 py-2">{row.bets}</td>
                    <td className="px-3 py-2">{row.won}-{row.lost}-{row.push}</td>
                    <td className="px-3 py-2">{Number(row.hit_rate || 0).toFixed(1)}%</td>
                    <td className="px-3 py-2">{Number(row.roi || 0).toFixed(1)}%</td>
                    <td className={`px-3 py-2 ${Number(row.units || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {Number(row.units || 0).toFixed(2)}
                    </td>
                    <td className={`px-3 py-2 ${Number(row.avg_clear_margin || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {Number(row.avg_clear_margin || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {perfBySport.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-xl border border-white/10">
            <table className="min-w-full text-left text-xs text-white/80">
              <thead className="bg-white/[0.04] text-[10px] uppercase tracking-[0.12em] text-white/55">
                <tr>
                  <th className="px-3 py-2">Deporte</th>
                  <th className="px-3 py-2">Bets</th>
                  <th className="px-3 py-2">W-L-P</th>
                  <th className="px-3 py-2">Hit %</th>
                  <th className="px-3 py-2">ROI %</th>
                  <th className="px-3 py-2">Units</th>
                </tr>
              </thead>
              <tbody>
                {perfBySport.slice(0, 12).map((row) => (
                  <tr key={`sport-${row.key}`} className="border-t border-white/10">
                    <td className="px-3 py-2 font-semibold text-cyan-100">{String(row.key || "").toUpperCase()}</td>
                    <td className="px-3 py-2">{row.bets}</td>
                    <td className="px-3 py-2">{row.won}-{row.lost}-{row.push}</td>
                    <td className="px-3 py-2">{Number(row.hit_rate || 0).toFixed(1)}%</td>
                    <td className="px-3 py-2">{Number(row.roi || 0).toFixed(1)}%</td>
                    <td className={`px-3 py-2 ${Number(row.units || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {Number(row.units || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {perfByPattern.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-xl border border-white/10">
            <table className="min-w-full text-left text-xs text-white/80">
              <thead className="bg-white/[0.04] text-[10px] uppercase tracking-[0.12em] text-white/55">
                <tr>
                  <th className="px-3 py-2">Pattern support</th>
                  <th className="px-3 py-2">Bets</th>
                  <th className="px-3 py-2">W-L-P</th>
                  <th className="px-3 py-2">Hit %</th>
                  <th className="px-3 py-2">ROI %</th>
                  <th className="px-3 py-2">Units</th>
                </tr>
              </thead>
              <tbody>
                {perfByPattern.slice(0, 8).map((row) => (
                  <tr key={`pattern-${row.key}`} className="border-t border-white/10">
                    <td className="px-3 py-2 font-semibold text-violet-100">{String(row.key || "N/A").toUpperCase()}</td>
                    <td className="px-3 py-2">{row.bets}</td>
                    <td className="px-3 py-2">{row.won}-{row.lost}-{row.push}</td>
                    <td className="px-3 py-2">{Number(row.hit_rate || 0).toFixed(1)}%</td>
                    <td className="px-3 py-2">{Number(row.roi || 0).toFixed(1)}%</td>
                    <td className={`px-3 py-2 ${Number(row.units || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                      {Number(row.units || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {loading ? (
        <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
          Cargando Live Edge...
        </div>
      ) : error ? (
        <div className="rounded-[30px] border border-rose-400/30 bg-rose-500/10 p-10 text-center text-rose-100">
          {error}
        </div>
      ) : recommendations.length === 0 ? (
        <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
          No hay recomendaciones Live Edge en este momento.
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-3">
          {recommendations.map((item, idx) => {
            const sportKey = String(item?.sport || "").toLowerCase();
            const confText = Number.isFinite(Number(item?.pregame_confidence))
              ? `${Math.round(Number(item?.pregame_confidence))}%`
              : "N/A";
            const livePickTeamName = item?.live_pick_team
              ? getTeamDisplayName(sportKey, item.live_pick_team)
              : "";
            const fallbackPickText = item?.entry_action === "BET"
              ? `ML en vivo: ${livePickTeamName || item?.pregame_pick_team || item?.pregame_pick_raw || "Pick principal"}`
              : item?.entry_action === "NO_BET"
                ? "No entrar por ahora."
                : "Esperar mejor momento de entrada.";
            const livePickText = String(item?.live_pick_text || fallbackPickText);
            const boardPath = SPORT_PATHS[sportKey];
            const cardKey = `${item?.sport || "sport"}-${item?.game_id || idx}-${item?.generated_at || "gen"}`;

            return (
              <article
                key={cardKey}
                className="rounded-[26px] border border-white/10 bg-[linear-gradient(180deg,rgba(14,20,30,0.96),rgba(10,15,24,0.98))] p-4 shadow-[0_16px_40px_rgba(0,0,0,0.22)]"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.14em] text-cyan-100/85">
                      {item?.sport_label || sportKey.toUpperCase()} - {item?.status_clock || "En vivo"}
                    </p>
                    <p className="mt-1 text-xs text-white/55">{item?.event_date || ""} {item?.event_time || ""}</p>
                  </div>
                  <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${signalTone(item?.signal_state)}`}>
                    {item?.recommendation_short || "Signal"}
                  </span>
                </div>

                <div className="mt-3 grid gap-2">
                  <TeamBadge
                    sportKey={sportKey}
                    teamCode={item?.away_team}
                    sideLabel="Visitante"
                    score={item?.away_score}
                  />
                  <TeamBadge
                    sportKey={sportKey}
                    teamCode={item?.home_team}
                    sideLabel="Local"
                    score={item?.home_score}
                  />
                </div>

                <div className="mt-3 flex items-center justify-between gap-3">
                  <span className="rounded-xl border border-white/15 bg-white/[0.05] px-3 py-1.5 text-sm font-semibold text-white">
                    Score live por equipo
                  </span>
                  <span className={`rounded-xl border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] ${riskTone(item?.risk_level)}`}>
                    Riesgo {item?.risk_level || "medium"}
                  </span>
                </div>

                <div className="mt-3 rounded-xl border border-cyan-300/25 bg-cyan-400/10 px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-[11px] uppercase tracking-[0.12em] text-cyan-100/85">Jugada sugerida ahora</p>
                    <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${actionTone(item?.entry_action)}`}>
                      {actionLabel(item?.entry_action)}
                    </span>
                  </div>
                  <p className="mt-1 text-sm font-semibold text-white">{livePickText}</p>
                  <p className="mt-1 text-xs text-cyan-100/75">
                    Mercado: {item?.live_pick_market || "NO_BET"}{item?.live_pick_line ? ` (${item.live_pick_line})` : ""}
                  </p>
                  {item?.live_pick_condition && (
                    <p className="mt-1 text-xs text-cyan-100/85">
                      Condicion: {item.live_pick_condition}
                    </p>
                  )}
                  <p className="mt-1 text-[11px] text-white/60">
                    Fuente de linea: {item?.odds_line_source || "n/a"}
                  </p>
                </div>

                <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-white/55">Recomendacion</p>
                  <p className="mt-1 text-sm font-semibold text-white">{item?.recommendation || "Monitor"}</p>
                  <p className="mt-1 text-xs text-white/70">{item?.reason || "Sin razon adicional."}</p>
                </div>

                <div className="mt-3 rounded-xl border border-violet-300/20 bg-violet-400/10 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-violet-100/85">Pattern edge</p>
                  <p className="mt-1 text-sm font-semibold text-white">
                    Soporte: {item?.pattern_edge?.support_level || "LOW"} | Alineacion: {item?.pattern_edge?.alignment || "neutral"}
                  </p>
                  <p className="mt-1 text-xs text-violet-100/80">
                    {item?.pattern_edge?.explanation || "Sin patron util para este evento."}
                  </p>
                  <p className="mt-1 text-[11px] text-violet-100/70">
                    Muestra: {item?.pattern_edge?.samples ?? 0} | Confianza lado: {Number(item?.pattern_edge?.confidence_pct || 0).toFixed(1)}%
                  </p>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-white/75">
                  <span className="rounded-full border border-white/12 bg-white/[0.04] px-2.5 py-1">
                    Pick base: {item?.pregame_pick_team || item?.pregame_pick_raw || "N/A"}
                  </span>
                  <span className="rounded-full border border-white/12 bg-white/[0.04] px-2.5 py-1">
                    Confianza: {confText}
                  </span>
                  <span className="rounded-full border border-white/12 bg-white/[0.04] px-2.5 py-1">
                    Strength: {item?.strength || "NORMAL"}
                  </span>
                  <span className="rounded-full border border-white/12 bg-white/[0.04] px-2.5 py-1">
                    Datos live: {item?.stats_snapshot?.coverage || "limited"}
                  </span>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-[0.12em] text-cyan-200/70">
                    {payload?.model_key || "live_edge_v1"}
                  </span>
                  {boardPath ? (
                    <button
                      type="button"
                      onClick={() => navigate(`${boardPath}?date=${item?.event_date || ""}`)}
                      className="rounded-lg border border-cyan-300/35 bg-cyan-300/10 px-3 py-1.5 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-300/20"
                    >
                      Ver board
                    </button>
                  ) : null}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </main>
  );
}
