import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchTodayPredictions } from "../services/api.js";
import { resolveEventTeams, resolveSidePick } from "../utils/teams.js";
import { expandTeamCodeInText, getTeamDisplayName } from "../utils/teamNames.js";
import { getTeamLogoUrl } from "../utils/teamLogos.js";
import DetailModal from "../components/DetailModal.jsx";

const LIVE_SPORTS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "triple_a", label: "Triple-A", path: "/triple-a" },
  { key: "tennis", label: "Tennis", path: "/tennis" },
  { key: "kbo", label: "KBO", path: "/kbo" },
  { key: "nhl", label: "NHL", path: "/nhl" },
  { key: "euroleague", label: "EuroLeague", path: "/euroleague" },
  { key: "liga_mx", label: "Liga MX", path: "/liga-mx" },
  { key: "laliga", label: "LaLiga", path: "/laliga" },
];

function toHitValue(value) {
  if (value === true || value === false) return value;
  if (value === 1 || value === "1") return true;
  if (value === 0 || value === "0") return false;
  return null;
}

function isPendingPick(value) {
  const v = String(value ?? "").trim().toUpperCase();
  return !v || ["PENDIENTE", "N/A", "NAN", "RECONSTRUIDO", "PASS", "PASAR"].includes(v);
}

function toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function toLine(value) {
  const n = toFiniteNumber(value);
  if (n === null || n === 0) return null;
  return n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1);
}

function normalizeTotalDirection(rawPick) {
  const txt = String(rawPick || "").toUpperCase();
  if (txt.includes("OVER")) return "Over";
  if (txt.includes("UNDER")) return "Under";
  return null;
}

function isLiveEvent(event) {
  const state = String(event?.status_state || "").trim().toLowerCase();
  if (state === "in" || state === "live") return true;

  const desc = String(event?.status_description || "").trim().toLowerCase();
  const detail = String(event?.status_detail || "").trim().toLowerCase();
  return (
    desc.includes("en vivo") ||
    desc.includes("live") ||
    detail.includes("en vivo") ||
    detail.includes("q") ||
    detail.includes("period")
  );
}

function formatClock(event) {
  const shortState = String(event?.status_description || "").trim();
  const detail = String(event?.status_detail || "").trim();
  if (shortState && detail && shortState !== detail) return `${shortState} - ${detail}`;
  return shortState || detail || "En vivo";
}

function formatScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(n) : "-";
}

function pickBadgeTone(hit) {
  if (hit === true) return "border-emerald-400/45 bg-emerald-500/12 text-emerald-100";
  if (hit === false) return "border-rose-400/45 bg-rose-500/12 text-rose-100";
  return "border-white/15 bg-white/[0.045] text-white/80";
}

function resolveSummaryPicks(event, sportKey, teams) {
  const picks = [];

  const mlRaw = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  if (!isPendingPick(mlRaw)) {
    picks.push({
      label: "ML",
      value: mlRaw,
      hit: toHitValue(event.full_game_hit),
    });
  }

  const spreadRaw = String(event.spread_pick || "").trim();
  const spreadLine = toLine(event.spread_line_signed)
    || toLine(event.home_spread)
    || toLine(event.closing_spread_line);
  if (!isPendingPick(spreadRaw) || spreadLine) {
    const spreadPick = !isPendingPick(spreadRaw)
      ? (expandTeamCodeInText(sportKey, resolveSidePick(spreadRaw, teams)) || spreadRaw)
      : "Linea";
    picks.push({
      label: "HCP",
      value: spreadLine ? `${spreadPick} ${spreadLine}` : spreadPick,
      hit: toHitValue(event.correct_spread),
    });
  }

  const totalRaw = event.total_recommended_pick || event.total_pick;
  const totalDirection = normalizeTotalDirection(totalRaw);
  const totalLine = toFiniteNumber(event.closing_total_line ?? event.odds_over_under ?? event.total_line);
  if (!isPendingPick(totalRaw) || totalLine) {
    const totalText = totalDirection || String(totalRaw || "Total");
    const lineText = Number.isFinite(totalLine) ? ` ${totalLine.toFixed(1)}` : "";
    picks.push({
      label: "O/U",
      value: `${totalText}${lineText}`.trim(),
      hit: toHitValue(event.correct_total_adjusted ?? event.correct_total),
    });
  }

  const q1Raw = String(event.q1_pick || "").trim();
  if (!isPendingPick(q1Raw)) {
    picks.push({
      label: "Q1",
      value: expandTeamCodeInText(sportKey, resolveSidePick(q1Raw, teams)) || q1Raw,
      hit: toHitValue(event.q1_hit),
    });
  }

  const h1Raw = String(event.h1_pick || "").trim();
  if (!isPendingPick(h1Raw)) {
    picks.push({
      label: "HT",
      value: expandTeamCodeInText(sportKey, resolveSidePick(h1Raw, teams)) || h1Raw,
      hit: toHitValue(event.h1_hit),
    });
  }

  const bttsRaw = String(event.btts_recommended_pick || event.btts_pick || "").trim();
  if (!isPendingPick(bttsRaw)) {
    picks.push({
      label: "BTTS",
      value: expandTeamCodeInText(sportKey, resolveSidePick(bttsRaw, teams)) || bttsRaw,
      hit: toHitValue(event.correct_btts_adjusted ?? event.correct_btts ?? event.correct_btts_base),
    });
  }

  const cornersRaw = String(event.corners_pick || "").trim();
  if (!isPendingPick(cornersRaw)) {
    picks.push({
      label: "Corners",
      value: cornersRaw,
      hit: toHitValue(event.correct_corners_adjusted ?? event.correct_corners_base ?? event.correct_corners),
    });
  }

  return picks.slice(0, 5);
}

function TeamCell({ sportKey, abbr, side }) {
  const logoUrl = getTeamLogoUrl(sportKey, abbr);
  const fullName = getTeamDisplayName(sportKey, abbr);

  return (
    <div className="flex items-center gap-2">
      {logoUrl ? (
        <img
          src={logoUrl}
          alt={`Logo ${fullName}`}
          className="h-7 w-7 rounded-full bg-white/95 p-1 object-contain"
          loading="lazy"
          onError={(e) => {
            e.currentTarget.onerror = null;
            e.currentTarget.src = "/logos/default-team.svg";
          }}
        />
      ) : (
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-[10px] font-semibold text-white/70">
          {String(abbr || side).slice(0, 3).toUpperCase()}
        </div>
      )}
      <span className="text-sm text-white/90">{fullName}</span>
    </div>
  );
}

export default function LiveBoardPage() {
  const navigate = useNavigate();
  const [eventsBySport, setEventsBySport] = useState({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [activeEvent, setActiveEvent] = useState(null);

  const loadLiveBoards = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setLoading(true);
    if (silent) setRefreshing(true);

    try {
      const results = await Promise.allSettled(
        LIVE_SPORTS.map((sport) => fetchTodayPredictions(sport.key)),
      );

      const grouped = {};

      for (let idx = 0; idx < LIVE_SPORTS.length; idx += 1) {
        const sport = LIVE_SPORTS[idx];
        const result = results[idx];
        if (result.status !== "fulfilled" || !Array.isArray(result.value)) continue;

        const liveEvents = result.value.filter(isLiveEvent).map((event) => ({
          ...event,
          sportKey: sport.key,
          sportLabel: sport.label,
          sportPath: sport.path,
        }));

        if (liveEvents.length > 0) {
          grouped[sport.key] = liveEvents;
        }
      }

      setEventsBySport(grouped);
      setLastUpdated(new Date());
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadLiveBoards();
  }, [loadLiveBoards]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      loadLiveBoards({ silent: true });
    }, 25000);
    return () => clearInterval(intervalId);
  }, [loadLiveBoards]);

  const sportsWithLive = useMemo(
    () => LIVE_SPORTS.filter((sport) => Array.isArray(eventsBySport[sport.key]) && eventsBySport[sport.key].length > 0),
    [eventsBySport],
  );
  const totalLiveEvents = useMemo(
    () => sportsWithLive.reduce((sum, sport) => sum + (eventsBySport[sport.key]?.length || 0), 0),
    [eventsBySport, sportsWithLive],
  );

  return (
    <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
      <section className="mb-6 rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.12),transparent_32%),linear-gradient(180deg,rgba(16,23,34,0.96),rgba(10,16,26,0.98))] p-5 shadow-[0_20px_56px_rgba(0,0,0,0.24)]">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-200/85">Live Board</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Eventos en vivo con picks resumidos</h2>
            <p className="mt-1 text-sm text-white/60">Vista tipo lista para monitorear rapido sin salir de tu flujo.</p>
          </div>
          <button
            type="button"
            onClick={() => loadLiveBoards({ silent: true })}
            className="rounded-xl border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/18"
          >
            {refreshing ? "Actualizando..." : "Actualizar ahora"}
          </button>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Eventos live</p>
            <p className="mt-2 text-3xl font-semibold text-emerald-300">{totalLiveEvents}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Deportes activos</p>
            <p className="mt-2 text-3xl font-semibold text-amber-200">{sportsWithLive.length}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Ultima actualizacion</p>
            <p className="mt-2 text-base font-semibold text-cyan-100">
              {lastUpdated ? lastUpdated.toLocaleTimeString() : "--:--:--"}
            </p>
          </div>
        </div>
      </section>

      {loading ? (
        <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
          Cargando eventos en vivo...
        </div>
      ) : sportsWithLive.length === 0 ? (
        <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
          No hay eventos en vivo detectados en este momento.
        </div>
      ) : (
        <div className="space-y-6">
          {sportsWithLive.map((sport) => (
            <section
              key={sport.key}
              className="overflow-hidden rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(14,20,30,0.96),rgba(10,15,24,0.98))] shadow-[0_16px_40px_rgba(0,0,0,0.22)]"
            >
              <div className="flex items-center justify-between gap-4 border-b border-white/10 bg-cyan-500/10 px-4 py-3">
                <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-cyan-100">
                  {sport.label} - {eventsBySport[sport.key].length} en vivo
                </h3>
                <button
                  type="button"
                  onClick={() => navigate(sport.path)}
                  className="text-xs font-semibold text-cyan-100/85 underline underline-offset-4 transition hover:text-white"
                >
                  Abrir board
                </button>
              </div>

              <div className="divide-y divide-white/8">
                {eventsBySport[sport.key].map((event) => {
                  const teams = resolveEventTeams(event);
                  const picks = resolveSummaryPicks(event, sport.key, teams);
                  return (
                    <div
                      key={`${sport.key}-${event.game_id || `${event.date}-${event.time}-${teams.homeTeam}`}`}
                      onClick={() => setActiveEvent(event)}
                      className="w-full cursor-pointer px-4 py-3 text-left transition hover:bg-white/[0.035]"
                    >
                      <div className="grid gap-4 xl:grid-cols-[220px_minmax(0,1fr)_minmax(320px,420px)] xl:items-center">
                        <div className="flex items-start gap-3">
                          <span className="mt-1 inline-flex h-3 w-3 shrink-0 rounded-full bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.9)]" />
                          <div>
                            <p className="text-sm font-semibold text-white/90">{event.time || "LIVE"}</p>
                            <p className="text-xs text-rose-200/90">{formatClock(event)}</p>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <TeamCell sportKey={sport.key} abbr={teams.awayTeam} side="away" />
                          <TeamCell sportKey={sport.key} abbr={teams.homeTeam} side="home" />
                        </div>

                        <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`${sport.path}?date=${event.date}`);
                            }}
                            className="rounded-xl border border-cyan-300/35 bg-cyan-300/10 px-3 py-1.5 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-300/20"
                          >
                            Ver board
                          </button>
                          <span className="rounded-xl border border-white/15 bg-white/[0.04] px-3 py-1.5 text-sm font-semibold text-white/90">
                            {formatScore(event.away_score)} - {formatScore(event.home_score)}
                          </span>
                          {picks.length === 0 ? (
                            <span className="rounded-xl border border-white/15 bg-white/[0.04] px-3 py-1.5 text-xs text-white/70">
                              Sin picks para resumir
                            </span>
                          ) : (
                            picks.map((pick) => (
                              <span
                                key={`${event.game_id || event.time}-${pick.label}`}
                                className={`rounded-xl border px-3 py-1.5 text-xs font-semibold ${pickBadgeTone(pick.hit)}`}
                              >
                                {pick.label}: {pick.value}
                              </span>
                            ))
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}

      {activeEvent && (
        <DetailModal
          event={activeEvent}
          sportKey={activeEvent.sportKey}
          onClose={() => setActiveEvent(null)}
        />
      )}
    </main>
  );
}
