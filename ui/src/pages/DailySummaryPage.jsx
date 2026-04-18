import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import DetailModal from "../components/DetailModal.jsx";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { fetchAvailableDates, fetchPredictionsByDate } from "../services/api.js";
import { resolveEventTeams, resolveSidePick } from "../utils/teams.js";
import { expandTeamCodeInText, getTeamDisplayName } from "../utils/teamNames.js";
import { getTeamLogoUrl } from "../utils/teamLogos.js";

const SPORTS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "wnba", label: "WNBA", path: "/wnba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "lmb", label: "LMB", path: "/lmb" },
  { key: "triple_a", label: "Triple-A", path: "/triple-a" },
  { key: "tennis", label: "Tennis", path: "/tennis" },
  { key: "kbo", label: "KBO", path: "/kbo" },
  { key: "nhl", label: "NHL", path: "/nhl" },
  { key: "euroleague", label: "EuroLeague", path: "/euroleague" },
  { key: "liga_mx", label: "Liga MX", path: "/liga-mx" },
  { key: "laliga", label: "LaLiga", path: "/laliga" },
];

function toYmdLocal(dateObj) {
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, "0");
  const d = String(dateObj.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

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

function normalizeTotalDirection(rawPick) {
  const txt = String(rawPick || "").toUpperCase();
  if (txt.includes("OVER")) return "Over";
  if (txt.includes("UNDER")) return "Under";
  return null;
}

function extractNumericFromText(value) {
  const match = String(value ?? "").match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : null;
}

function formatLineValue(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function buildTotalPickLabel(rawPick, fallbackLineValue) {
  const raw = String(rawPick || "").trim();
  if (!raw) return "N/A";

  const direction = normalizeTotalDirection(raw);
  if (!direction) return raw;

  const explicitLine = extractNumericFromText(raw);
  const explicitLineDisplay = formatLineValue(explicitLine);
  if (explicitLineDisplay) return `${direction} ${explicitLineDisplay}`;

  const fallbackLineDisplay = formatLineValue(fallbackLineValue);
  if (fallbackLineDisplay) return `${direction} ${fallbackLineDisplay}`;

  return direction;
}

function hasFinalResult(event) {
  const state = String(event?.status_state || "").trim().toLowerCase();
  return (
    event?.result_available === true ||
    ["post", "final", "completed"].includes(state) ||
    Boolean(String(event?.final_score_text || "").trim())
  );
}

function rowBorderClass(event) {
  const fullHit = toHitValue(event?.full_game_hit);
  if (fullHit === true) return "border-emerald-400/60";
  if (fullHit === false) return "border-rose-400/60";
  return "border-white/10";
}

function pickBadgeTone(hit) {
  if (hit === true) return "border-emerald-400/45 bg-emerald-500/12 text-emerald-100";
  if (hit === false) return "border-rose-400/45 bg-rose-500/12 text-rose-100";
  return "border-white/15 bg-white/[0.045] text-white/80";
}

function resolveQ1ShortLabel(sportKey) {
  const sport = String(sportKey || "").trim().toLowerCase();
  if (["mlb", "lmb", "kbo", "ncaa_baseball", "triple_a"].includes(sport)) return "1IN";
  if (["nba", "wnba", "euroleague"].includes(sport)) return "1Q";
  if (sport === "nhl") return "1P";
  if (["liga_mx", "laliga", "bundesliga", "ligue1"].includes(sport)) return "1T";
  return "Q1";
}

function resolveSummaryPicks(event, sportKey, teams) {
  const picks = [];

  const mlRaw = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  if (!isPendingPick(mlRaw)) {
    picks.push({ label: "ML", value: mlRaw, hit: toHitValue(event.full_game_hit) });
  }

  const spreadRaw = String(event.spread_pick || "").trim();
  if (!isPendingPick(spreadRaw)) {
    picks.push({
      label: "HCP",
      value: expandTeamCodeInText(sportKey, resolveSidePick(spreadRaw, teams)) || spreadRaw,
      hit: toHitValue(event.correct_spread),
    });
  }

  const totalRaw = event.total_recommended_pick || event.total_pick;
  if (!isPendingPick(totalRaw)) {
    const value = buildTotalPickLabel(totalRaw, event.closing_total_line ?? event.odds_over_under);
    picks.push({
      label: "O/U",
      value,
      hit: toHitValue(event.correct_total_adjusted ?? event.correct_total),
    });
  }

  const hasTotalHitsMarket =
    Object.prototype.hasOwnProperty.call(event, "total_hits_pick")
    || Object.prototype.hasOwnProperty.call(event, "total_hits_recommended_pick")
    || Object.prototype.hasOwnProperty.call(event, "total_hits_model_available");
  const totalHitsRaw = event.total_hits_recommended_pick || event.total_hits_pick;
  if (hasTotalHitsMarket) {
    const normalizedRaw = String(totalHitsRaw || "PASS");
    const value = buildTotalPickLabel(
      normalizedRaw,
      event.closing_total_hits_line ?? event.odds_total_hits_event ?? event.odds_total_hits,
    );
    picks.push({
      label: "HITS",
      value,
      hit: toHitValue(event.correct_total_hits),
    });
  }

  const q1Raw = String(event.q1_pick || "").trim();
  if (!isPendingPick(q1Raw)) {
    picks.push({
      label: resolveQ1ShortLabel(sportKey),
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

  const h1OuRaw = String(event.h1_over15_recommended_pick || event.h1_over15_pick || "").trim();
  if (!isPendingPick(h1OuRaw)) {
    picks.push({
      label: "1T O/U",
      value: h1OuRaw,
      hit: toHitValue(event.h1_over15_hit ?? event.correct_h1_over15_adjusted ?? event.correct_h1_over15_base),
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

  return picks;
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

function parseTimeValue(event) {
  const raw = String(event?.time || "").trim();
  const match = raw.match(/^(\d{1,2}):(\d{2})/);
  if (!match) return Number.MAX_SAFE_INTEGER;
  return Number(match[1]) * 60 + Number(match[2]);
}

export default function DailySummaryPage() {
  const { uiTheme } = useAppSettings();
  const isDashboardPro = uiTheme === "dashboard_pro";
  const navigate = useNavigate();
  const [eventsBySport, setEventsBySport] = useState({});
  const [expandedSports, setExpandedSports] = useState({});
  const [dashboardSidebarCompact, setDashboardSidebarCompact] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [activeEvent, setActiveEvent] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [availableDates, setAvailableDates] = useState([]);
  const [calendarMonth, setCalendarMonth] = useState(new Date());

  const refreshAvailableDates = useCallback(async () => {
    const responses = await Promise.allSettled(
      SPORTS.map((sport) => fetchAvailableDates(sport.key)),
    );
    const merged = new Set();

    responses.forEach((res) => {
      if (res.status === "fulfilled" && Array.isArray(res.value)) {
        res.value.forEach((dateStr) => merged.add(dateStr));
      }
    });

    const sorted = [...merged].sort();
    setAvailableDates(sorted);
    return sorted;
  }, []);

  const loadSummaryByDate = useCallback(async (dateStr, { silent = false } = {}) => {
    if (!silent) {
      setLoading(true);
      setError("");
    }
    if (silent) setRefreshing(true);

    try {
      const results = await Promise.allSettled(
        SPORTS.map((sport) => fetchPredictionsByDate(sport.key, dateStr)),
      );
      const grouped = {};

      for (let idx = 0; idx < SPORTS.length; idx += 1) {
        const sport = SPORTS[idx];
        const result = results[idx];
        if (result.status !== "fulfilled" || !Array.isArray(result.value) || result.value.length === 0) continue;

        grouped[sport.key] = [...result.value]
          .map((event) => ({ ...event, sportKey: sport.key, sportLabel: sport.label, sportPath: sport.path }))
          .sort((a, b) => parseTimeValue(a) - parseTimeValue(b));
      }

      setEventsBySport(grouped);
      setSelectedDate(dateStr);
      const d = new Date(`${dateStr}T00:00:00`);
      if (!Number.isNaN(d.getTime())) setCalendarMonth(d);
      setLastUpdated(new Date());
    } catch {
      setEventsBySport({});
      setSelectedDate(dateStr);
      setError("No se pudo cargar el resumen para esa fecha.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const loadTodaySummary = useCallback(async () => {
    const today = toYmdLocal(new Date());
    await loadSummaryByDate(today);
  }, [loadSummaryByDate]);

  useEffect(() => {
    let mounted = true;

    async function bootstrap() {
      setLoading(true);
      setError("");
      try {
        const dates = await refreshAvailableDates();
        if (!mounted) return;

        const today = toYmdLocal(new Date());
        const initial = dates.includes(today) ? today : (dates[dates.length - 1] || today);
        await loadSummaryByDate(initial);
      } catch {
        if (!mounted) return;
        await loadTodaySummary();
      }
    }

    bootstrap();
    return () => {
      mounted = false;
    };
  }, [loadSummaryByDate, loadTodaySummary, refreshAvailableDates]);

  const sportsWithData = useMemo(
    () => SPORTS.filter((sport) => Array.isArray(eventsBySport[sport.key]) && eventsBySport[sport.key].length > 0),
    [eventsBySport],
  );
  const totalEvents = useMemo(
    () => sportsWithData.reduce((sum, sport) => sum + (eventsBySport[sport.key]?.length || 0), 0),
    [eventsBySport, sportsWithData],
  );
  const settledEvents = useMemo(
    () =>
      sportsWithData.reduce(
        (sum, sport) => sum + eventsBySport[sport.key].filter((event) => toHitValue(event.full_game_hit) !== null).length,
        0,
      ),
    [eventsBySport, sportsWithData],
  );
  const wonEvents = useMemo(
    () =>
      sportsWithData.reduce(
        (sum, sport) => sum + eventsBySport[sport.key].filter((event) => toHitValue(event.full_game_hit) === true).length,
        0,
      ),
    [eventsBySport, sportsWithData],
  );
  const hitRate = settledEvents > 0 ? (wonEvents / settledEvents) * 100 : 0;

  useEffect(() => {
    setExpandedSports((prev) => {
      const next = {};
      sportsWithData.forEach((sport) => {
        next[sport.key] = prev[sport.key] ?? false;
      });
      return next;
    });
  }, [sportsWithData]);

  function toggleSportSection(sportKey) {
    setExpandedSports((prev) => ({
      ...prev,
      [sportKey]: !prev[sportKey],
    }));
  }

  return (
    <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
      <div className={isDashboardPro ? (dashboardSidebarCompact ? "grid gap-6 xl:grid-cols-[78px_minmax(0,1fr)]" : "grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]") : "grid gap-6 xl:grid-cols-[290px_minmax(0,1fr)] 2xl:grid-cols-[310px_minmax(0,1fr)]"}>
        <SidebarCalendar
          calendarMonth={calendarMonth}
          setCalendarMonth={setCalendarMonth}
          selectedDate={selectedDate}
          loading={loading}
          error={error}
          onSelectDate={(dateStr) => loadSummaryByDate(dateStr)}
          onLoadToday={loadTodaySummary}
          availableDates={availableDates}
          title="Calendario resumen"
          subtitle="Revisa resultados y picks de cualquier fecha historica."
          todayButtonLabel="Cargar hoy"
          compactMode={dashboardSidebarCompact}
          onCompactChange={setDashboardSidebarCompact}
        />

        <section className="min-w-0">
          <section className="mb-6 rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.12),transparent_32%),linear-gradient(180deg,rgba(16,23,34,0.96),rgba(10,16,26,0.98))] p-5 shadow-[0_20px_56px_rgba(0,0,0,0.24)]">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-200/85">Resumen del dia</p>
                <h2 className="mt-2 text-2xl font-semibold text-white">Todos los eventos y estado de picks</h2>
                <p className="mt-1 text-sm text-white/60">Fecha activa: {selectedDate || "Sin fecha seleccionada"}.</p>
              </div>
              <button
                type="button"
                onClick={() => loadSummaryByDate(selectedDate || toYmdLocal(new Date()), { silent: true })}
                className="rounded-xl border border-emerald-300/35 bg-emerald-300/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-300/18"
              >
                {refreshing ? "Actualizando..." : "Actualizar"}
              </button>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-4">
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Eventos</p>
                <p className="mt-2 text-3xl font-semibold text-white">{totalEvents}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Terminados</p>
                <p className="mt-2 text-3xl font-semibold text-amber-200">{settledEvents}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Ganados</p>
                <p className="mt-2 text-3xl font-semibold text-emerald-300">{wonEvents}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">% de Acierto</p>
                <p className="mt-2 text-3xl font-semibold text-cyan-200">{hitRate.toFixed(1)}%</p>
                <p className="mt-1 text-xs text-white/50">{lastUpdated ? `Actualizado ${lastUpdated.toLocaleTimeString()}` : ""}</p>
              </div>
            </div>
          </section>

          {loading ? (
            <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
              Cargando resumen del dia...
            </div>
          ) : sportsWithData.length === 0 ? (
            <div className="rounded-[30px] border border-white/10 bg-white/[0.035] p-10 text-center text-white/70">
              No hay eventos disponibles para esta fecha.
            </div>
          ) : (
            <div className="space-y-6">
              {sportsWithData.map((sport) => {
                const sportEvents = eventsBySport[sport.key] || [];
                const sportWon = sportEvents.filter((event) => toHitValue(event.full_game_hit) === true).length;
                const sportLost = sportEvents.filter((event) => toHitValue(event.full_game_hit) === false).length;
                const isExpanded = expandedSports[sport.key] === true;

                return (
                  <section
                    key={sport.key}
                    className="overflow-hidden rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(14,20,30,0.96),rgba(10,15,24,0.98))] shadow-[0_16px_40px_rgba(0,0,0,0.22)]"
                  >
                    <div className="flex items-center justify-between gap-3 border-b border-white/10 bg-emerald-500/10 px-4 py-3">
                      <button
                        type="button"
                        onClick={() => toggleSportSection(sport.key)}
                        className="flex min-w-0 flex-1 items-center gap-3 text-left"
                      >
                        <span className={`inline-flex h-6 w-6 items-center justify-center rounded-full border border-emerald-300/35 bg-emerald-400/10 text-[11px] text-emerald-100 transition ${isExpanded ? "rotate-180" : ""}`}>
                          v
                        </span>
                        <h3 className="truncate text-sm font-semibold uppercase tracking-[0.14em] text-emerald-100">
                          {sport.label} - {sportEvents.length} eventos
                        </h3>
                        <div className="ml-auto flex items-center gap-2">
                          <span className="rounded-full border border-emerald-400/45 bg-emerald-500/12 px-2 py-1 text-[10px] font-semibold text-emerald-100">
                            G: {sportWon}
                          </span>
                          <span className="rounded-full border border-rose-400/45 bg-rose-500/12 px-2 py-1 text-[10px] font-semibold text-rose-100">
                            P: {sportLost}
                          </span>
                        </div>
                      </button>

                      <button
                        type="button"
                        onClick={() => navigate(`${sport.path}?date=${selectedDate}`)}
                        className="shrink-0 text-xs font-semibold text-emerald-100/85 underline underline-offset-4 transition hover:text-white"
                      >
                        Abrir board
                      </button>
                    </div>

                    {isExpanded && (
                      <div className="space-y-2 p-3">
                        {sportEvents.map((event) => {
                          const teams = resolveEventTeams(event);
                          const picks = resolveSummaryPicks(event, sport.key, teams);
                          const fullHit = toHitValue(event.full_game_hit);
                          const finalScore = String(event.final_score_text || "").trim();
                          const finalState = hasFinalResult(event);

                          return (
                            <div
                              key={`${sport.key}-${event.game_id || `${event.date}-${event.time}-${teams.homeTeam}`}`}
                              className={`cursor-pointer rounded-2xl border bg-white/[0.02] px-4 py-3 transition hover:bg-white/[0.045] ${rowBorderClass(event)}`}
                              onClick={() => setActiveEvent(event)}
                            >
                              <div className="grid gap-4 xl:grid-cols-[190px_minmax(0,1fr)_minmax(340px,430px)] xl:items-center">
                                <div>
                                  <p className="text-sm font-semibold text-white/90">{event.time || "Sin hora"}</p>
                                  <p className="mt-1 text-xs text-white/55">{event.date}</p>
                                  <p className={`mt-2 inline-flex rounded-full border px-2 py-1 text-[11px] font-semibold ${
                                    fullHit === true
                                      ? "border-emerald-400/45 bg-emerald-500/12 text-emerald-100"
                                      : fullHit === false
                                        ? "border-rose-400/45 bg-rose-500/12 text-rose-100"
                                        : "border-white/15 bg-white/[0.045] text-white/75"
                                  }`}>
                                    {fullHit === true ? "FG ACIERTO" : fullHit === false ? "FG FALLO" : "FG PENDIENTE"}
                                  </p>
                                </div>

                                <div className="space-y-2">
                                  <TeamCell sportKey={sport.key} abbr={teams.awayTeam} side="away" />
                                  <TeamCell sportKey={sport.key} abbr={teams.homeTeam} side="home" />
                                  {finalState && finalScore && (
                                    <p className="text-xs font-semibold text-white/80">Final: {finalScore}</p>
                                  )}
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
                                  {picks.length === 0 ? (
                                    <span className="rounded-xl border border-white/15 bg-white/[0.04] px-3 py-1.5 text-xs text-white/70">
                                      Sin picks detectados
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
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </section>
      </div>

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
