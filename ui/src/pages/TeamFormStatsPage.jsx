import { useEffect, useMemo, useState } from "react";
import { fetchAvailableDates, fetchPredictionsByDate } from "../services/api.js";
import { resolveEventTeams } from "../utils/teams.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

const SPORTS = [
  { key: "nba", label: "NBA" },
  { key: "mlb", label: "MLB" },
  { key: "triple_a", label: "Triple-A" },
  { key: "tennis", label: "Tennis" },
  { key: "kbo", label: "KBO" },
  { key: "nhl", label: "NHL" },
  { key: "ncaa_baseball", label: "NCAA Baseball" },
  { key: "euroleague", label: "EuroLeague" },
  { key: "liga_mx", label: "Liga MX" },
  { key: "laliga", label: "LaLiga" },
];

const MAX_DATES_PER_SPORT = 18;
const RECENT_GAMES = 8;

function parseNumeric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseScores(event) {
  const home = parseNumeric(event?.home_score);
  const away = parseNumeric(event?.away_score);
  if (home !== null && away !== null) return { home, away };

  const text = String(event?.final_score_text || "");
  const numbers = [...text.matchAll(/(\d+)/g)].map((m) => Number(m[1]));
  if (numbers.length >= 2) {
    const parsedAway = numbers[numbers.length - 2];
    const parsedHome = numbers[numbers.length - 1];
    if (Number.isFinite(parsedHome) && Number.isFinite(parsedAway)) {
      return { home: parsedHome, away: parsedAway };
    }
  }
  return null;
}

function hasFinalResult(event) {
  const state = String(event?.status_state || "").trim().toLowerCase();
  return (
    event?.result_available === true ||
    ["post", "final", "completed"].includes(state) ||
    Boolean(String(event?.final_score_text || "").trim())
  );
}

function classifyForm(scorePct) {
  if (scorePct < 30) return { label: "Mala racha", tone: "text-rose-200 border-rose-400/40 bg-rose-500/12" };
  if (scorePct < 45) return { label: "Racha media", tone: "text-orange-200 border-orange-400/40 bg-orange-500/12" };
  if (scorePct < 60) return { label: "Racha normal", tone: "text-white/90 border-white/20 bg-white/[0.06]" };
  if (scorePct < 75) return { label: "Buena racha", tone: "text-cyan-100 border-cyan-400/40 bg-cyan-500/12" };
  return { label: "Muy buena racha", tone: "text-emerald-100 border-emerald-400/45 bg-emerald-500/12" };
}

function streakLabel(outcomes) {
  if (!outcomes.length) return "Sin datos";
  const first = outcomes[0];
  if (!first) return "Sin datos";
  let count = 0;
  for (const outcome of outcomes) {
    if (outcome === first) count += 1;
    else break;
  }
  if (first === "W") return `W${count}`;
  if (first === "L") return `L${count}`;
  return `D${count}`;
}

function outcomeFromScores(forGoals, againstGoals) {
  if (forGoals > againstGoals) return "W";
  if (forGoals < againstGoals) return "L";
  return "D";
}

function buildTeamFormRows(allEvents) {
  const byTeam = new Map();

  for (const event of allEvents) {
    if (!hasFinalResult(event)) continue;
    const scores = parseScores(event);
    if (!scores) continue;

    const { awayTeam, homeTeam } = resolveEventTeams(event);
    const timePart = String(event.time || "00:00");
    const stamp = new Date(`${event.date}T${timePart.length === 5 ? timePart : "00:00"}:00`).getTime();
    const ts = Number.isFinite(stamp) ? stamp : Date.now();

    const homeOutcome = outcomeFromScores(scores.home, scores.away);
    const awayOutcome = outcomeFromScores(scores.away, scores.home);

    const homeKey = `${event.sportKey}::${homeTeam}`;
    const awayKey = `${event.sportKey}::${awayTeam}`;

    if (!byTeam.has(homeKey)) byTeam.set(homeKey, []);
    if (!byTeam.has(awayKey)) byTeam.set(awayKey, []);

    byTeam.get(homeKey).push({
      sportKey: event.sportKey,
      teamCode: homeTeam,
      ts,
      date: event.date,
      outcome: homeOutcome,
      goalsFor: scores.home,
      goalsAgainst: scores.away,
    });
    byTeam.get(awayKey).push({
      sportKey: event.sportKey,
      teamCode: awayTeam,
      ts,
      date: event.date,
      outcome: awayOutcome,
      goalsFor: scores.away,
      goalsAgainst: scores.home,
    });
  }

  const rows = [];

  for (const records of byTeam.values()) {
    const sorted = [...records].sort((a, b) => b.ts - a.ts);
    const recent = sorted.slice(0, RECENT_GAMES);
    if (recent.length < 3) continue;

    let wins = 0;
    let losses = 0;
    let draws = 0;
    let gf = 0;
    let ga = 0;
    let points = 0;

    for (const r of recent) {
      gf += r.goalsFor;
      ga += r.goalsAgainst;
      if (r.outcome === "W") {
        wins += 1;
        points += 3;
      } else if (r.outcome === "L") {
        losses += 1;
      } else {
        draws += 1;
        points += 1;
      }
    }

    const maxPoints = recent.length * 3;
    const formScorePct = maxPoints > 0 ? (points / maxPoints) * 100 : 0;
    const formState = classifyForm(formScorePct);
    const latest = recent[0];

    rows.push({
      sportKey: latest.sportKey,
      sportLabel: SPORTS.find((s) => s.key === latest.sportKey)?.label || latest.sportKey,
      teamCode: latest.teamCode,
      teamName: getTeamDisplayName(latest.sportKey, latest.teamCode),
      gamesSample: recent.length,
      formScorePct,
      wins,
      losses,
      draws,
      gf,
      ga,
      diff: gf - ga,
      streak: streakLabel(recent.map((x) => x.outcome)),
      formLabel: formState.label,
      formTone: formState.tone,
      lastMatchDate: latest.date,
    });
  }

  return rows.sort((a, b) => b.formScorePct - a.formScorePct);
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [search, setSearch] = useState("");
  const [sportFilter, setSportFilter] = useState("all");

  useEffect(() => {
    let cancelled = false;

    async function loadTeamStats() {
      try {
        setLoading(true);
        setError("");

        const dateResponses = await Promise.allSettled(
          SPORTS.map((sport) => fetchAvailableDates(sport.key)),
        );

        const fetchJobs = [];
        for (let i = 0; i < SPORTS.length; i += 1) {
          const sport = SPORTS[i];
          const dateRes = dateResponses[i];
          if (dateRes.status !== "fulfilled" || !Array.isArray(dateRes.value) || dateRes.value.length === 0) continue;

          const dates = [...dateRes.value].sort().slice(-MAX_DATES_PER_SPORT);
          dates.forEach((dateStr) => {
            fetchJobs.push({ sportKey: sport.key, dateStr });
          });
        }

        const eventResponses = await Promise.allSettled(
          fetchJobs.map((job) => fetchPredictionsByDate(job.sportKey, job.dateStr)),
        );

        const events = [];
        for (let j = 0; j < fetchJobs.length; j += 1) {
          const job = fetchJobs[j];
          const res = eventResponses[j];
          if (res.status !== "fulfilled" || !Array.isArray(res.value)) continue;
          for (const event of res.value) {
            events.push({ ...event, sportKey: job.sportKey });
          }
        }

        const computedRows = buildTeamFormRows(events);
        if (cancelled) return;
        setRows(computedRows);
        setLastUpdated(new Date());
      } catch (err) {
        if (cancelled) return;
        setRows([]);
        setError(err?.message || "No se pudieron calcular las rachas de equipos.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadTeamStats();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredRows = useMemo(() => {
    const text = String(search || "").trim().toLowerCase();
    return rows.filter((row) => {
      if (sportFilter !== "all" && row.sportKey !== sportFilter) return false;
      if (!text) return true;
      return `${row.teamName} ${row.teamCode} ${row.sportLabel}`.toLowerCase().includes(text);
    });
  }, [rows, search, sportFilter]);

  const topRows = filteredRows.slice(0, 5);
  const lowRows = [...filteredRows].sort((a, b) => a.formScorePct - b.formScorePct).slice(0, 5);

  return (
    <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
      <section className="mb-6 rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.10),transparent_28%),linear-gradient(180deg,rgba(15,19,27,0.97),rgba(10,13,20,0.98))] p-6 shadow-[0_20px_56px_rgba(0,0,0,0.24)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-200/85">Estadisticas IA</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Racha actual por equipo</h2>
            <p className="mt-1 text-sm text-white/60">
              La IA clasifica la forma reciente en: mala, media, normal, buena o muy buena.
            </p>
            {lastUpdated && (
              <p className="mt-2 text-xs text-white/45">Actualizado: {lastUpdated.toLocaleString()}</p>
            )}
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Equipos analizados</p>
            <p className="mt-2 text-3xl font-semibold text-white">{rows.length}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Filtro activo</p>
            <p className="mt-2 text-lg font-semibold text-amber-200">
              {sportFilter === "all" ? "Todos los deportes" : SPORTS.find((s) => s.key === sportFilter)?.label}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mejor score</p>
            <p className="mt-2 text-3xl font-semibold text-emerald-300">
              {filteredRows[0] ? `${filteredRows[0].formScorePct.toFixed(1)}%` : "-"}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Muestra reciente</p>
            <p className="mt-2 text-lg font-semibold text-cyan-200">Ultimos {RECENT_GAMES} juegos</p>
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
                    <p className="text-sm font-semibold text-white">{row.teamName} <span className="text-white/50">({row.sportLabel})</span></p>
                    <p className="text-sm font-semibold text-emerald-200">{row.formScorePct.toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-black/20 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-white/60">Top baja forma</h3>
              <div className="mt-3 space-y-2">
                {lowRows.map((row) => (
                  <div key={`low-${row.sportKey}-${row.teamCode}`} className="flex items-center justify-between rounded-xl border border-rose-400/25 bg-rose-500/8 px-3 py-2">
                    <p className="text-sm font-semibold text-white">{row.teamName} <span className="text-white/50">({row.sportLabel})</span></p>
                    <p className="text-sm font-semibold text-rose-200">{row.formScorePct.toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {filteredRows.map((row) => (
              <article
                key={`${row.sportKey}-${row.teamCode}`}
                className={`rounded-[24px] border bg-[linear-gradient(180deg,rgba(20,24,34,0.96),rgba(13,17,25,0.98))] p-4 shadow-[0_14px_34px_rgba(0,0,0,0.2)] ${borderByScore(row.formScorePct)}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-base font-semibold text-white">{row.teamName}</p>
                    <p className="mt-0.5 text-xs text-white/55">{row.sportLabel}</p>
                  </div>
                  <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold ${row.formTone}`}>
                    {row.formLabel}
                  </span>
                </div>

                <div className="mt-4 flex items-end justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Form Score</p>
                    <p className="mt-1 text-3xl font-semibold text-white">{row.formScorePct.toFixed(1)}%</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Racha actual</p>
                    <p className="mt-1 text-lg font-semibold text-cyan-100">{row.streak}</p>
                  </div>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-rose-400 via-amber-300 to-emerald-400"
                    style={{ width: `${Math.max(0, Math.min(100, row.formScorePct))}%` }}
                  />
                </div>

                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/85">
                    W-D-L: {row.wins}-{row.draws}-{row.losses}
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-white/85">
                    DG: {row.diff >= 0 ? `+${row.diff}` : row.diff}
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
    </main>
  );
}
