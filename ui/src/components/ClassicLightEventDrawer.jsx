import { useEffect, useState } from "react";
import { getTeamLogoUrl } from "../utils/teamLogos.js";
import { resolveEventTeams, resolveSidePick } from "../utils/teams.js";
import { expandTeamCodeInText, getTeamDisplayName } from "../utils/teamNames.js";
import { fetchAvailableDates, fetchPredictionsByDate } from "../services/api.js";

const ODDS_KEYS = {
  moneyline: [
    "closing_moneyline_odds",
    "closing_ml_odds",
    "closing_odds_ml",
    "moneyline_odds",
    "ml_odds",
    "odds_ml",
    "odds_moneyline",
    "home_moneyline_odds",
    "away_moneyline_odds",
  ],
  q1: ["closing_q1_odds", "q1_odds"],
  total: ["closing_total_odds", "total_odds", "odds_total"],
  spread: ["closing_spread_odds", "spread_odds", "odds_spread"],
};

function toHitValue(value) {
  if (value === true || value === false) return value;
  if (value === 1 || value === "1") return true;
  if (value === 0 || value === "0") return false;
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (["true", "yes", "si", "acierto", "win", "won"].includes(v)) return true;
    if (["false", "no", "fallo", "lose", "lost"].includes(v)) return false;
  }
  return null;
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
  if (!Number.isFinite(n) || n === 0) return null;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(1);
}

function formatDecimalOdds(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return null;
  if (n > 1 && n < 20) return n.toFixed(2);
  if (n > 0) return (1 + n / 100).toFixed(2);
  if (n <= -100) return (1 + 100 / Math.abs(n)).toFixed(2);
  return null;
}

function resolveOddsValue(event, keys) {
  for (const key of keys) {
    const formatted = formatDecimalOdds(event?.[key]);
    if (formatted) return formatted;
  }
  return null;
}

function resolveResultState(event) {
  const state = String(event?.status_state || "").trim().toLowerCase();
  const statusDescription = String(event?.status_description || "").trim().toLowerCase();
  const statusDetail = String(event?.status_detail || "").trim().toLowerCase();
  const statusText = `${statusDescription} ${statusDetail}`.trim();
  const completed = Number(event?.status_completed) === 1;
  const finalScoreText = String(event?.final_score_text || "").trim();
  const liveStates = ["in", "live", "in_progress", "inprogress", "status_in_progress", "halftime", "mid"];
  const nonFinalStates = [...liveStates, "pre", "scheduled", "not_started", "ns"];
  const hasLiveHint = /\ben vivo\b|\bin progress\b|\blive\b|\bq[1-4]\b|\bquarter\b|\bhalftime\b|\bhalf\b|\bot\b|\bperiod\b|\b[1-4](?:st|nd|rd|th)\b|\btop\s*\d+\b|\bbot\s*\d+\b/.test(statusText);
  const hasFinalState = event?.result_available === true || completed || ["post", "final", "completed"].includes(state);
  const hasFinalScoreFallback = Boolean(finalScoreText) && !nonFinalStates.includes(state) && !hasLiveHint;
  const hasResult = hasFinalState || hasFinalScoreFallback;
  const isLive = !hasResult && (liveStates.includes(state) || hasLiveHint);
  return { hasResult, isLive };
}

function normalizeTeamToken(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9]/g, "")
    .toUpperCase()
    .trim();
}

function isSameMatchup(baseEvent, candidateEvent) {
  const baseTeams = resolveEventTeams(baseEvent);
  const candidateTeams = resolveEventTeams(candidateEvent);

  const baseAway = normalizeTeamToken(baseTeams.awayTeam);
  const baseHome = normalizeTeamToken(baseTeams.homeTeam);
  const candidateAway = normalizeTeamToken(candidateTeams.awayTeam);
  const candidateHome = normalizeTeamToken(candidateTeams.homeTeam);

  if (!baseAway || !baseHome || !candidateAway || !candidateHome) return false;
  return (
    (baseAway === candidateAway && baseHome === candidateHome)
    || (baseAway === candidateHome && baseHome === candidateAway)
  );
}

function parseScoreArray(value) {
  if (Array.isArray(value)) return value.map((item) => toFiniteNumber(item));
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) return parsed.map((item) => toFiniteNumber(item));
    } catch {
      return [];
    }
  }
  return [];
}

function trimTrailingNullScores(scores) {
  const out = [...scores];
  while (out.length && out[out.length - 1] === null) out.pop();
  return out;
}

function buildInningScoresFromColumns(event, prefix) {
  const scores = [];
  for (let inning = 1; inning <= 20; inning += 1) {
    const key = `${prefix}_r${inning}`;
    const hasKey = Object.prototype.hasOwnProperty.call(event || {}, key);
    const value = toFiniteNumber(event?.[key]);
    if (hasKey || value !== null) scores.push(value);
  }
  return trimTrailingNullScores(scores);
}

function resolveSegmentRowLabel(sportKey, segmentIndex) {
  const sport = String(sportKey || "").toLowerCase();
  if (["mlb", "lmb", "kbo", "triple_a", "ncaa_baseball"].includes(sport)) return `${segmentIndex}`;
  if (["nba", "wnba", "euroleague"].includes(sport)) {
    if (segmentIndex <= 4) return `Q${segmentIndex}`;
    return `OT${segmentIndex - 4}`;
  }
  if (["nhl"].includes(sport)) {
    if (segmentIndex <= 3) return `P${segmentIndex}`;
    return `OT${segmentIndex - 3}`;
  }
  if (["liga_mx", "laliga", "bundesliga", "ligue1"].includes(sport)) {
    if (segmentIndex === 1) return "1T";
    if (segmentIndex === 2) return "2T";
    return `ET${segmentIndex - 2}`;
  }
  return `${segmentIndex}`;
}

function buildSegmentRows(event, sportKey) {
  const sport = String(sportKey || "").toLowerCase();

  if (["mlb", "lmb", "kbo", "triple_a", "ncaa_baseball"].includes(sport)) {
    const homeFromArray = trimTrailingNullScores(parseScoreArray(event?.home_inning_scores));
    const awayFromArray = trimTrailingNullScores(parseScoreArray(event?.away_inning_scores));
    const homeFromColumns = buildInningScoresFromColumns(event, "home");
    const awayFromColumns = buildInningScoresFromColumns(event, "away");
    const homeRaw = homeFromArray.length ? homeFromArray : homeFromColumns;
    const awayRaw = awayFromArray.length ? awayFromArray : awayFromColumns;
    const count = Math.max(homeRaw.length, awayRaw.length);
    return Array.from({ length: count }, (_, index) => ({
      label: resolveSegmentRowLabel(sportKey, index + 1),
      home: index < homeRaw.length ? homeRaw[index] : null,
      away: index < awayRaw.length ? awayRaw[index] : null,
    })).filter((row) => row.home !== null || row.away !== null);
  }

  const homeArray = trimTrailingNullScores(parseScoreArray(event?.home_segment_scores));
  const awayArray = trimTrailingNullScores(parseScoreArray(event?.away_segment_scores));
  const fallbackHome = trimTrailingNullScores([
    toFiniteNumber(event?.home_q1_score ?? event?.home_q1),
    toFiniteNumber(event?.home_q2_score ?? event?.home_q2),
    toFiniteNumber(event?.home_q3_score ?? event?.home_q3),
    toFiniteNumber(event?.home_q4_score ?? event?.home_q4),
  ]);
  const fallbackAway = trimTrailingNullScores([
    toFiniteNumber(event?.away_q1_score ?? event?.away_q1),
    toFiniteNumber(event?.away_q2_score ?? event?.away_q2),
    toFiniteNumber(event?.away_q3_score ?? event?.away_q3),
    toFiniteNumber(event?.away_q4_score ?? event?.away_q4),
  ]);

  const homeRaw = homeArray.length ? homeArray : fallbackHome;
  const awayRaw = awayArray.length ? awayArray : fallbackAway;
  const count = Math.max(homeRaw.length, awayRaw.length);

  return Array.from({ length: count }, (_, index) => ({
    label: resolveSegmentRowLabel(sportKey, index + 1),
    home: index < homeRaw.length ? homeRaw[index] : null,
    away: index < awayRaw.length ? awayRaw[index] : null,
  })).filter((row) => row.home !== null || row.away !== null);
}

function marketBoxBorder(hit) {
  if (hit === true) return "border-[2.6px] border-[#64a981]";
  if (hit === false) return "border-[2.6px] border-[#cf727f]";
  return "border-[1.5px] border-[#9fa2ab]";
}

function confidenceLabel(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${Math.round(n)}%`;
}

function toStatusText(event, isLive, hasResult) {
  if (isLive) {
    const detail = String(event?.status_detail || "").trim();
    const shortState = String(event?.status_description || "").trim();
    return detail || shortState || "En vivo";
  }
  if (hasResult) return "Final";
  return "Programado";
}

function TeamLogo({ sportKey, teamCode, teamName }) {
  const logoUrl = getTeamLogoUrl(sportKey, teamCode);
  return (
    <img
      src={logoUrl || "/logos/default-team.svg"}
      alt={`Logo ${teamName}`}
      loading="lazy"
      onError={(e) => {
        e.currentTarget.onerror = null;
        e.currentTarget.src = "/logos/default-team.svg";
      }}
      className="h-16 w-16 object-contain"
    />
  );
}

function MarketBox({ title, pick, confidence, odds, action, hit }) {
  return (
    <article className={`rounded-[16px] border bg-[#d9d9de] px-3 py-3 shadow-[0_4px_10px_rgba(0,0,0,0.08)] ${marketBoxBorder(hit)}`}>
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#545865]">{title}</p>
      <p className="mt-2 text-[21px] font-black uppercase leading-tight text-[#252831] [font-family:var(--classic-display-font)]">{pick || "N/A"}</p>
      <div className="mt-2 space-y-1 text-[11px] text-[#4f5361]">
        <p>Confianza: {confidenceLabel(confidence)}</p>
        <p>Cuota: {odds || "N/A"}</p>
        <p>Accion: {action || "N/A"}</p>
      </div>
    </article>
  );
}

function buildEventMarkets(event, sportKey, teams) {
  const fullGamePick = expandTeamCodeInText(sportKey, resolveSidePick(event?.full_game_pick, teams)) || "N/A";
  const fullGameHit = toHitValue(event?.full_game_hit);
  const fullGameOdds = resolveOddsValue(event, ODDS_KEYS.moneyline);

  const q1PickRaw = String(event?.q1_pick || "").trim();
  const q1Pick = !isPendingPick(q1PickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(q1PickRaw, teams)) || q1PickRaw)
    : "N/A";
  const q1Hit = toHitValue(event?.q1_hit);
  const q1Odds = resolveOddsValue(event, ODDS_KEYS.q1);

  const h1PickRaw = String(event?.h1_pick || "").trim();
  const h1Pick = !isPendingPick(h1PickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(h1PickRaw, teams)) || h1PickRaw)
    : "N/A";
  const h1Hit = toHitValue(event?.h1_hit);

  const spreadPickRaw = String(event?.spread_pick || "").trim();
  const spreadPickText = !isPendingPick(spreadPickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)) || spreadPickRaw)
    : fullGamePick;
  const spreadLine =
    String(event?.spread_line_display || "").trim()
    || toLine(event?.spread_line_signed)
    || toLine(event?.home_spread)
    || toLine(event?.closing_spread_line)
    || "";
  const handicapPick = `${spreadPickText}${spreadLine ? ` ${spreadLine}` : ""}`.trim() || "N/A";
  const spreadHit = toHitValue(event?.correct_spread);
  const spreadOdds = resolveOddsValue(event, ODDS_KEYS.spread);

  const totalRaw = event?.total_recommended_pick || event?.total_pick;
  const totalDirection = normalizeTotalDirection(totalRaw);
  const totalLine = formatLineValue(event?.closing_total_line ?? event?.odds_over_under);
  const embeddedLine = formatLineValue(extractNumericFromText(totalRaw));
  const totalPick =
    totalDirection && (embeddedLine || totalLine)
      ? `${totalDirection} ${embeddedLine || totalLine}`
      : String(totalRaw || "N/A");
  const totalHit = toHitValue(event?.correct_total_adjusted ?? event?.correct_total);
  const totalOdds = resolveOddsValue(event, ODDS_KEYS.total);

  return [
    {
      label: "Ganador del partido",
      pick: fullGamePick,
      confidence: event?.full_game_confidence,
      hit: fullGameHit,
      odds: fullGameOdds,
      action: event?.full_game_action || "N/A",
    },
    {
      label: "1er parcial",
      pick: q1Pick,
      confidence: event?.q1_confidence,
      hit: q1Hit,
      odds: q1Odds,
      action: event?.q1_action || "N/A",
    },
    {
      label: "Primera mitad",
      pick: h1Pick,
      confidence: event?.h1_confidence,
      hit: h1Hit,
      odds: "N/A",
      action: event?.h1_action || "N/A",
    },
    {
      label: "Handicap",
      pick: handicapPick,
      confidence: event?.spread_confidence,
      hit: spreadHit,
      odds: spreadOdds,
      action: event?.spread_market || "N/A",
    },
    {
      label: "Over / Under",
      pick: totalPick,
      confidence: event?.total_confidence,
      hit: totalHit,
      odds: totalOdds,
      action: event?.total_action || "N/A",
    },
  ].filter((item) => item.pick && item.pick !== "N/A");
}

function resolveRecommendedMarket(candidates) {
  return [...candidates]
    .filter((item) => Number.isFinite(Number(item.confidence)))
    .sort((a, b) => Number(b.confidence) - Number(a.confidence))[0] || candidates[0] || {
      label: "Pick recomendado",
      pick: "N/A",
      confidence: null,
      hit: null,
      odds: null,
      action: "N/A",
    };
}

function HistoricalPredictionBlock({ event, sportKey }) {
  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const { hasResult, isLive } = resolveResultState(event);
  const segmentRows = buildSegmentRows(event, sportKey);
  const marketCards = buildEventMarkets(event, sportKey, teams);
  const finalScoreText = String(event?.final_score_text || "").trim();
  const awayScore = toFiniteNumber(event?.away_score);
  const homeScore = toFiniteNumber(event?.home_score);

  return (
    <section className="rounded-[20px] border border-[#a7aab2] bg-[#d5d6db] p-3">
      <div className="mb-2 flex items-center justify-between text-[11px] text-[#3f4350]">
        <p className="font-semibold uppercase tracking-[0.12em]">Prediccion IA</p>
        <p className="font-semibold">{String(event?.date || "").trim()}</p>
      </div>

      <div className="rounded-[16px] border border-[#a7aab2] bg-[#dedee2] p-3">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
          <div className="flex flex-col items-center gap-2 text-center">
            <TeamLogo sportKey={sportKey} teamCode={teams.awayTeam} teamName={awayName} />
            <p className="text-[11px] font-black uppercase tracking-[0.02em] text-[#252831] [font-family:var(--classic-display-font)]">{awayName}</p>
          </div>

          <div className="text-center">
            <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-[#535865]">vs</p>
          </div>

          <div className="flex flex-col items-center gap-2 text-center">
            <TeamLogo sportKey={sportKey} teamCode={teams.homeTeam} teamName={homeName} />
            <p className="text-[11px] font-black uppercase tracking-[0.02em] text-[#252831] [font-family:var(--classic-display-font)]">{homeName}</p>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2">
          <div className="rounded-[12px] border border-[#b0b3bb] bg-[#efeff2] p-2 text-center">
            <p className="text-[11px] uppercase text-[#636775]">{awayName}</p>
            <p className="text-4xl font-black leading-none text-[#252831] [font-family:var(--classic-display-font)]">{awayScore ?? "-"}</p>
          </div>
          <div className="rounded-[12px] border border-[#b0b3bb] bg-[#efeff2] p-2 text-center">
            <p className="text-[11px] uppercase text-[#636775]">{homeName}</p>
            <p className="text-4xl font-black leading-none text-[#252831] [font-family:var(--classic-display-font)]">{homeScore ?? "-"}</p>
          </div>
        </div>

        {segmentRows.length > 0 && (
          <div className="mt-3 overflow-x-auto rounded-[12px] border border-[#aeb1b8]">
            <table className="min-w-full text-center text-[11px] text-[#3f4350]">
              <thead className="bg-[#ececf0] text-[10px] uppercase tracking-[0.1em] text-[#616574]">
                <tr>
                  <th className="px-2 py-1 text-left">Parcial</th>
                  <th className="px-2 py-1">{awayName}</th>
                  <th className="px-2 py-1">{homeName}</th>
                </tr>
              </thead>
              <tbody>
                {segmentRows.map((row) => (
                  <tr key={row.label} className="border-t border-[#c4c6cd]">
                    <td className="px-2 py-1 text-left font-semibold">{row.label}</td>
                    <td className="px-2 py-1">{row.away ?? "-"}</td>
                    <td className="px-2 py-1">{row.home ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-2 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[#5a5f6c]">{toStatusText(event, isLive, hasResult)}</p>
          {finalScoreText && <p className="text-xs text-[#40444f]">{finalScoreText}</p>}
        </div>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {marketCards.map((market) => (
          <MarketBox
            key={`${event?.game_id || "hist"}-${market.label}`}
            title={market.label}
            pick={market.pick}
            confidence={market.confidence}
            odds={market.odds}
            action={market.action}
            hit={market.hit}
          />
        ))}
      </div>
    </section>
  );
}

export default function ClassicLightEventDrawer({ event, sportKey, onClose }) {
  const [previousOpen, setPreviousOpen] = useState(false);
  const [previousLoading, setPreviousLoading] = useState(false);
  const [previousError, setPreviousError] = useState("");
  const [previousGames, setPreviousGames] = useState([]);

  if (!event) return null;

  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const { hasResult, isLive } = resolveResultState(event);
  const segmentRows = buildSegmentRows(event, sportKey);
  const marketCards = buildEventMarkets(event, sportKey, teams);

  const recommended = resolveRecommendedMarket(marketCards);

  const finalScoreText = String(event?.final_score_text || "").trim();
  const awayScore = toFiniteNumber(event?.away_score);
  const homeScore = toFiniteNumber(event?.home_score);

  useEffect(() => {
    setPreviousOpen(false);
    setPreviousLoading(false);
    setPreviousError("");
    setPreviousGames([]);
  }, [event?.game_id, event?.date, sportKey]);

  async function handleComparePrevious() {
    const nextState = !previousOpen;
    setPreviousOpen(nextState);

    if (!nextState || previousLoading || previousGames.length > 0) return;

    setPreviousLoading(true);
    setPreviousError("");
    try {
      const currentDate = String(event?.date || "").trim();
      if (!currentDate) {
        setPreviousError("Este evento no tiene fecha valida para comparar.");
        setPreviousGames([]);
        return;
      }

      const allDates = await fetchAvailableDates(sportKey);
      const previousDates = (Array.isArray(allDates) ? allDates : [])
        .filter((d) => d < currentDate)
        .sort()
        .reverse()
        .slice(0, 60);

      const matches = [];
      const seen = new Set();
      for (const dateStr of previousDates) {
        if (matches.length >= 3) break;
        let dayEvents = [];
        try {
          dayEvents = await fetchPredictionsByDate(sportKey, dateStr);
        } catch {
          continue;
        }
        for (const item of dayEvents) {
          if (matches.length >= 3) break;
          if (!isSameMatchup(event, item)) continue;
          const teamsItem = resolveEventTeams(item);
          const dedupeKey = [
            item?.date || dateStr,
            item?.game_id || "",
            normalizeTeamToken(teamsItem.awayTeam),
            normalizeTeamToken(teamsItem.homeTeam),
          ].join("|");
          if (seen.has(dedupeKey)) continue;
          seen.add(dedupeKey);
          matches.push(item);
        }
      }

      setPreviousGames(matches);
      if (matches.length === 0) {
        setPreviousError("No encontramos partidos anteriores de este mismo enfrentamiento.");
      }
    } catch {
      setPreviousError("No se pudieron cargar las predicciones anteriores.");
      setPreviousGames([]);
    } finally {
      setPreviousLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        onClick={onClose}
        className="absolute inset-0 bg-black/16"
        aria-label="Cerrar panel"
      />

      <aside className="classic-light-surface classic-light-event-drawer absolute right-0 top-0 flex h-full w-full max-w-[min(1180px,96vw)] flex-col border-l border-[#a5a8b0] bg-[#c6c7cc] p-4 shadow-[-18px_0_36px_rgba(0,0,0,0.2)]">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#4d5160]">Detalle del evento</p>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-[#8f929c] px-2.5 py-1 text-xs font-semibold text-[#3d404b] transition hover:bg-[#ececf0]"
          >
            Cerrar
          </button>
        </div>

        <div className="mt-3 grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(350px,430px)_minmax(0,1fr)]">
          <div className="min-h-0 overflow-y-auto pr-1">
            <div className="rounded-[20px] border border-[#a7aab2] bg-[#dedee2] p-3">
              <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                <div className="flex flex-col items-center gap-2 text-center">
                  <TeamLogo sportKey={sportKey} teamCode={teams.awayTeam} teamName={awayName} />
                  <p className="text-[11px] font-black uppercase tracking-[0.02em] text-[#252831] [font-family:var(--classic-display-font)]">{awayName}</p>
                </div>

                <div className="text-center">
                  <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-[#535865]">vs</p>
                </div>

                <div className="flex flex-col items-center gap-2 text-center">
                  <TeamLogo sportKey={sportKey} teamCode={teams.homeTeam} teamName={homeName} />
                  <p className="text-[11px] font-black uppercase tracking-[0.02em] text-[#252831] [font-family:var(--classic-display-font)]">{homeName}</p>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="rounded-[12px] border border-[#b0b3bb] bg-[#efeff2] p-2 text-center">
                  <p className="text-[11px] uppercase text-[#636775]">{awayName}</p>
                  <p className="text-4xl font-black leading-none text-[#252831] [font-family:var(--classic-display-font)]">{awayScore ?? "-"}</p>
                </div>
                <div className="rounded-[12px] border border-[#b0b3bb] bg-[#efeff2] p-2 text-center">
                  <p className="text-[11px] uppercase text-[#636775]">{homeName}</p>
                  <p className="text-4xl font-black leading-none text-[#252831] [font-family:var(--classic-display-font)]">{homeScore ?? "-"}</p>
                </div>
              </div>

              {segmentRows.length > 0 && (
                <div className="mt-3 overflow-x-auto rounded-[12px] border border-[#aeb1b8]">
                  <table className="min-w-full text-center text-[11px] text-[#3f4350]">
                    <thead className="bg-[#ececf0] text-[10px] uppercase tracking-[0.1em] text-[#616574]">
                      <tr>
                        <th className="px-2 py-1 text-left">Parcial</th>
                        <th className="px-2 py-1">{awayName}</th>
                        <th className="px-2 py-1">{homeName}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {segmentRows.map((row) => (
                        <tr key={row.label} className="border-t border-[#c4c6cd]">
                          <td className="px-2 py-1 text-left font-semibold">{row.label}</td>
                          <td className="px-2 py-1">{row.away ?? "-"}</td>
                          <td className="px-2 py-1">{row.home ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="mt-2 flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[#5a5f6c]">{toStatusText(event, isLive, hasResult)}</p>
                {finalScoreText && <p className="text-xs text-[#40444f]">{finalScoreText}</p>}
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {marketCards.map((market) => (
                <MarketBox
                  key={market.label}
                  title={market.label}
                  pick={market.pick}
                  confidence={market.confidence}
                  odds={market.odds}
                  action={market.action}
                  hit={market.hit}
                />
              ))}
              <MarketBox
                title="Pick recomendado"
                pick={recommended.pick}
                confidence={recommended.confidence}
                odds={recommended.odds}
                action={recommended.label}
                hit={recommended.hit}
              />
            </div>

            <div className="mt-4">
              <button
                type="button"
                onClick={handleComparePrevious}
                className="w-full rounded-[12px] border border-[#8f929c] bg-[#ececf0] px-3 py-2 text-xs font-black uppercase tracking-[0.08em] text-[#2f3340] transition hover:bg-[#f3f3f6]"
              >
                {previousOpen ? "OCULTAR PREDICCIONES ANTERIORES(3DIAS)" : "COMPARAR PREDICCIONES ANTERIORES(3DIAS)"}
              </button>
            </div>
          </div>

          <section className="min-h-0 overflow-y-auto rounded-[20px] border border-[#a7aab2] bg-[#d0d1d6] p-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#4d5160]">
              Predicciones anteriores ({previousGames.length})
            </p>

            {!previousOpen && (
              <div className="mt-3 rounded-[14px] border border-[#adb0b8] bg-[#ebebef] px-3 py-4 text-center text-xs text-[#4c5160]">
                Activa el boton de comparar para ver los 3 juegos anteriores aqui, en paralelo.
              </div>
            )}

            {previousOpen && (
              <div className="mt-3 space-y-3">
                {previousLoading && (
                  <p className="rounded-[12px] border border-[#b0b3bb] bg-[#efeff2] px-3 py-2 text-xs text-[#454a58]">
                    Cargando comparativa...
                  </p>
                )}
                {!previousLoading && previousError && (
                  <p className="rounded-[12px] border border-[#cf727f] bg-[#f8e8eb] px-3 py-2 text-xs text-[#8b2434]">
                    {previousError}
                  </p>
                )}
                {!previousLoading && !previousError && previousGames.map((pastEvent, index) => (
                  <HistoricalPredictionBlock
                    key={`${pastEvent?.date || "date"}-${pastEvent?.game_id || index}`}
                    event={pastEvent}
                    sportKey={sportKey}
                  />
                ))}
              </div>
            )}
          </section>
        </div>
      </aside>
    </div>
  );
}
