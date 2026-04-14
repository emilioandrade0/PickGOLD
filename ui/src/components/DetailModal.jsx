import { resolveEventTier, resolveMarketTier, tierClasses, tierLabel } from "../utils/picks.js";
import { getTeamLogoUrl } from "../utils/teamLogos.js";
import { resolveEventTeams, resolveSidePick } from "../utils/teams.js";
import { expandTeamCodeInText, getTeamDisplayName } from "../utils/teamNames.js";
import { useAppSettings } from "../context/AppSettingsContext.jsx";

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

function isPendingPick(value) {
  const v = String(value ?? "").trim().toUpperCase();
  return !v || ["PENDIENTE", "N/A", "NAN", "RECONSTRUIDO", "PASS", "PASAR"].includes(v);
}

function marketBorderClass(hit, base = "border-white/15") {
  if (hit === true) return "border-emerald-400/70";
  if (hit === false) return "border-rose-400/70";
  return base;
}

function formatDecimalOdds(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return null;

  // If odds are already in decimal format, keep them as-is.
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

function formatLineValue(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return null;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(1);
}

function resolveLineValue(event, keys) {
  for (const key of keys) {
    const formatted = formatLineValue(event?.[key]);
    if (formatted) return formatted;
  }
  return null;
}

function toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
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
  const hasResult =
    hasFinalState || hasFinalScoreFallback;
  const isLive = !hasResult && (liveStates.includes(state) || hasLiveHint);
  return { hasResult, isLive, state };
}

function resolveSpreadSideTeam(event, teams, fullGamePick) {
  const explicit = String(event?.spread_side_team || "").trim();
  if (explicit) return explicit;

  const raw = String(event?.spread_pick || "").trim();
  const upper = raw.toUpperCase();
  if (teams.homeTeam && upper.includes(String(teams.homeTeam).toUpperCase())) return teams.homeTeam;
  if (teams.awayTeam && upper.includes(String(teams.awayTeam).toUpperCase())) return teams.awayTeam;
  if (upper.includes("HOME") || upper.includes("LOCAL")) return teams.homeTeam;
  if (upper.includes("AWAY") || upper.includes("VISITANTE") || upper.includes("VISITOR")) return teams.awayTeam;

  return fullGamePick || null;
}

function resolveSignedSpreadLine(event, teams, sideTeam) {
  const display = String(event?.spread_line_display || "").trim();
  if (display) return display;

  const normalized = toFiniteNumber(event?.spread_line_signed);
  if (normalized !== null && normalized !== 0) {
    return normalized > 0 ? `+${normalized.toFixed(1)}` : normalized.toFixed(1);
  }

  const homeSpread = toFiniteNumber(event?.home_spread);
  if (homeSpread !== null && homeSpread !== 0 && sideTeam) {
    const line = sideTeam === teams.homeTeam ? homeSpread : sideTeam === teams.awayTeam ? -homeSpread : null;
    if (line !== null && line !== 0) return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
  }

  const absLine = Math.abs(toFiniteNumber(event?.spread_abs) ?? 0);
  const predictedHomeMargin = toFiniteNumber(event?.predicted_home_margin);
  if (absLine > 0 && predictedHomeMargin !== null && sideTeam) {
    const sideMargin = sideTeam === teams.homeTeam ? predictedHomeMargin : -predictedHomeMargin;
    const line = sideMargin >= absLine ? -absLine : absLine;
    return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
  }

  const raw = String(event?.spread_pick || "").match(/([+-]?\d+(?:\.\d+)?)/);
  if (raw) {
    const line = Number(raw[1]);
    if (Number.isFinite(line) && line !== 0) return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
  }
  return null;
}

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
  f5: ["closing_f5_odds", "f5_odds"],
  total: ["closing_total_odds", "total_odds", "odds_total"],
  spread: ["closing_spread_odds", "spread_odds", "odds_spread"],
  btts: ["closing_btts_odds", "btts_odds"],
  corners: ["closing_corners_odds", "corners_odds"],
};

const LINE_KEYS = {
  spread: ["closing_spread_line", "home_spread", "spread_abs"],
  total: ["closing_total_line", "odds_over_under"],
};

const BASEBALL_SPORTS = new Set(["mlb", "kbo", "ncaa_baseball", "triple_a"]);
const BASKETBALL_SPORTS = new Set(["nba", "wnba", "euroleague"]);
const HOCKEY_SPORTS = new Set(["nhl"]);
const SOCCER_SPORTS = new Set(["liga_mx", "laliga", "bundesliga", "ligue1"]);

function normalizeSportKey(sportKey) {
  return String(sportKey || "").trim().toLowerCase();
}

function ordinalLabel(index) {
  if (index === 1) return "1er";
  if (index === 2) return "2do";
  if (index === 3) return "3er";
  if (index === 4) return "4to";
  return `${index}to`;
}

function resolveQ1MarketLabel(sportKey) {
  const normalizedSport = normalizeSportKey(sportKey);
  if (BASEBALL_SPORTS.has(normalizedSport)) return "1er Inning";
  if (BASKETBALL_SPORTS.has(normalizedSport)) return "1er Cuarto";
  if (HOCKEY_SPORTS.has(normalizedSport)) return "1er Periodo";
  if (SOCCER_SPORTS.has(normalizedSport)) return "1er Tiempo";
  return "1er Parcial";
}

function resolveSegmentRowLabel(sportKey, segmentIndex) {
  const normalizedSport = normalizeSportKey(sportKey);

  if (BASEBALL_SPORTS.has(normalizedSport)) {
    return `${ordinalLabel(segmentIndex)} Inning`;
  }

  if (BASKETBALL_SPORTS.has(normalizedSport)) {
    if (segmentIndex <= 4) return `${ordinalLabel(segmentIndex)} Cuarto`;
    return `OT ${segmentIndex - 4}`;
  }

  if (HOCKEY_SPORTS.has(normalizedSport)) {
    if (segmentIndex <= 3) return `${ordinalLabel(segmentIndex)} Periodo`;
    return segmentIndex === 4 ? "OT" : `OT ${segmentIndex - 3}`;
  }

  if (SOCCER_SPORTS.has(normalizedSport)) {
    if (segmentIndex === 1) return "1er Tiempo";
    if (segmentIndex === 2) return "2do Tiempo";
    if (segmentIndex === 3) return "Tiempo Extra 1";
    if (segmentIndex === 4) return "Tiempo Extra 2";
    if (segmentIndex === 5) return "Penales";
    return `Tiempo Extra ${segmentIndex - 2}`;
  }

  return `Parcial ${segmentIndex}`;
}

function resolveOddsForSecondary(event, marketKey) {
  if (marketKey === "q1") return resolveOddsValue(event, ODDS_KEYS.q1);
  if (marketKey === "f5") return resolveOddsValue(event, ODDS_KEYS.f5);
  if (marketKey === "total") return resolveOddsValue(event, ODDS_KEYS.total);
  if (marketKey === "spread") return resolveOddsValue(event, ODDS_KEYS.spread);
  if (marketKey === "btts") return resolveOddsValue(event, ODDS_KEYS.btts);
  if (marketKey === "h1_over15") return null;
  return null;
}

function resolveSecondaryMarket(event, sportKey, teams, q1MarketLabel) {
  const q1PickRaw = event.q1_pick;
  if (!isPendingPick(q1PickRaw)) {
    return {
      marketKey: "q1",
      label: q1MarketLabel,
      pick: expandTeamCodeInText(sportKey, resolveSidePick(q1PickRaw, teams)),
      confidence: event.q1_confidence,
      action: event.q1_action || "N/A",
      hit: toHitValue(event.q1_hit),
    };
  }

  const h1TotalPickRaw = event.h1_over15_recommended_pick || event.h1_over15_pick;
  if (!isPendingPick(h1TotalPickRaw)) {
    return {
      marketKey: "h1_over15",
      label: "1T O/U 1.5",
      pick: String(h1TotalPickRaw),
      confidence: event.h1_over15_confidence,
      action: event.h1_over15_action || "N/A",
      hit: toHitValue(event.h1_over15_hit ?? event.correct_h1_over15_adjusted ?? event.correct_h1_over15_base),
    };
  }

  const bttsPickRaw = event.btts_recommended_pick || event.btts_pick;
  if (!isPendingPick(bttsPickRaw)) {
    return {
      marketKey: "btts",
      label: "BTTS",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(bttsPickRaw, teams)),
      confidence: event.btts_confidence,
      action: event.btts_action || "N/A",
      hit: toHitValue(event.correct_btts_adjusted ?? event.correct_btts ?? event.correct_btts_base),
    };
  }

  const f5PickRaw = event.assists_pick || event.f5_pick;
  if (!isPendingPick(f5PickRaw) && String(f5PickRaw).toUpperCase().includes("F5")) {
    return {
      marketKey: "f5",
      label: "F5",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(f5PickRaw, teams)),
      confidence: event.extra_f5_confidence,
      action: "F5",
      hit: toHitValue(event.correct_f5 ?? event.correct_home_win_f5),
    };
  }

  const totalPickRaw = event.total_recommended_pick || event.total_pick;
  if (!isPendingPick(totalPickRaw)) {
    return {
      marketKey: "total",
      label: "Total O/U",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(totalPickRaw, teams)),
      confidence: event.total_confidence,
      action: event.total_action || "N/A",
      hit: toHitValue(event.correct_total_adjusted ?? event.correct_total),
    };
  }

  const spreadPickRaw = event.spread_pick;
  if (!isPendingPick(spreadPickRaw)) {
    return {
      marketKey: "spread",
      label: "Spread / ML",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)),
      confidence: event.spread_confidence,
      action: event.spread_market || "N/A",
      hit: toHitValue(event.correct_spread),
    };
  }

  return {
    marketKey: null,
    label: "Predicción secundaria",
    pick: "N/A",
    confidence: "-",
    action: "N/A",
    hit: null,
  };
}

function normalizeBetAction(actionValue) {
  const txt = String(actionValue || "").trim().toUpperCase();
  if (!txt) return "No apostar";
  if (["JUGAR", "APOSTAR", "PLAY", "BET"].includes(txt)) return "Apostar";
  if (["PASS", "PASAR", "NO BET", "NO_APUESTA", "SKIP"].includes(txt)) return "No apostar";
  return txt;
}

function formatSocialAction(actionValue) {
  const txt = String(actionValue || "").trim().toUpperCase();
  if (!txt || ["NO APOSTAR", "PASS", "PASAR", "NO BET", "NO_APUESTA", "SKIP", "N/A"].includes(txt)) {
    return "Seguimiento";
  }
  return "Alta prioridad";
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

function parseFinalTotalPoints(event) {
  const homeScore = Number(event?.home_score);
  const awayScore = Number(event?.away_score);
  if (Number.isFinite(homeScore) && Number.isFinite(awayScore) && (homeScore > 0 || awayScore > 0)) {
    return homeScore + awayScore;
  }

  const text = String(event?.final_score_text || "");
  const matches = [...text.matchAll(/(\d+)/g)].map((m) => Number(m[1]));
  if (matches.length >= 2) {
    return matches[matches.length - 2] + matches[matches.length - 1];
  }
  return null;
}

function resolveTotalHit(event, direction, totalLineValue) {
  const storedHit = toHitValue(event.correct_total_adjusted ?? event.correct_total);
  if (storedHit !== null) return storedHit;
  if (!direction || !Number.isFinite(totalLineValue) || event.result_available !== true) return null;

  const finalTotal = parseFinalTotalPoints(event);
  if (!Number.isFinite(finalTotal)) return null;
  if (direction === "Over") return finalTotal > totalLineValue;
  if (direction === "Under") return finalTotal < totalLineValue;
  return null;
}

function buildTotalPickDisplay(rawPick, direction, totalLineDisplay) {
  if (direction && totalLineDisplay !== "Por definir") {
    return `${direction} ${totalLineDisplay} puntos`;
  }
  if (direction) return direction;
  if (totalLineDisplay !== "Por definir") {
    return `Total ${totalLineDisplay} puntos`;
  }
  return rawPick || "N/A";
}

function resolveFirstHalfCard(event, sportKey, teams) {
  const h1PickRaw = String(event?.h1_pick || "").trim();
  if (isPendingPick(h1PickRaw)) return null;
  return {
    pick: expandTeamCodeInText(sportKey, resolveSidePick(h1PickRaw, teams)) || h1PickRaw,
    confidence: event?.h1_confidence,
    action: event?.h1_action || "N/A",
    hit: toHitValue(event?.h1_hit),
  };
}

function resolveFirstHalfTotalCard(event) {
  const pickRaw = String(event?.h1_over15_recommended_pick || event?.h1_over15_pick || "").trim();
  if (isPendingPick(pickRaw)) return null;
  return {
    pick: pickRaw,
    confidence: event?.h1_over15_confidence,
    action: event?.h1_over15_action || "N/A",
    hit: toHitValue(event?.h1_over15_hit ?? event?.correct_h1_over15_adjusted ?? event?.correct_h1_over15_base),
  };
}

function formatLiveClock(event) {
  const parts = [];
  const shortState = String(event?.status_description || "").trim();
  const detail = String(event?.status_detail || "").trim();
  if (shortState) parts.push(shortState);
  if (detail && detail !== shortState) parts.push(detail);
  return parts.join(" · ") || "En vivo";
}

function quarterValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildSportSegmentRows(event, sportKey) {
  const normalizedSport = normalizeSportKey(sportKey);
  if (BASEBALL_SPORTS.has(normalizedSport)) return [];

  const homeArray = trimTrailingNullScores(parseInningScoreArray(event?.home_segment_scores));
  const awayArray = trimTrailingNullScores(parseInningScoreArray(event?.away_segment_scores));

  const fallbackHome = trimTrailingNullScores([
    quarterValue(event?.home_q1_score ?? event?.home_q1),
    quarterValue(event?.home_q2_score ?? event?.home_q2),
    quarterValue(event?.home_q3_score ?? event?.home_q3),
    quarterValue(event?.home_q4_score ?? event?.home_q4),
  ]);
  const fallbackAway = trimTrailingNullScores([
    quarterValue(event?.away_q1_score ?? event?.away_q1),
    quarterValue(event?.away_q2_score ?? event?.away_q2),
    quarterValue(event?.away_q3_score ?? event?.away_q3),
    quarterValue(event?.away_q4_score ?? event?.away_q4),
  ]);

  const homeRaw = homeArray.length ? homeArray : fallbackHome;
  const awayRaw = awayArray.length ? awayArray : fallbackAway;
  const segmentCount = Math.max(homeRaw.length, awayRaw.length);
  if (!segmentCount) return [];

  const rows = Array.from({ length: segmentCount }, (_, index) => ({
    label: resolveSegmentRowLabel(sportKey, index + 1),
    home: index < homeRaw.length ? homeRaw[index] : null,
    away: index < awayRaw.length ? awayRaw[index] : null,
  }));

  return rows.filter((row) => row.home !== null || row.away !== null);
}

function parseInningScoreArray(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => quarterValue(entry));
  }
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed.map((entry) => quarterValue(entry));
      }
    } catch {
      return [];
    }
  }
  return [];
}

function trimTrailingNullScores(scores) {
  const out = [...scores];
  while (out.length && out[out.length - 1] === null) {
    out.pop();
  }
  return out;
}

function buildInningScoresFromColumns(event, prefix) {
  const scores = [];
  for (let inning = 1; inning <= 20; inning += 1) {
    const key = `${prefix}_r${inning}`;
    const hasKey = Object.prototype.hasOwnProperty.call(event || {}, key);
    const value = quarterValue(event?.[key]);
    if (hasKey || value !== null) {
      scores.push(value);
    }
  }
  return trimTrailingNullScores(scores);
}

function sumKnownScores(scores) {
  const values = (scores || []).filter((value) => Number.isFinite(value));
  if (!values.length) return null;
  return values.reduce((acc, value) => acc + value, 0);
}

function buildBaseballInningTable(event) {
  const homeFromArray = trimTrailingNullScores(parseInningScoreArray(event?.home_inning_scores));
  const awayFromArray = trimTrailingNullScores(parseInningScoreArray(event?.away_inning_scores));
  const homeFromColumns = buildInningScoresFromColumns(event, "home");
  const awayFromColumns = buildInningScoresFromColumns(event, "away");

  const homeRaw = homeFromArray.length ? homeFromArray : homeFromColumns;
  const awayRaw = awayFromArray.length ? awayFromArray : awayFromColumns;
  const inningsCount = Math.max(homeRaw.length, awayRaw.length);

  if (!inningsCount) {
    return {
      hasRows: false,
      innings: [],
      homeScores: [],
      awayScores: [],
      homeTotal: quarterValue(event?.home_score),
      awayTotal: quarterValue(event?.away_score),
    };
  }

  const innings = Array.from({ length: inningsCount }, (_, index) => index + 1);
  const homeScores = Array.from({ length: inningsCount }, (_, index) => (index < homeRaw.length ? homeRaw[index] : null));
  const awayScores = Array.from({ length: inningsCount }, (_, index) => (index < awayRaw.length ? awayRaw[index] : null));
  const parsedHomeTotal = quarterValue(event?.home_score);
  const parsedAwayTotal = quarterValue(event?.away_score);

  return {
    hasRows: true,
    innings,
    homeScores,
    awayScores,
    homeTotal: parsedHomeTotal ?? sumKnownScores(homeScores),
    awayTotal: parsedAwayTotal ?? sumKnownScores(awayScores),
  };
}

function toConfidenceValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildBestRecommendedPick({
  fullGamePick,
  fullGameConfidence,
  handicapPick,
  spreadConfidence,
  totalPick,
  totalConfidence,
  q1Pick,
  q1Confidence,
  q1MarketLabel,
  h1Pick,
  h1Confidence,
  bttsPick,
  bttsConfidence,
  cornersPick,
  cornersConfidence,
}) {
  const candidates = [
    { market: "Full Game", pick: fullGamePick, confidence: toConfidenceValue(fullGameConfidence) },
    { market: "Handicap", pick: handicapPick, confidence: toConfidenceValue(spreadConfidence) },
    { market: "Over/Under", pick: totalPick, confidence: toConfidenceValue(totalConfidence) },
    { market: q1MarketLabel || "1er Parcial", pick: q1Pick, confidence: toConfidenceValue(q1Confidence) },
    { market: "Primera Mitad", pick: h1Pick, confidence: toConfidenceValue(h1Confidence) },
    { market: "BTTS", pick: bttsPick, confidence: toConfidenceValue(bttsConfidence) },
    { market: "Corners", pick: cornersPick, confidence: toConfidenceValue(cornersConfidence) },
  ].filter((c) => c.pick && c.pick !== "N/A" && c.confidence !== null);

  if (!candidates.length) {
    return { market: "Sin mercado", pick: "N/A", confidence: null };
  }

  candidates.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  return candidates[0];
}

function marketKeyFromBestPickLabel(label) {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized === "full game") return "full_game";
  if (normalized === "handicap") return "spread";
  if (normalized === "over/under") return "total";
  if (normalized === "primer cuarto") return "q1";
  if (normalized.includes("inning") || normalized.includes("cuarto") || normalized.includes("periodo") || normalized.includes("tiempo")) {
    return "q1";
  }
  if (normalized === "primera mitad") return "h1";
  if (normalized === "btts") return "btts";
  if (normalized === "corners") return "corners";
  return "full_game";
}

function TeamBadge({ sportKey, abbr }) {
  const logoUrl = getTeamLogoUrl(sportKey, abbr);
  const fullName = getTeamDisplayName(sportKey, abbr);

  return (
    <div className="flex items-center gap-3 rounded-2xl bg-white/5 px-4 py-3">
      {logoUrl ? (
        <img
          src={logoUrl}
          alt={`Logo ${fullName}`}
          onError={(e) => {
            e.currentTarget.onerror = null;
            e.currentTarget.src = "/logos/default-team.svg";
          }}
          className="h-10 w-10 rounded-full bg-white/95 p-1 object-contain"
        />
      ) : null}
      <span className="text-lg font-semibold">{fullName}</span>
    </div>
  );
}

function MarketTierBadge({ tier }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-[0.16em] ${tierClasses(tier)}`}>
      {tierLabel(tier).replace(" PICK", "")}
    </span>
  );
}

function outcomeLabel(hit, socialMode) {
  if (hit === true) return socialMode ? "Correcto" : "ACIERTO";
  if (hit === false) return socialMode ? "Incorrecto" : "FALLO";
  return "N/A";
}

export default function DetailModal({ event, onClose, sportKey }) {
  const { socialMode } = useAppSettings();
  if (!event) return null;
  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const eventTier = resolveEventTier(event);

  const { hasResult, isLive } = resolveResultState(event);
  const hasLiveScore =
    (event.home_score !== undefined && event.home_score !== null) &&
    (event.away_score !== undefined && event.away_score !== null);
  const normalizedSportKey = normalizeSportKey(sportKey);
  const liveClock = formatLiveClock(event);
  const q1MarketLabel = resolveQ1MarketLabel(normalizedSportKey);
  const isBaseballSport = BASEBALL_SPORTS.has(normalizedSportKey);
  const segmentRows = buildSportSegmentRows(event, normalizedSportKey);
  const inningTable = isBaseballSport ? buildBaseballInningTable(event) : null;
  const hasPartialRows = isBaseballSport ? Boolean(inningTable?.hasRows) : segmentRows.length > 0;
  const showScoreCard = (isLive || hasResult) && (hasPartialRows || hasLiveScore);
  const scoreCardClass = isLive
    ? "rounded-3xl border border-rose-400/45 bg-gradient-to-br from-rose-500/10 via-[#1b1d24] to-black/30 p-5 shadow-[0_0_28px_rgba(251,113,133,0.10)]"
    : "rounded-3xl border border-emerald-400/35 bg-gradient-to-br from-emerald-500/10 via-[#1b1d24] to-black/30 p-5 shadow-[0_0_24px_rgba(16,185,129,0.08)]";
  const scoreValueClass = isLive ? "text-rose-100" : "text-emerald-100";
  const scoreBadgeClass = isLive
    ? "text-rose-200"
    : "text-emerald-200";
  const scorePulseClass = isLive
    ? "bg-rose-400 shadow-[0_0_12px_rgba(251,113,133,0.9)]"
    : "bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.85)]";
  const scoreStateLabel = isLive ? "Live" : (hasResult ? "Final" : "Marcador");
  const scoreStateDetail = isLive ? liveClock : (String(event?.status_description || "").trim() || "Final");
  const awayScoreName = isBaseballSport ? (teams.awayTeam || awayName) : awayName;
  const homeScoreName = isBaseballSport ? (teams.homeTeam || homeName) : homeName;

  const secondaryMarket = resolveSecondaryMarket(event, sportKey, teams, q1MarketLabel);
  const firstHalfCard = resolveFirstHalfCard(event, sportKey, teams);
  const firstHalfTotalCard = resolveFirstHalfTotalCard(event);
  const secondaryLabel = secondaryMarket.label;
  const secondaryPick = secondaryMarket.pick || "N/A";
  const secondaryMarketKey = secondaryMarket.marketKey;
  const secondaryTierKey = secondaryMarketKey || "q1";
  const fullGamePick = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  const secondaryConfidence = secondaryMarket.confidence ?? "-";
  const secondaryAction = secondaryMarket.action ?? "N/A";
  const secondaryActionDisplay = normalizeBetAction(secondaryAction);
  const secondaryHit = secondaryMarket.hit ?? null;
  const mainOdds = resolveOddsValue(event, ODDS_KEYS.moneyline);
  const secondaryOdds = resolveOddsForSecondary(event, secondaryMarketKey);
  const spreadOdds = resolveOddsValue(event, ODDS_KEYS.spread);
  const totalOdds = resolveOddsValue(event, ODDS_KEYS.total);
  const spreadLine = resolveLineValue(event, LINE_KEYS.spread);
  const totalLine = resolveLineValue(event, LINE_KEYS.total);
  const spreadMarket = String(event.spread_market || "").toUpperCase();
  const hasMlSpreadMarket = spreadMarket.includes("ML");
  const spreadLineDisplay = spreadLine || (hasMlSpreadMarket ? "ML" : "Por definir");
  const spreadOddsDisplay = spreadOdds || (hasMlSpreadMarket ? mainOdds : "Por definir");
  const totalLineDisplay = totalLine || "Por definir";
  const totalOddsDisplay = totalOdds || "Por definir";
  const spreadPickRaw = String(event.spread_pick || "").trim();
  const spreadSideTeam = resolveSpreadSideTeam(event, teams, fullGamePick);
  const spreadPickDisplayRaw = String(event.spread_pick_display || "").trim();
  const spreadPickFromEvent = spreadPickDisplayRaw
    ? expandTeamCodeInText(sportKey, resolveSidePick(spreadPickDisplayRaw, teams))
    : (expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)) || spreadPickRaw);
  const spreadPick = !isPendingPick(spreadPickRaw) || spreadPickDisplayRaw
    ? spreadPickFromEvent
    : ((spreadLineDisplay !== "Por definir" || hasMlSpreadMarket) ? (expandTeamCodeInText(sportKey, spreadSideTeam) || fullGamePick) : "N/A");
  const signedSpreadLineDisplay = resolveSignedSpreadLine(event, teams, spreadSideTeam) || spreadLineDisplay;
  const spreadHit = toHitValue(event.correct_spread);
  const totalPickRaw = event.total_recommended_pick || event.total_pick;
  const totalDirection = normalizeTotalDirection(totalPickRaw);
  const totalLineValue = extractNumericFromText(totalLineDisplay);
  const totalHit = resolveTotalHit(event, totalDirection, totalLineValue);
  const totalPickDisplay = buildTotalPickDisplay(totalPickRaw, totalDirection, totalLineDisplay);
  const q1PickRaw = String(event.q1_pick || "").trim();
  const q1PickDisplay = !isPendingPick(q1PickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(q1PickRaw, teams)) || q1PickRaw)
    : "N/A";
  const q1ConfidenceDisplay = !isPendingPick(q1PickRaw) ? event.q1_confidence : null;
  const secondaryTier = resolveMarketTier(event, secondaryTierKey);
  const h1PickDisplay = firstHalfCard?.pick || "N/A";
  const h1ConfidenceDisplay = firstHalfCard?.confidence ?? "-";
  const h1ActionDisplay = firstHalfCard?.action || "No apostar";
  const h1Hit = firstHalfCard?.hit ?? null;
  const h1Tier = resolveMarketTier(event, "h1");
  const h1TotalPickDisplay = firstHalfTotalCard?.pick || "N/A";
  const h1TotalConfidenceDisplay = firstHalfTotalCard?.confidence ?? "-";
  const h1TotalActionDisplay = firstHalfTotalCard?.action || "No apostar";
  const h1TotalHit = firstHalfTotalCard?.hit ?? null;
  const h1TotalTier = resolveMarketTier(event, "total");
  const handicapPickDisplay = spreadPick || "N/A";
  const spreadTier = resolveMarketTier(event, "spread");
  const totalTier = resolveMarketTier(event, "total");
  const bttsPickRaw = String(event.btts_recommended_pick || event.btts_pick || "").trim();
  const bttsPickDisplay = !isPendingPick(bttsPickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(bttsPickRaw, teams)) || bttsPickRaw)
    : "N/A";
  const bttsConfidenceDisplay = !isPendingPick(bttsPickRaw) ? (event.btts_confidence ?? "-") : "-";
  const bttsActionDisplay = event.btts_action || "No apostar";
  const bttsHit = !isPendingPick(bttsPickRaw)
    ? toHitValue(event.correct_btts_adjusted ?? event.correct_btts ?? event.correct_btts_base)
    : null;
  const bttsTier = resolveMarketTier(event, "btts");
  const bttsOdds = resolveOddsValue(event, ODDS_KEYS.btts);
  const cornersOdds = resolveOddsValue(event, ODDS_KEYS.corners);
  const secondaryResultBorderClass = marketBorderClass(secondaryHit);
  const resultBorderClass =
    event.full_game_hit === true
      ? "border-emerald-400/70"
      : event.full_game_hit === false
        ? "border-rose-400/70"
        : "border-white/15";

  const propBest = buildBestRecommendedPick({
    fullGamePick,
    fullGameConfidence: event.full_game_confidence,
    handicapPick: handicapPickDisplay,
    spreadConfidence: event.spread_confidence,
    totalPick: totalPickDisplay,
    totalConfidence: event.total_confidence,
    q1Pick: q1PickDisplay,
    q1Confidence: q1ConfidenceDisplay,
    q1MarketLabel,
    h1Pick: h1PickDisplay,
    h1Confidence: h1ConfidenceDisplay,
    bttsPick: bttsPickDisplay,
    bttsConfidence: bttsConfidenceDisplay,
    cornersPick: event.corners_pick,
    cornersConfidence: event.corners_confidence,
  });
  const propLabel = "Pick recomendado";
  const propPick = propBest.pick;
  const propConfidence = propBest.confidence;
  const hasCorners = Boolean(event.corners_pick);
  const propHit = toHitValue(event.correct_f5 ?? event.correct_home_win_f5);
  const propTier = resolveMarketTier(event, marketKeyFromBestPickLabel(propBest.market));
  const cornersHit = toHitValue(event.correct_corners_adjusted ?? event.correct_corners_base ?? event.correct_corners);
  const cornersTier = resolveMarketTier(event, "corners");
  const fullGameLabel = socialMode ? "Proyeccion principal" : "Ganador del partido";
  const secondaryLabelText = socialMode ? "Proyeccion de arranque" : secondaryLabel;
  const h1Label = socialMode ? "Proyeccion primera mitad" : "Primera Mitad";
  const h1TotalLabel = socialMode ? "Ritmo primera mitad" : "1T O/U 1.5";
  const spreadLabel = socialMode ? "Proyeccion de margen" : "Handicap";
  const totalLabel = socialMode ? "Proyeccion total" : "Over/Under";
  const propLabelText = socialMode ? "Proyeccion recomendada" : propLabel;
  const bttsLabel = socialMode ? "Coincidencia ofensiva" : "BTTS";
  const cornersLabel = socialMode ? "Actividad ofensiva" : "Corners O/U";
  const lineLabel = socialMode ? "Referencia" : "Linea";
  const oddsLabel = socialMode ? "Valor modelo" : "Cuota";
  const actionLabel = socialMode ? "Enfoque" : "Accion";
  const resultLabel = socialMode ? "Resultado del modelo" : "Resultado";
  const finalLabel = socialMode ? "Marcador final" : "Resultado real del juego";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="relative max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-3xl border border-white/10 bg-[#171a21] shadow-2xl shadow-black/45">
        <div className="sticky top-0 z-10 flex items-start justify-between border-b border-white/10 bg-[#171a21]/95 px-6 py-5 backdrop-blur">
          <div>
            <p className="text-sm text-white/55">
              {event.date} · {event.time || "Sin hora"}
            </p>
            <h3 className="mt-1 text-3xl font-semibold leading-tight">
              {awayName} @ {homeName}
            </h3>
            <div className="mt-4 flex flex-wrap gap-3">
              <TeamBadge sportKey={sportKey} abbr={teams.awayTeam} />
              <TeamBadge sportKey={sportKey} abbr={teams.homeTeam} />
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm text-white/70 transition hover:bg-white/10 hover:text-white"
          >
            Cerrar
          </button>
        </div>

        <div className="space-y-5 p-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className={`rounded-full border px-4 py-2 text-xs ${tierClasses(eventTier)}`}>
              {tierLabel(eventTier)}
            </span>

            <span className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm text-white/70">
              {event.game_name}
            </span>
            {hasResult && (
              <span className="rounded-full border border-emerald-400/35 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-100">
                FINAL
              </span>
            )}
            {isLive && hasLiveScore && (
              <span className="rounded-full border border-rose-400/60 bg-rose-500/10 px-4 py-2 text-sm text-rose-100">
                LIVE · {liveClock}
              </span>
            )}
          </div>

          {showScoreCard && (
            <div className={scoreCardClass}>
              <div className="flex items-center justify-between gap-4">
                <div className="space-y-3 flex-1">
                  <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <img
                        src={getTeamLogoUrl(sportKey, teams.awayTeam) || "/logos/default-team.svg"}
                        alt={`Logo ${awayName}`}
                        className="h-10 w-10 shrink-0 rounded-full bg-white/95 p-1 object-contain"
                      />
                      <span
                        className="truncate text-base font-semibold text-white md:text-lg"
                        title={awayName}
                      >
                        {awayScoreName}
                      </span>
                    </div>
                    <span className={`ml-3 shrink-0 text-3xl font-bold ${scoreValueClass}`}>
                      {(quarterValue(event.away_score) ?? inningTable?.awayTotal) ?? "-"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <img
                        src={getTeamLogoUrl(sportKey, teams.homeTeam) || "/logos/default-team.svg"}
                        alt={`Logo ${homeName}`}
                        className="h-10 w-10 shrink-0 rounded-full bg-white/95 p-1 object-contain"
                      />
                      <span
                        className="truncate text-base font-semibold text-white md:text-lg"
                        title={homeName}
                      >
                        {homeScoreName}
                      </span>
                    </div>
                    <span className={`ml-3 shrink-0 text-3xl font-bold ${scoreValueClass}`}>
                      {(quarterValue(event.home_score) ?? inningTable?.homeTotal) ?? "-"}
                    </span>
                  </div>
                </div>

                <div className="min-w-[220px] rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div className={`flex items-center gap-2 ${scoreBadgeClass}`}>
                      <span className={`h-2.5 w-2.5 rounded-full ${scorePulseClass} ${isLive ? "animate-pulse" : ""}`} />
                      <span className="text-xs font-semibold uppercase tracking-[0.2em]">{scoreStateLabel}</span>
                    </div>
                    <span className="text-sm text-white/70">{scoreStateDetail}</span>
                  </div>

                  {isBaseballSport && inningTable?.hasRows ? (
                    <div className="overflow-x-auto rounded-xl border border-white/10">
                      <table className="min-w-full text-sm text-white/85">
                        <thead className="bg-white/5 text-[11px] uppercase tracking-[0.16em] text-white/45">
                          <tr>
                            <th className="px-3 py-2 text-left">Equipo</th>
                            {inningTable.innings.map((inning) => (
                              <th key={`inning-head-${inning}`} className="min-w-[30px] px-2 py-2 text-center">
                                {inning}
                              </th>
                            ))}
                            <th className="px-3 py-2 text-center">R</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr className="border-t border-white/10">
                            <td className="px-3 py-2 font-semibold text-white/70">{awayName}</td>
                            {inningTable.awayScores.map((score, idx) => (
                              <td key={`away-inning-${idx + 1}`} className="px-2 py-2 text-center">
                                {score ?? "-"}
                              </td>
                            ))}
                            <td className="px-3 py-2 text-center font-semibold text-white">
                              {inningTable.awayTotal ?? "-"}
                            </td>
                          </tr>
                          <tr className="border-t border-white/10">
                            <td className="px-3 py-2 font-semibold text-white/70">{homeName}</td>
                            {inningTable.homeScores.map((score, idx) => (
                              <td key={`home-inning-${idx + 1}`} className="px-2 py-2 text-center">
                                {score ?? "-"}
                              </td>
                            ))}
                            <td className="px-3 py-2 text-center font-semibold text-white">
                              {inningTable.homeTotal ?? "-"}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  ) : segmentRows.length > 0 ? (
                    <div className="overflow-hidden rounded-xl border border-white/10">
                      <div className="grid grid-cols-[132px_1fr_1fr] bg-white/5 px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-white/45">
                        <span>Parciales</span>
                        <span className="text-center">{awayName}</span>
                        <span className="text-center">{homeName}</span>
                      </div>
                      {segmentRows.map((row) => (
                        <div key={row.label} className="grid grid-cols-[132px_1fr_1fr] border-t border-white/10 px-3 py-2 text-sm text-white/85">
                          <span className="truncate font-semibold text-white/60" title={row.label}>{row.label}</span>
                          <span className="text-center">{row.away ?? "-"}</span>
                          <span className="text-center">{row.home ?? "-"}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-black/15 p-5">
              <p className="text-sm text-white/50">{fullGameLabel}</p>
              <p className="mt-1 text-2xl font-semibold">{fullGamePick}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{socialMode ? "Intensidad" : "Confianza"}</p>
                  <p className="mt-1 font-semibold">{event.full_game_confidence}%</p>
                </div>

                {!socialMode && (
                  <div className="rounded-xl bg-white/5 p-3">
                    <p className="text-white/50">{oddsLabel}</p>
                    <p className="mt-1 font-semibold">{mainOdds || "N/A"}</p>
                  </div>
                )}
              </div>
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 ${secondaryResultBorderClass}`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-white/50">{secondaryLabelText}</p>
                <MarketTierBadge tier={secondaryTier} />
              </div>
              <p className="mt-1 text-2xl font-semibold">{secondaryPick}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{socialMode ? "Intensidad" : "Confianza"}</p>
                  <p className="mt-1 font-semibold">
                    {secondaryConfidence === "-" ? "-" : `${secondaryConfidence}%`}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{actionLabel}</p>
                  <p className="mt-1 font-semibold">{socialMode ? formatSocialAction(secondaryActionDisplay) : secondaryActionDisplay}</p>
                </div>
              </div>

              {!socialMode && <p className="mt-2 text-xs text-white/60">{oddsLabel} decimal: {secondaryOdds || "N/A"}</p>}

              {hasResult && secondaryHit !== undefined && secondaryHit !== null && (
                <p className="mt-3 text-sm font-semibold text-white/85">
                  {resultLabel}: {outcomeLabel(secondaryHit, socialMode)}
                </p>
              )}
            </div>
          </div>

          {hasResult && (
            <div className={`rounded-2xl border bg-black/20 p-5 ${resultBorderClass}`}>
              <p className="text-sm text-white/50">{finalLabel}</p>
              <p className="mt-1 text-lg font-semibold">{event.final_score_text}</p>

              <div className="mt-4 grid gap-3 md:grid-cols-2 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{fullGameLabel}</p>
                  <p className="mt-1 font-semibold">
                    {outcomeLabel(event.full_game_hit, socialMode)}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{secondaryLabelText}</p>
                  <p className="mt-1 font-semibold">
                    {outcomeLabel(secondaryHit, socialMode)}
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {firstHalfCard && (
              <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(h1Hit)}`}>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm text-white/50">{h1Label}</p>
                  <MarketTierBadge tier={h1Tier} />
                </div>
                <p className="mt-1 text-lg font-semibold">{h1PickDisplay}</p>
                <p className="mt-2 text-sm text-white/65">{socialMode ? "Intensidad" : "Confianza"}: {h1ConfidenceDisplay === "-" ? "-" : `${h1ConfidenceDisplay}%`}</p>
                <p className="mt-1 text-sm text-white/65">{socialMode ? "Enfoque" : "Accion"}: {socialMode ? formatSocialAction(h1ActionDisplay) : h1ActionDisplay}</p>
                {hasResult && h1Hit !== undefined && h1Hit !== null && (
                  <p className="mt-2 text-sm font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(h1Hit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {firstHalfTotalCard && (
              <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(h1TotalHit)}`}>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm text-white/50">{h1TotalLabel}</p>
                  <MarketTierBadge tier={h1TotalTier} />
                </div>
                <p className="mt-1 text-lg font-semibold">{h1TotalPickDisplay}</p>
                <p className="mt-2 text-sm text-white/65">{socialMode ? "Intensidad" : "Confianza"}: {h1TotalConfidenceDisplay === "-" ? "-" : `${h1TotalConfidenceDisplay}%`}</p>
                <p className="mt-1 text-sm text-white/65">{actionLabel}: {socialMode ? formatSocialAction(h1TotalActionDisplay) : normalizeBetAction(h1TotalActionDisplay)}</p>
                {hasResult && h1TotalHit !== undefined && h1TotalHit !== null && (
                  <p className="mt-2 text-sm font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(h1TotalHit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {bttsPickDisplay !== "N/A" && secondaryMarketKey !== "btts" && (
              <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(bttsHit)}`}>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm text-white/50">{bttsLabel}</p>
                  <MarketTierBadge tier={bttsTier} />
                </div>
                <p className="mt-1 text-lg font-semibold">{bttsPickDisplay}</p>
                <p className="mt-2 text-sm text-white/65">{socialMode ? "Intensidad" : "Confianza"}: {bttsConfidenceDisplay === "-" ? "-" : `${bttsConfidenceDisplay}%`}</p>
                <p className="mt-1 text-sm text-white/65">{actionLabel}: {socialMode ? formatSocialAction(bttsActionDisplay) : normalizeBetAction(bttsActionDisplay)}</p>
                {!socialMode && <p className="mt-1 text-sm text-white/65">{oddsLabel}: {bttsOdds || "N/A"}</p>}
                {hasResult && bttsHit !== undefined && bttsHit !== null && (
                  <p className="mt-2 text-sm font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(bttsHit, socialMode)}
                  </p>
                )}
              </div>
            )}

            <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(spreadHit)}`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-white/50">{spreadLabel}</p>
                <MarketTierBadge tier={spreadTier} />
              </div>
              <p className="mt-1 text-lg font-semibold">{socialMode ? "Lectura" : "Pick"}: {handicapPickDisplay}</p>
              {!socialMode && <p className="mt-2 text-sm text-white/65">{lineLabel}: {signedSpreadLineDisplay}</p>}
              {!socialMode && <p className="mt-1 text-sm text-white/65">{oddsLabel}: {spreadOddsDisplay}</p>}
              {hasResult && spreadHit !== undefined && spreadHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  {resultLabel}: {outcomeLabel(spreadHit, socialMode)}
                </p>
              )}
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(totalHit)}`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-white/50">{totalLabel}</p>
                <MarketTierBadge tier={totalTier} />
              </div>
              <p className="mt-1 text-lg font-semibold">{socialMode ? "Lectura" : "Pick"}: {totalPickDisplay}</p>
              {!socialMode && <p className="mt-2 text-sm text-white/65">Linea: {totalLineDisplay}</p>}
              {!socialMode && <p className="mt-1 text-sm text-white/65">{oddsLabel}: {totalOddsDisplay}</p>}
              {hasResult && totalHit !== undefined && totalHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  {resultLabel}: {outcomeLabel(totalHit, socialMode)}
                </p>
              )}
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 md:col-span-2 xl:col-span-1 ${marketBorderClass(propHit)}`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-white/50">{propLabelText}</p>
                <MarketTierBadge tier={propTier} />
              </div>
              <p className="mt-1 text-lg font-semibold leading-snug">{propPick}</p>
              <p className="mt-2 text-sm text-white/65">{socialMode ? "Base del modelo" : "Mercado"}: {propBest.market}</p>
              {propConfidence && (
                <p className="mt-2 text-sm text-white/65">{socialMode ? "Intensidad" : "Confianza"}: {propConfidence}%</p>
              )}
              {hasResult && propHit !== undefined && propHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  {resultLabel}: {outcomeLabel(propHit, socialMode)}
                </p>
              )}
            </div>

            {hasCorners && (
              <div className={`rounded-2xl border bg-cyan-500/10 p-5 md:col-span-2 xl:col-span-3 ${marketBorderClass(cornersHit, "border-cyan-400/40")}`}>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm text-cyan-100">{cornersLabel}</p>
                  <MarketTierBadge tier={cornersTier} />
                </div>
                <p className="mt-1 text-lg font-semibold text-cyan-50">{event.corners_pick}</p>
                <div className="mt-2 flex flex-wrap gap-4 text-sm text-cyan-100/90">
                  {!socialMode && <span>Linea: {event.corners_line ?? 9.5}</span>}
                  {!socialMode && <span>{oddsLabel}: {cornersOdds || "N/A"}</span>}
                  <span>{socialMode ? "Intensidad" : "Confianza"}: {event.corners_confidence ?? "-"}%</span>
                  <span>Score: {event.corners_recommended_score ?? "-"}</span>
                  <span>{socialMode ? "Enfoque" : "Accion"}: {socialMode ? formatSocialAction(event.corners_action) : (event.corners_action || "N/A")}</span>
                </div>
                {hasResult && cornersHit !== undefined && cornersHit !== null && (
                  <p className="mt-2 text-sm font-semibold text-cyan-50">
                    {resultLabel}: {outcomeLabel(cornersHit, socialMode)}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
