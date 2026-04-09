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
  const hasResult =
    event?.result_available === true ||
    ["post", "final", "completed"].includes(state) ||
    Boolean(String(event?.final_score_text || "").trim());
  const isLive = !hasResult && state === "in";
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

function resolveOddsForSecondary(event, label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("q1") || key.includes("yrfi")) return resolveOddsValue(event, ODDS_KEYS.q1);
  if (key.includes("f5")) return resolveOddsValue(event, ODDS_KEYS.f5);
  if (key.includes("total")) return resolveOddsValue(event, ODDS_KEYS.total);
  if (key.includes("spread")) return resolveOddsValue(event, ODDS_KEYS.spread);
  if (key.includes("btts")) return resolveOddsValue(event, ODDS_KEYS.btts);
  return null;
}

function resolveSecondaryMarket(event, sportKey, teams) {
  const q1PickRaw = event.q1_pick;
  if (!isPendingPick(q1PickRaw)) {
    const q1Label = "Primer Cuarto";
    return {
      label: q1Label,
      pick: expandTeamCodeInText(sportKey, resolveSidePick(q1PickRaw, teams)),
      confidence: event.q1_confidence,
      action: event.q1_action || "N/A",
      hit: toHitValue(event.q1_hit),
    };
  }

  const bttsPickRaw = event.btts_recommended_pick || event.btts_pick;
  if (!isPendingPick(bttsPickRaw)) {
    return {
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
      label: "Spread / ML",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)),
      confidence: event.spread_confidence,
      action: event.spread_market || "N/A",
      hit: toHitValue(event.correct_spread),
    };
  }

  return {
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

function buildQuarterRows(event) {
  const rows = [
    { label: "Q1", home: quarterValue(event?.home_q1_score ?? event?.home_q1), away: quarterValue(event?.away_q1_score ?? event?.away_q1) },
    { label: "Q2", home: quarterValue(event?.home_q2_score ?? event?.home_q2), away: quarterValue(event?.away_q2_score ?? event?.away_q2) },
    { label: "Q3", home: quarterValue(event?.home_q3_score ?? event?.home_q3), away: quarterValue(event?.away_q3_score ?? event?.away_q3) },
    { label: "Q4", home: quarterValue(event?.home_q4_score ?? event?.home_q4), away: quarterValue(event?.away_q4_score ?? event?.away_q4) },
  ];
  return rows.filter((row) => row.home !== null || row.away !== null);
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
    { market: "Primer Cuarto", pick: q1Pick, confidence: toConfidenceValue(q1Confidence) },
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
  const liveClock = formatLiveClock(event);
  const quarterRows = buildQuarterRows(event);

  const secondaryMarket = resolveSecondaryMarket(event, sportKey, teams);
  const firstHalfCard = resolveFirstHalfCard(event, sportKey, teams);
  const secondaryLabel = secondaryMarket.label;
  const secondaryPick = secondaryMarket.pick || "N/A";
  const fullGamePick = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  const secondaryConfidence = secondaryMarket.confidence ?? "-";
  const secondaryAction = secondaryMarket.action ?? "N/A";
  const secondaryHit = secondaryMarket.hit ?? null;
  const mainOdds = resolveOddsValue(event, ODDS_KEYS.moneyline);
  const secondaryOdds = resolveOddsForSecondary(event, secondaryLabel);
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
  const q1PickDisplay = secondaryLabel === "Primer Cuarto" ? secondaryPick : "N/A";
  const q1ConfidenceDisplay = secondaryLabel === "Primer Cuarto" ? secondaryConfidence : "-";
  const q1ActionDisplay = secondaryLabel === "Primer Cuarto" ? normalizeBetAction(secondaryAction) : "No apostar";
  const q1Hit = secondaryLabel === "Primer Cuarto" ? secondaryHit : null;
  const q1Tier = resolveMarketTier(event, "q1");
  const h1PickDisplay = firstHalfCard?.pick || "N/A";
  const h1ConfidenceDisplay = firstHalfCard?.confidence ?? "-";
  const h1ActionDisplay = firstHalfCard?.action || "No apostar";
  const h1Hit = firstHalfCard?.hit ?? null;
  const h1Tier = resolveMarketTier(event, "h1");
  const handicapPickDisplay = spreadPick || "N/A";
  const spreadTier = resolveMarketTier(event, "spread");
  const totalTier = resolveMarketTier(event, "total");
  const cornersOdds = resolveOddsValue(event, ODDS_KEYS.corners);
  const secondaryResultBorderClass = marketBorderClass(q1Hit);
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
    h1Pick: h1PickDisplay,
    h1Confidence: h1ConfidenceDisplay,
    bttsPick: event.btts_recommended_pick || event.btts_pick,
    bttsConfidence: event.btts_confidence,
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
  const q1Label = socialMode ? "Proyeccion de arranque" : "Primer Cuarto";
  const h1Label = socialMode ? "Proyeccion primera mitad" : "Primera Mitad";
  const spreadLabel = socialMode ? "Proyeccion de margen" : "Handicap";
  const totalLabel = socialMode ? "Proyeccion total" : "Over/Under";
  const propLabelText = socialMode ? "Proyeccion recomendada" : propLabel;
  const cornersLabel = socialMode ? "Actividad ofensiva" : "Corners O/U";
  const lineLabel = socialMode ? "Referencia" : "Linea";
  const oddsLabel = socialMode ? "Valor modelo" : "Cuota";
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

          {isLive && hasLiveScore && (
            <div className="rounded-3xl border border-rose-400/45 bg-gradient-to-br from-rose-500/10 via-[#1b1d24] to-black/30 p-5 shadow-[0_0_28px_rgba(251,113,133,0.10)]">
              <div className="flex items-center justify-between gap-4">
                <div className="space-y-3 flex-1">
                  <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                    <div className="flex items-center gap-3">
                      <img
                        src={getTeamLogoUrl(sportKey, teams.awayTeam) || "/logos/default-team.svg"}
                        alt={`Logo ${awayName}`}
                        className="h-10 w-10 rounded-full bg-white/95 p-1 object-contain"
                      />
                      <span className="text-lg font-semibold text-white">{awayName}</span>
                    </div>
                    <span className="text-3xl font-bold text-rose-100">{event.away_score}</span>
                  </div>

                  <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                    <div className="flex items-center gap-3">
                      <img
                        src={getTeamLogoUrl(sportKey, teams.homeTeam) || "/logos/default-team.svg"}
                        alt={`Logo ${homeName}`}
                        className="h-10 w-10 rounded-full bg-white/95 p-1 object-contain"
                      />
                      <span className="text-lg font-semibold text-white">{homeName}</span>
                    </div>
                    <span className="text-3xl font-bold text-rose-100">{event.home_score}</span>
                  </div>
                </div>

                <div className="min-w-[220px] rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-rose-200">
                      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-rose-400 shadow-[0_0_12px_rgba(251,113,133,0.9)]" />
                      <span className="text-xs font-semibold uppercase tracking-[0.2em]">Live</span>
                    </div>
                    <span className="text-sm text-white/70">{liveClock}</span>
                  </div>

                  {quarterRows.length > 0 && (
                    <div className="overflow-hidden rounded-xl border border-white/10">
                      <div className="grid grid-cols-[72px_1fr_1fr] bg-white/5 px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-white/45">
                        <span>Parcial</span>
                        <span className="text-center">{awayName}</span>
                        <span className="text-center">{homeName}</span>
                      </div>
                      {quarterRows.map((row) => (
                        <div key={row.label} className="grid grid-cols-[72px_1fr_1fr] border-t border-white/10 px-3 py-2 text-sm text-white/85">
                          <span className="font-semibold text-white/60">{row.label}</span>
                          <span className="text-center">{row.away ?? "-"}</span>
                          <span className="text-center">{row.home ?? "-"}</span>
                        </div>
                      ))}
                    </div>
                  )}
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
                <p className="text-sm text-white/50">{q1Label}</p>
                <MarketTierBadge tier={q1Tier} />
              </div>
              <p className="mt-1 text-2xl font-semibold">{q1PickDisplay}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{socialMode ? "Intensidad" : "Confianza"}</p>
                  <p className="mt-1 font-semibold">
                    {q1ConfidenceDisplay === "-" ? "-" : `${q1ConfidenceDisplay}%`}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{socialMode ? "Enfoque" : oddsLabel}</p>
                  <p className="mt-1 font-semibold">{socialMode ? formatSocialAction(q1ActionDisplay) : q1ActionDisplay}</p>
                </div>
              </div>

              {!socialMode && <p className="mt-2 text-xs text-white/60">{oddsLabel} decimal: {secondaryOdds || "N/A"}</p>}

              {hasResult && q1Hit !== undefined && q1Hit !== null && (
                <p className="mt-3 text-sm font-semibold text-white/85">
                  {resultLabel}: {outcomeLabel(q1Hit, socialMode)}
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
                  <p className="text-white/50">{q1Label}</p>
                  <p className="mt-1 font-semibold">
                    {outcomeLabel(q1Hit, socialMode)}
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
