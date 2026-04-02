import { resolveEventTier, tierClasses, tierLabel } from "../utils/picks.js";
import { getTeamLogoUrl } from "../utils/teamLogos.js";
import { resolveEventTeams, resolveSidePick } from "../utils/teams.js";
import { expandTeamCodeInText, getTeamDisplayName } from "../utils/teamNames.js";

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
      action: event.q1_action,
      hit: toHitValue(event.q1_hit),
    };
  }

  const bttsPickRaw = event.btts_recommended_pick || event.btts_pick;
  if (!isPendingPick(bttsPickRaw)) {
    return {
      label: "BTTS",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(bttsPickRaw, teams)),
      confidence: event.btts_confidence,
      hit: toHitValue(event.correct_btts_adjusted ?? event.correct_btts ?? event.correct_btts_base),
    };
  }

  const f5PickRaw = event.assists_pick || event.f5_pick;
  if (!isPendingPick(f5PickRaw) && String(f5PickRaw).toUpperCase().includes("F5")) {
    return {
      label: "F5",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(f5PickRaw, teams)),
      confidence: event.extra_f5_confidence,
      hit: toHitValue(event.correct_f5 ?? event.correct_home_win_f5),
    };
  }

  const totalPickRaw = event.total_recommended_pick || event.total_pick;
  if (!isPendingPick(totalPickRaw)) {
    return {
      label: "Total O/U",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(totalPickRaw, teams)),
      confidence: event.total_confidence,
      hit: toHitValue(event.correct_total_adjusted ?? event.correct_total),
    };
  }

  const spreadPickRaw = event.spread_pick;
  if (!isPendingPick(spreadPickRaw)) {
    return {
      label: "Spread / ML",
      pick: expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)),
      confidence: event.spread_confidence,
      hit: toHitValue(event.correct_spread),
    };
  }

  return null;
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

function normalizeBetAction(actionValue) {
  const txt = String(actionValue || "").trim().toUpperCase();
  if (!txt) return "No apostar";
  if (["JUGAR", "APOSTAR", "PLAY", "BET"].includes(txt)) return "Apostar";
  if (["PASS", "PASAR", "NO BET", "NO_APUESTA", "SKIP"].includes(txt)) return "No apostar";
  return txt;
}

function normalizeTotalDirection(rawPick) {
  const txt = String(rawPick || "").toUpperCase();
  if (txt.includes("OVER")) return "Over";
  if (txt.includes("UNDER")) return "Under";
  return null;
}

function marketBorderClass(hit, base = "border-white/10") {
  if (hit === true) return "border-emerald-400/70";
  if (hit === false) return "border-rose-400/70";
  return base;
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

function formatLiveClock(event) {
  const parts = [];
  const shortState = String(event?.status_description || "").trim();
  const detail = String(event?.status_detail || "").trim();
  if (shortState) parts.push(shortState);
  if (detail && detail !== shortState) parts.push(detail);
  return parts.join(" · ") || "En vivo";
}

function CompactScoreRow({ sportKey, abbr, score }) {
  const logoUrl = getTeamLogoUrl(sportKey, abbr);
  const fullName = getTeamDisplayName(sportKey, abbr);

  return (
    <div className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2">
      <div className="flex items-center gap-3">
        <img
          src={logoUrl || "/logos/default-team.svg"}
          alt={`Logo ${fullName}`}
          className="h-8 w-8 rounded-full bg-white/95 p-1 object-contain"
        />
        <span className="text-sm font-semibold text-white">{fullName}</span>
      </div>
      <span className="text-2xl font-bold text-rose-100">{score}</span>
    </div>
  );
}

function TeamRow({ sportKey, abbr }) {
  const logoUrl = getTeamLogoUrl(sportKey, abbr);
  const fullName = getTeamDisplayName(sportKey, abbr);

  return (
    <div className="flex items-center gap-3">
      {logoUrl ? (
        <img
          src={logoUrl}
          alt={`Logo ${fullName}`}
          loading="lazy"
          onError={(e) => {
            e.currentTarget.onerror = null;
            e.currentTarget.src = "/logos/default-team.svg";
          }}
          className="team-logo rounded-full bg-white/95 p-1 shadow-sm"
        />
      ) : (
        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-white/5 text-xs font-semibold text-white/70">
          {abbr}
        </div>
      )}
      <span>{fullName}</span>
    </div>
  );
}

export default function EventCard({ event, onOpen, sportKey }) {
  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const eventTier = resolveEventTier(event);
  const hasResult = event.result_available === true;
  const isLive = !hasResult && String(event.status_state || "").toLowerCase() === "in";
  const hasLiveScore =
    (event.home_score !== undefined && event.home_score !== null) &&
    (event.away_score !== undefined && event.away_score !== null);
  const gameHit = event.full_game_hit;
  const secondaryMarket = resolveSecondaryMarket(event, sportKey, teams);
  const firstHalfCard = resolveFirstHalfCard(event, sportKey, teams);
  const secondaryPick = secondaryMarket?.pick;
  const secondaryAction = secondaryMarket?.action;
  const fullGamePick = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  const secondaryHit = secondaryMarket?.hit ?? null;
  const secondaryConfidence = secondaryMarket?.confidence;
  const secondaryLabel = secondaryMarket?.label || "Pick secundario";
  const mainOdds = resolveOddsValue(event, ODDS_KEYS.moneyline);
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
  const secondaryOdds = resolveOddsForSecondary(event, secondaryLabel);
  const cornersOdds = resolveOddsValue(event, ODDS_KEYS.corners);
  const spreadPickRaw = String(event.spread_pick || "").trim();
  const spreadPickFromEvent = expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)) || spreadPickRaw;
  const spreadPick = !isPendingPick(spreadPickRaw)
    ? spreadPickFromEvent
    : ((spreadLineDisplay !== "Por definir" || hasMlSpreadMarket) ? fullGamePick : null);
  const spreadHit = toHitValue(event.correct_spread);
  const totalPickRaw = event.total_recommended_pick || event.total_pick;
  const totalDirection = normalizeTotalDirection(totalPickRaw);
  const totalLineValue = extractNumericFromText(totalLineDisplay);
  const totalHit = resolveTotalHit(event, totalDirection, totalLineValue);
  const totalPickDisplay = buildTotalPickDisplay(totalPickRaw, totalDirection, totalLineDisplay);
  const spreadPredictionLabel = spreadPick || fullGamePick || "N/A";
  const q1BetActionLabel = normalizeBetAction(secondaryAction);
  const cornersPick = event.corners_pick;
  const cornersConfidence = event.corners_confidence;
  const cornersAction = event.corners_action;
  const cornersHit = toHitValue(event.correct_corners_adjusted ?? event.correct_corners_base ?? event.correct_corners);
  const resultBorderClass =
    gameHit === true
      ? "border-emerald-400/70"
      : gameHit === false
        ? "border-rose-400/70"
        : "border-white/15";
  const secondaryResultBorderClass =
    secondaryHit === true
      ? "border-emerald-400/70"
      : secondaryHit === false
        ? "border-rose-400/70"
        : "border-white/10";
  const cornersResultBorderClass =
    cornersHit === true
      ? "border-emerald-400/70"
      : cornersHit === false
        ? "border-rose-400/70"
        : "border-cyan-400/40";
  const spreadResultBorderClass = marketBorderClass(spreadHit);
  const totalResultBorderClass = marketBorderClass(totalHit);
  const cardBorderClass = isLive ? "border-rose-400/70 shadow-[0_0_30px_rgba(251,113,133,0.12)]" : "border-white/15";
  const cardHoverClass = isLive ? "hover:border-rose-300/80" : "hover:border-amber-300/70";
  const liveClock = formatLiveClock(event);

  return (
    <button
      onClick={() => onOpen(event)}
      className={`rounded-3xl border bg-[#171a21] p-0 text-left transition hover:-translate-y-0.5 hover:shadow-2xl hover:shadow-black/35 ${cardBorderClass} ${cardHoverClass} ${isLive ? "animate-[pulse_2.2s_ease-in-out_infinite]" : ""}`}
    >
      <div className={`rounded-t-3xl border-b px-4 py-2 text-sm text-white/85 ${isLive ? "border-rose-400/30 bg-rose-500/10" : "border-white/10 bg-black/35"}`}>
        <div className="flex items-center justify-between gap-3">
          <span>{event.date.split("-").reverse().join("/")} {event.time || ""}</span>
          {isLive && (
            <span className="inline-flex items-center gap-2 rounded-full border border-rose-400/50 bg-rose-500/12 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-rose-200">
              <span className="h-2 w-2 animate-pulse rounded-full bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.9)]" />
              Live
            </span>
          )}
        </div>
      </div>

      <div className="space-y-5 px-4 py-4">
        {isLive && hasLiveScore ? (
          <div className="space-y-2 rounded-2xl border border-rose-400/30 bg-gradient-to-br from-rose-500/10 via-white/[0.03] to-transparent p-3 shadow-[0_0_24px_rgba(251,113,133,0.12)]">
            <div className="flex items-center justify-between text-[11px] text-rose-200/85">
              <span className="uppercase tracking-[0.18em]">Marcador en vivo</span>
              <span className="normal-case text-white/70">{liveClock}</span>
            </div>
            <CompactScoreRow sportKey={sportKey} abbr={teams.awayTeam} score={event.away_score} />
            <CompactScoreRow sportKey={sportKey} abbr={teams.homeTeam} score={event.home_score} />
          </div>
        ) : (
          <div className="min-h-[92px] text-xl leading-tight">
            <TeamRow sportKey={sportKey} abbr={teams.awayTeam} />
            <div className="mt-4">
              <TeamRow sportKey={sportKey} abbr={teams.homeTeam} />
            </div>
          </div>
        )}

        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-sm text-white/70">Pick principal</p>
            <p className="mt-1 text-base font-semibold">{fullGamePick}</p>
            <p className="mt-1 text-xs text-white/70">Cuota ML: {mainOdds || "N/A"}</p>
          </div>

          <div className="rounded-xl border border-white/40 px-3 py-2 text-xs text-white/90">
            {event.full_game_confidence}%
          </div>
        </div>

        <div className="grid gap-2 text-xs sm:grid-cols-2">
          <div className={`rounded-xl border bg-white/5 px-3 py-2 text-white/80 ${spreadResultBorderClass}`}>
            <p className="text-white/60">Handicap</p>
            <p className="mt-1">Pick: {spreadPredictionLabel}</p>
            <p className="mt-1">Linea: {spreadLineDisplay}</p>
            <p className="mt-1">Cuota: {spreadOddsDisplay}</p>
            {hasResult && spreadHit !== undefined && spreadHit !== null && (
              <p className="mt-1 font-semibold text-white/90">
                Resultado: {spreadHit === true ? "ACIERTO" : "FALLO"}
              </p>
            )}
          </div>
          <div className={`rounded-xl border bg-white/5 px-3 py-2 text-white/80 ${totalResultBorderClass}`}>
            <p className="text-white/60">Over/Under</p>
            <p className="mt-1">Pick: {totalPickDisplay}</p>
            <p className="mt-1">Linea: {totalLineDisplay}</p>
            <p className="mt-1">Cuota: {totalOddsDisplay}</p>
            {hasResult && totalHit !== undefined && totalHit !== null && (
              <p className="mt-1 font-semibold text-white/90">
                Resultado: {totalHit === true ? "ACIERTO" : "FALLO"}
              </p>
            )}
          </div>
        </div>

        {secondaryPick && (
          <div className={`rounded-xl border bg-black/15 px-3 py-2 ${secondaryResultBorderClass}`}>
            <p className="text-xs text-white/60">{secondaryLabel}</p>
            <p className="mt-1 text-sm font-semibold text-white">{secondaryPick}</p>
            {secondaryConfidence !== undefined && secondaryConfidence !== null && (
              <p className="mt-1 text-xs text-white/70">Confianza: {secondaryConfidence}%</p>
            )}
            <p className="mt-1 text-xs text-white/70">
              {secondaryLabel === "Primer Cuarto" ? `Cuota: ${q1BetActionLabel}` : `Cuota: ${secondaryOdds || "N/A"}`}
            </p>
            {hasResult && secondaryHit !== undefined && secondaryHit !== null && (
              <p className="mt-1 text-xs font-semibold text-white/85">
                Resultado: {secondaryHit === true ? "ACIERTO" : "FALLO"}
              </p>
            )}
          </div>
        )}

        {firstHalfCard && (
          <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(firstHalfCard.hit)}`}>
            <p className="text-xs text-white/60">Primera Mitad</p>
            <p className="mt-1 text-sm font-semibold text-white">{firstHalfCard.pick}</p>
            {firstHalfCard.confidence !== undefined && firstHalfCard.confidence !== null && (
              <p className="mt-1 text-xs text-white/70">Confianza: {firstHalfCard.confidence}%</p>
            )}
            <p className="mt-1 text-xs text-white/70">Accion: {firstHalfCard.action}</p>
            {hasResult && firstHalfCard.hit !== undefined && firstHalfCard.hit !== null && (
              <p className="mt-1 text-xs font-semibold text-white/85">
                Resultado: {firstHalfCard.hit === true ? "ACIERTO" : "FALLO"}
              </p>
            )}
          </div>
        )}

        {cornersPick && (
          <div className={`rounded-xl border bg-cyan-500/10 px-3 py-2 ${cornersResultBorderClass}`}>
            <p className="text-xs text-cyan-100">Corners O/U</p>
            <p className="mt-1 text-sm font-semibold text-cyan-50">{cornersPick}</p>
            <p className="mt-1 text-xs text-cyan-100/80">
              Confianza: {cornersConfidence ?? "-"}% · Acción: {cornersAction || "N/A"}
            </p>
            <p className="mt-1 text-xs text-cyan-100/80">Cuota: {cornersOdds || "N/A"}</p>
            {hasResult && cornersHit !== undefined && cornersHit !== null && (
              <p className="mt-1 text-xs font-semibold text-cyan-50">
                Resultado: {cornersHit === true ? "ACIERTO" : "FALLO"}
              </p>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-3">
          <span className={`rounded-full border px-3 py-1 text-xs ${tierClasses(eventTier)}`}>
            {tierLabel(eventTier)}
          </span>
          <span className="text-xs text-white/55">Abrir detalle</span>
        </div>

        {hasResult && (
          <div className={`rounded-xl border bg-black/20 px-3 py-2 text-xs ${resultBorderClass}`}>
            <p className="text-white/75">Final: {event.final_score_text}</p>
            <p className="mt-1 font-semibold text-white">
              Resultado pick: {gameHit === true ? "ACIERTO" : gameHit === false ? "FALLO" : "N/A"}
            </p>
          </div>
        )}
      </div>
    </button>
  );
}
