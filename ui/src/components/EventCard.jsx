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

export default function EventCard({ event, onOpen, sportKey }) {
  const { socialMode } = useAppSettings();
  const teams = resolveEventTeams(event);
  const eventTier = resolveEventTier(event);
  const { hasResult, isLive } = resolveResultState(event);
  const hasLiveScore =
    (event.home_score !== undefined && event.home_score !== null) &&
    (event.away_score !== undefined && event.away_score !== null);
  const gameHit = event.full_game_hit;
  const firstHalfCard = resolveFirstHalfCard(event, sportKey, teams);
  const rawFullGamePick = expandTeamCodeInText(sportKey, resolveSidePick(event.full_game_pick, teams));
  const fullGamePick = rawFullGamePick || (sportKey === "ncaa_baseball" ? "Sin linea disponible" : "Pendiente");
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
  const cornersOdds = resolveOddsValue(event, ODDS_KEYS.corners);
  const spreadPickRaw = String(event.spread_pick || "").trim();
  const spreadSideTeam = resolveSpreadSideTeam(event, teams, fullGamePick);
  const spreadPickDisplayRaw = String(event.spread_pick_display || "").trim();
  const spreadPickFromEvent = spreadPickDisplayRaw
    ? expandTeamCodeInText(sportKey, resolveSidePick(spreadPickDisplayRaw, teams))
    : (expandTeamCodeInText(sportKey, resolveSidePick(spreadPickRaw, teams)) || spreadPickRaw);
  const spreadPick = !isPendingPick(spreadPickRaw) || spreadPickDisplayRaw
    ? spreadPickFromEvent
    : ((spreadLineDisplay !== "Por definir" || hasMlSpreadMarket) ? (expandTeamCodeInText(sportKey, spreadSideTeam) || fullGamePick) : null);
  const signedSpreadLineDisplay = resolveSignedSpreadLine(event, teams, spreadSideTeam) || spreadLineDisplay;
  const spreadHit = toHitValue(event.correct_spread);
  const totalPickRaw = event.total_recommended_pick || event.total_pick;
  const totalDirection = normalizeTotalDirection(totalPickRaw);
  const totalLineValue = extractNumericFromText(totalLineDisplay);
  const totalHit = resolveTotalHit(event, totalDirection, totalLineValue);
  const totalPickDisplay = buildTotalPickDisplay(totalPickRaw, totalDirection, totalLineDisplay);
  const spreadPredictionLabel = spreadPick || fullGamePick || "N/A";

  const q1PickRaw = String(event.q1_pick || "").trim();
  const q1Pick = !isPendingPick(q1PickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(q1PickRaw, teams)) || q1PickRaw)
    : null;
  const q1Hit = toHitValue(event.q1_hit);
  const q1Confidence = event.q1_confidence;
  const q1Action = normalizeBetAction(event.q1_action);
  const q1Odds = resolveOddsValue(event, ODDS_KEYS.q1);
  const spreadTier = resolveMarketTier(event, "spread");
  const totalTier = resolveMarketTier(event, "total");
  const q1Tier = resolveMarketTier(event, "q1");

  const homeOverPickRaw = String(event.home_over_pick || "").trim();
  const homeOverPick = !isPendingPick(homeOverPickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(homeOverPickRaw, teams)) || homeOverPickRaw)
    : null;
  const homeOverHit = toHitValue(event.correct_home_over ?? event.home_over_correct);
  const homeOverConfidence = event.home_over_confidence;
  const homeOverOdds = resolveOddsValue(event, ["closing_home_over_odds", "home_over_odds"]);
  const homeOverTier = resolveMarketTier(event, "home_over");

  const bttsPickRaw = String(event.btts_recommended_pick || event.btts_pick || "").trim();
  const bttsPick = !isPendingPick(bttsPickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(bttsPickRaw, teams)) || bttsPickRaw)
    : null;
  const bttsHit = toHitValue(event.correct_btts_adjusted ?? event.correct_btts ?? event.correct_btts_base);
  const bttsConfidence = event.btts_confidence;
  const bttsOdds = resolveOddsValue(event, ODDS_KEYS.btts);
  const bttsTier = resolveMarketTier(event, "btts");

  const f5PickRaw = String(event.assists_pick || event.f5_pick || "").trim();
  const f5Pick = !isPendingPick(f5PickRaw)
    ? (expandTeamCodeInText(sportKey, resolveSidePick(f5PickRaw, teams)) || f5PickRaw)
    : null;
  const f5Hit = toHitValue(event.correct_f5 ?? event.correct_home_win_f5);
  const f5Confidence = event.extra_f5_confidence;
  const f5Odds = resolveOddsValue(event, ODDS_KEYS.f5);
  const f5Tier = resolveMarketTier(event, "f5");

  const cornersPick = event.corners_pick;
  const cornersConfidence = event.corners_confidence;
  const cornersAction = event.corners_action;
  const cornersHit = toHitValue(event.correct_corners_adjusted ?? event.correct_corners_base ?? event.correct_corners);
  const cornersTier = resolveMarketTier(event, "corners");
  const firstHalfTier = resolveMarketTier(event, "h1");
  const resultBorderClass =
    gameHit === true
      ? "border-emerald-400/70"
      : gameHit === false
        ? "border-rose-400/70"
        : "border-white/15";
  const cornersResultBorderClass =
    cornersHit === true
      ? "border-emerald-400/70"
      : cornersHit === false
        ? "border-rose-400/70"
        : "border-cyan-400/40";
  const spreadResultBorderClass = marketBorderClass(spreadHit);
  const totalResultBorderClass = marketBorderClass(totalHit);
  const cardBorderClass = isLive
    ? "border-rose-400/70 shadow-[0_0_30px_rgba(251,113,133,0.12)]"
    : gameHit === true
      ? "border-emerald-400/70 shadow-[0_0_28px_rgba(52,211,153,0.10)]"
      : gameHit === false
        ? "border-rose-400/70 shadow-[0_0_28px_rgba(251,113,133,0.10)]"
        : "border-white/10";
  const cardHoverClass = isLive
    ? "hover:border-rose-300/80"
    : gameHit === true
      ? "hover:border-emerald-300/80"
      : gameHit === false
        ? "hover:border-rose-300/80"
        : "hover:border-amber-300/70";
  const liveClock = formatLiveClock(event);
  const mainPickLabel = socialMode ? "Proyeccion principal" : "Pick principal";
  const mlLabel = socialMode ? "Referencia ML" : "Cuota ML";
  const spreadLabel = socialMode ? "Proyeccion de margen" : "Handicap";
  const totalLabel = socialMode ? "Proyeccion total" : "Over/Under";
  const q1Label = socialMode ? "Proyeccion de arranque" : "Primer Cuarto";
  const homeOverLabel = socialMode ? "Proyeccion local" : "Goles Local O/U 2.5";
  const bttsLabel = socialMode ? "Anotan ambos" : "BTTS";
  const f5Label = socialMode ? "Proyeccion F5" : "F5 / Props";
  const h1Label = socialMode ? "Proyeccion primera mitad" : "Primera Mitad";
  const cornersLabel = socialMode ? "Proyeccion de corners" : "Corners O/U";
  const lineLabel = socialMode ? "Referencia" : "Linea";
  const oddsLabel = socialMode ? "Valor modelo" : "Cuota";
  const resultLabel = socialMode ? "Resultado del modelo" : "Resultado";
  const finalLabel = socialMode ? "Cierre" : "Final";
  const finalPickLabel = socialMode ? "Resultado del modelo" : "Resultado pick";

  return (
    <button
      onClick={() => onOpen(event)}
      className={`group relative overflow-visible rounded-[30px] border bg-[linear-gradient(180deg,rgba(27,31,40,0.96),rgba(20,23,31,0.98))] p-0 text-left transition hover:-translate-y-1 hover:shadow-[0_22px_50px_rgba(0,0,0,0.28)] lg:hover:z-20 ${cardBorderClass} ${cardHoverClass} ${isLive ? "animate-[pulse_2.2s_ease-in-out_infinite]" : ""}`}
    >
      <div className={`rounded-t-[30px] border-b px-4 py-2 text-sm text-white/85 ${
        isLive
          ? "border-rose-400/30 bg-rose-500/10"
          : gameHit === true
            ? "border-emerald-400/30 bg-emerald-500/10"
            : gameHit === false
              ? "border-rose-400/30 bg-rose-500/10"
              : "border-white/10 bg-black/24"
      }`}>
        <div className="flex items-center justify-between gap-3">
          <span>{event.date.split("-").reverse().join("/")} {event.time || ""}</span>
          {isLive && (
            <span className="inline-flex items-center gap-2 rounded-full border border-rose-400/50 bg-rose-500/12 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-rose-200">
              <span className="h-2 w-2 animate-pulse rounded-full bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.9)]" />
              Live
            </span>
          )}
          {!isLive && hasResult && (
            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/35 bg-emerald-500/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-100">
              Final
            </span>
          )}
        </div>
      </div>

      <div className="relative space-y-5 px-4 py-4">
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
            <p className="text-sm text-white/70">{mainPickLabel}</p>
            <p className="mt-1 text-base font-semibold">{fullGamePick}</p>
            <p className="mt-1 text-xs text-white/70">{mlLabel}: {mainOdds || (sportKey === "ncaa_baseball" ? "Sin linea" : "N/A")}</p>
          </div>

          <div className="rounded-2xl border border-white/18 bg-white/[0.03] px-3 py-2 text-xs text-white/90">
            {event.full_game_confidence}%
          </div>
        </div>

        <div className="flex items-center justify-between gap-3">
          <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${tierClasses(eventTier)}`}>
            {hasResult && gameHit !== null && gameHit !== undefined && (
              <span
                className={`h-2 w-2 rounded-full animate-[pulse_1.8s_ease-in-out_infinite] ${
                  gameHit === true ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" : "bg-rose-400 shadow-[0_0_10px_rgba(251,113,133,0.7)]"
                }`}
              />
            )}
            {tierLabel(eventTier)}
          </span>
          <span className="text-xs text-white/55 transition-colors duration-300 lg:group-hover:text-white/75">
            Abrir detalle
          </span>
        </div>

        <div className="max-h-[2400px] overflow-hidden opacity-100 transition-all duration-300 ease-out lg:pointer-events-none lg:absolute lg:left-0 lg:right-0 lg:top-[calc(100%-18px)] lg:z-10 lg:max-h-none lg:overflow-visible lg:rounded-[2rem] lg:border lg:border-white/12 lg:bg-[#171a21] lg:px-4 lg:pb-4 lg:pt-8 lg:opacity-0 lg:shadow-[0_24px_48px_rgba(0,0,0,0.38)] lg:translate-y-1 lg:group-hover:opacity-100 lg:group-hover:translate-y-0">
          <div className="space-y-5 pt-1 lg:pt-0">
            <div className="hidden lg:block absolute left-6 right-6 top-0 h-5 -translate-y-[70%] rounded-t-[1.5rem] bg-[#171a21] border-x border-t border-white/12" />
            <div className="grid gap-2 text-xs sm:grid-cols-2">
              <div className={`rounded-xl border bg-white/5 px-3 py-2 text-white/80 ${spreadResultBorderClass}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-white/60">{spreadLabel}</p>
                  <MarketTierBadge tier={spreadTier} />
                </div>
                <p className="mt-1">Pick: {spreadPredictionLabel}</p>
                <p className="mt-1">{lineLabel}: {signedSpreadLineDisplay}</p>
                <p className="mt-1">{oddsLabel}: {spreadOddsDisplay}</p>
                {hasResult && spreadHit !== undefined && spreadHit !== null && (
                  <p className="mt-1 font-semibold text-white/90">
                    {resultLabel}: {outcomeLabel(spreadHit, socialMode)}
                  </p>
                )}
              </div>
              <div className={`rounded-xl border bg-white/5 px-3 py-2 text-white/80 ${totalResultBorderClass}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-white/60">{totalLabel}</p>
                  <MarketTierBadge tier={totalTier} />
                </div>
                <p className="mt-1">Pick: {totalPickDisplay}</p>
                <p className="mt-1">{lineLabel}: {totalLineDisplay}</p>
                <p className="mt-1">{oddsLabel}: {totalOddsDisplay}</p>
                {hasResult && totalHit !== undefined && totalHit !== null && (
                  <p className="mt-1 font-semibold text-white/90">
                    {resultLabel}: {outcomeLabel(totalHit, socialMode)}
                  </p>
                )}
              </div>
            </div>

            {q1Pick && (
              <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(q1Hit)}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-white/60">{q1Label}</p>
                  <MarketTierBadge tier={q1Tier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-white">{q1Pick}</p>
                {q1Confidence !== undefined && q1Confidence !== null && (
                  <p className="mt-1 text-xs text-white/70">Confianza: {q1Confidence}%</p>
                )}
                <p className="mt-1 text-xs text-white/70">Accion: {q1Action}</p>
                <p className="mt-1 text-xs text-white/70">{oddsLabel}: {q1Odds || "N/A"}</p>
                {hasResult && q1Hit !== undefined && q1Hit !== null && (
                  <p className="mt-1 text-xs font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(q1Hit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {homeOverPick && (
              <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(homeOverHit)}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-white/60">{homeOverLabel}</p>
                  <MarketTierBadge tier={homeOverTier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-white">{homeOverPick}</p>
                {homeOverConfidence !== undefined && homeOverConfidence !== null && (
                  <p className="mt-1 text-xs text-white/70">Confianza: {homeOverConfidence}%</p>
                )}
                <p className="mt-1 text-xs text-white/70">{oddsLabel}: {homeOverOdds || "N/A"}</p>
                {hasResult && homeOverHit !== undefined && homeOverHit !== null && (
                  <p className="mt-1 text-xs font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(homeOverHit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {bttsPick && (
              <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(bttsHit)}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-white/60">{bttsLabel}</p>
                  <MarketTierBadge tier={bttsTier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-white">{bttsPick}</p>
                {bttsConfidence !== undefined && bttsConfidence !== null && (
                  <p className="mt-1 text-xs text-white/70">Confianza: {bttsConfidence}%</p>
                )}
                <p className="mt-1 text-xs text-white/70">{oddsLabel}: {bttsOdds || "N/A"}</p>
                {hasResult && bttsHit !== undefined && bttsHit !== null && (
                  <p className="mt-1 text-xs font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(bttsHit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {f5Pick && (
              <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(f5Hit)}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-white/60">{f5Label}</p>
                  <MarketTierBadge tier={f5Tier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-white">{f5Pick}</p>
                {f5Confidence !== undefined && f5Confidence !== null && (
                  <p className="mt-1 text-xs text-white/70">Confianza: {f5Confidence}%</p>
                )}
                <p className="mt-1 text-xs text-white/70">{oddsLabel}: {f5Odds || "N/A"}</p>
                {hasResult && f5Hit !== undefined && f5Hit !== null && (
                  <p className="mt-1 text-xs font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(f5Hit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {firstHalfCard && (
              <div className={`rounded-xl border bg-black/15 px-3 py-2 ${marketBorderClass(firstHalfCard.hit)}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-white/60">{h1Label}</p>
                  <MarketTierBadge tier={firstHalfTier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-white">{firstHalfCard.pick}</p>
                {firstHalfCard.confidence !== undefined && firstHalfCard.confidence !== null && (
                  <p className="mt-1 text-xs text-white/70">Confianza: {firstHalfCard.confidence}%</p>
                )}
                <p className="mt-1 text-xs text-white/70">Accion: {firstHalfCard.action}</p>
                {hasResult && firstHalfCard.hit !== undefined && firstHalfCard.hit !== null && (
                  <p className="mt-1 text-xs font-semibold text-white/85">
                    {resultLabel}: {outcomeLabel(firstHalfCard.hit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {cornersPick && (
              <div className={`rounded-xl border bg-cyan-500/10 px-3 py-2 ${cornersResultBorderClass}`}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-cyan-100">{cornersLabel}</p>
                  <MarketTierBadge tier={cornersTier} />
                </div>
                <p className="mt-1 text-sm font-semibold text-cyan-50">{cornersPick}</p>
                <p className="mt-1 text-xs text-cyan-100/80">
                  Confianza: {cornersConfidence ?? "-"}% | Accion: {cornersAction || "N/A"}
                </p>
                <p className="mt-1 text-xs text-cyan-100/80">{oddsLabel}: {cornersOdds || "N/A"}</p>
                {hasResult && cornersHit !== undefined && cornersHit !== null && (
                  <p className="mt-1 text-xs font-semibold text-cyan-50">
                    {resultLabel}: {outcomeLabel(cornersHit, socialMode)}
                  </p>
                )}
              </div>
            )}

            {hasResult && (
              <div className={`rounded-xl border bg-black/20 px-3 py-2 text-xs ${resultBorderClass}`}>
                <p className="text-white/75">{finalLabel}: {event.final_score_text}</p>
                <p className="mt-1 font-semibold text-white">
                  {finalPickLabel}: {outcomeLabel(gameHit, socialMode)}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </button>
  );
}
