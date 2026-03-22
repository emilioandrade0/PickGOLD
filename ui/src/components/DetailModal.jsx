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
    const q1Label = sportKey === "nhl"
      ? (event.q1_market || "1P O/U 1.5")
      : (["mlb", "kbo", "ncaa_baseball"].includes(sportKey) ? "YRFI / NRFI" : "Q1");
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

export default function DetailModal({ event, onClose, sportKey }) {
  if (!event) return null;
  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const eventTier = resolveEventTier(event);

  const hasResult = event.result_available === true;
  const isLive = !hasResult && String(event.status_state || "").toLowerCase() === "in";
  const hasLiveScore =
    (event.home_score !== undefined && event.home_score !== null) &&
    (event.away_score !== undefined && event.away_score !== null);

  const secondaryMarket = resolveSecondaryMarket(event, sportKey, teams);
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
  const cornersOdds = resolveOddsValue(event, ODDS_KEYS.corners);
  const secondaryResultBorderClass = marketBorderClass(secondaryHit);
  const resultBorderClass =
    event.full_game_hit === true
      ? "border-emerald-400/70"
      : event.full_game_hit === false
        ? "border-rose-400/70"
        : "border-white/15";

  const propLabel = String(event.assists_pick || "").toUpperCase().includes("F5") ? "Pick F5" : "Prop destacada";
  const propPick = event.assists_pick || "N/A";
  const propConfidence = event.extra_f5_confidence ?? event.btts_confidence;
  const hasCorners = Boolean(event.corners_pick);
  const spreadHit = toHitValue(event.correct_spread);
  const totalHit = toHitValue(event.correct_total_adjusted ?? event.correct_total);
  const propHit = toHitValue(event.correct_f5 ?? event.correct_home_win_f5);
  const cornersHit = toHitValue(event.correct_corners_adjusted ?? event.correct_corners_base ?? event.correct_corners);

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
            {isLive && hasLiveScore && (
              <span className="rounded-full border border-sky-400/60 bg-sky-500/10 px-4 py-2 text-sm text-sky-100">
                En vivo: {awayName} {event.away_score} - {homeName} {event.home_score}
              </span>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-black/15 p-5">
              <p className="text-sm text-white/50">Ganador del partido</p>
              <p className="mt-1 text-2xl font-semibold">{fullGamePick}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">Confianza</p>
                  <p className="mt-1 font-semibold">{event.full_game_confidence}%</p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">Cuota ML</p>
                  <p className="mt-1 font-semibold">{mainOdds || "N/A"}</p>
                </div>
              </div>
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 ${secondaryResultBorderClass}`}>
              <p className="text-sm text-white/50">{secondaryLabel}</p>
              <p className="mt-1 text-2xl font-semibold">{secondaryPick}</p>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">Confianza</p>
                  <p className="mt-1 font-semibold">
                    {secondaryConfidence === "-" ? "-" : `${secondaryConfidence}%`}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">Cuota</p>
                  <p className="mt-1 font-semibold">{secondaryOdds || "N/A"}</p>
                </div>
              </div>

              <p className="mt-2 text-xs text-white/60">Acción: {secondaryAction}</p>

              {hasResult && secondaryHit !== undefined && secondaryHit !== null && (
                <p className="mt-3 text-sm font-semibold text-white/85">
                  Resultado: {secondaryHit === true ? "ACIERTO" : "FALLO"}
                </p>
              )}
            </div>
          </div>

          {hasResult && (
            <div className={`rounded-2xl border bg-black/20 p-5 ${resultBorderClass}`}>
              <p className="text-sm text-white/50">Resultado real del juego</p>
              <p className="mt-1 text-lg font-semibold">{event.final_score_text}</p>

              <div className="mt-4 grid gap-3 md:grid-cols-2 text-sm">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">Full Game</p>
                  <p className="mt-1 font-semibold">
                    {event.full_game_hit === true ? "ACIERTO" : event.full_game_hit === false ? "FALLO" : "N/A"}
                  </p>
                </div>

                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-white/50">{secondaryLabel}</p>
                  <p className="mt-1 font-semibold">
                    {secondaryHit === true ? "ACIERTO" : secondaryHit === false ? "FALLO" : "N/A"}
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(spreadHit)}`}>
              <p className="text-sm text-white/50">Mercado principal</p>
              <p className="mt-1 text-lg font-semibold">{event.spread_pick}</p>
              <p className="mt-2 text-sm text-white/65">Mercado: {event.spread_market}</p>
              <p className="mt-1 text-sm text-white/65">Linea: {spreadLineDisplay}</p>
              <p className="mt-1 text-sm text-white/65">Cuota: {spreadOddsDisplay}</p>
              {hasResult && spreadHit !== undefined && spreadHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  Resultado: {spreadHit === true ? "ACIERTO" : "FALLO"}
                </p>
              )}
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 ${marketBorderClass(totalHit)}`}>
              <p className="text-sm text-white/50">Total</p>
              <p className="mt-1 text-lg font-semibold">{event.total_pick}</p>
              <p className="mt-2 text-sm text-white/65">Línea: {totalLineDisplay}</p>
              <p className="mt-1 text-sm text-white/65">Cuota: {totalOddsDisplay}</p>
              {hasResult && totalHit !== undefined && totalHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  Resultado: {totalHit === true ? "ACIERTO" : "FALLO"}
                </p>
              )}
            </div>

            <div className={`rounded-2xl border bg-black/15 p-5 md:col-span-2 xl:col-span-1 ${marketBorderClass(propHit)}`}>
              <p className="text-sm text-white/50">{propLabel}</p>
              <p className="mt-1 text-lg font-semibold leading-snug">{propPick}</p>
              {propConfidence && (
                <p className="mt-2 text-sm text-white/65">Confianza: {propConfidence}%</p>
              )}
              {hasResult && propHit !== undefined && propHit !== null && (
                <p className="mt-2 text-sm font-semibold text-white/85">
                  Resultado: {propHit === true ? "ACIERTO" : "FALLO"}
                </p>
              )}
            </div>

            {hasCorners && (
              <div className={`rounded-2xl border bg-cyan-500/10 p-5 md:col-span-2 xl:col-span-3 ${marketBorderClass(cornersHit, "border-cyan-400/40")}`}>
                <p className="text-sm text-cyan-100">Corners O/U</p>
                <p className="mt-1 text-lg font-semibold text-cyan-50">{event.corners_pick}</p>
                <div className="mt-2 flex flex-wrap gap-4 text-sm text-cyan-100/90">
                  <span>Línea: {event.corners_line ?? 9.5}</span>
                  <span>Cuota: {cornersOdds || "N/A"}</span>
                  <span>Confianza: {event.corners_confidence ?? "-"}%</span>
                  <span>Score: {event.corners_recommended_score ?? "-"}</span>
                  <span>Acción: {event.corners_action || "N/A"}</span>
                </div>
                {hasResult && cornersHit !== undefined && cornersHit !== null && (
                  <p className="mt-2 text-sm font-semibold text-cyan-50">
                    Resultado: {cornersHit === true ? "ACIERTO" : "FALLO"}
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
