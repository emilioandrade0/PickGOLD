import { getTeamLogoUrl } from "../utils/teamLogos.js";
import { resolveEventTeams } from "../utils/teams.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

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

function resolveDisplayedScores(event) {
  const awayScore = toFiniteNumber(event?.away_score);
  const homeScore = toFiniteNumber(event?.home_score);
  if (awayScore !== null && homeScore !== null) {
    return { awayScore, homeScore };
  }

  const text = String(event?.final_score_text || "").trim();
  const numbers = [...text.matchAll(/(\d+)/g)].map((m) => Number(m[1]));
  if (numbers.length >= 2) {
    return {
      awayScore: numbers[numbers.length - 2],
      homeScore: numbers[numbers.length - 1],
    };
  }

  return { awayScore: null, homeScore: null };
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

function formatEventDate(dateValue) {
  if (!dateValue) return "Sin fecha";
  const date = new Date(`${dateValue}T00:00:00`);
  if (Number.isNaN(date.getTime())) return String(dateValue);
  return date.toLocaleDateString("es-MX");
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

export default function ClassicLightEventCard({ event, sportKey, onOpen, active = false }) {
  const teams = resolveEventTeams(event);
  const awayName = getTeamDisplayName(sportKey, teams.awayTeam);
  const homeName = getTeamDisplayName(sportKey, teams.homeTeam);
  const { hasResult, isLive } = resolveResultState(event);
  const gameHit = toHitValue(event?.full_game_hit);
  const { awayScore, homeScore } = resolveDisplayedScores(event);
  const showFinalScore = hasResult && !isLive && awayScore !== null && homeScore !== null;
  const statusLabel = isLive ? "LIVE" : hasResult ? "FINAL" : "PREVIO";
  const resultBorderClass =
    gameHit === true
      ? "border-[2.6px] border-[#64a981]"
      : gameHit === false
        ? "border-[2.6px] border-[#cf727f]"
        : "border-[1.6px] border-[#b8bac0]";
  const activeRingClass = active ? "ring-2 ring-[#4f525a]/30" : "";

  return (
    <button
      type="button"
      onClick={() => onOpen(event)}
      className={`classic-light-surface classic-light-event-card w-full rounded-[22px] border bg-[#e4e4e8] px-4 py-3 text-left shadow-[0_8px_20px_rgba(0,0,0,0.16)] transition hover:-translate-y-0.5 hover:shadow-[0_14px_28px_rgba(0,0,0,0.22)] ${resultBorderClass} ${activeRingClass}`}
    >
      <div className="flex items-center justify-between text-[11px] font-semibold text-[#454854]">
        <span>{formatEventDate(event?.date)} {event?.time || ""}</span>
        <span className={`rounded-full border px-2 py-0.5 text-[10px] tracking-[0.08em] ${
          isLive
            ? "border-[#ef6e7f] bg-[#f9d8de] text-[#8f1d2f]"
            : hasResult
              ? "border-[#848892] bg-[#f4f4f7] text-[#3f4350]"
              : "border-[#a7aab2] bg-[#ededf2] text-[#595d69]"
        }`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
        <div className="flex justify-center">
          <TeamLogo sportKey={sportKey} teamCode={teams.awayTeam} teamName={awayName} />
        </div>
        <div className="text-center text-sm font-semibold uppercase tracking-[0.14em] text-[#3f424d]">
          {showFinalScore ? (
            <div>
              <div className="flex items-end justify-center gap-3 text-[#1f232c]">
                <span className="text-5xl font-black leading-none [font-family:var(--classic-display-font)]">{awayScore}</span>
                <span className="mb-1 text-xl font-semibold uppercase tracking-[0.16em] text-[#495061]">vs</span>
                <span className="text-5xl font-black leading-none [font-family:var(--classic-display-font)]">{homeScore}</span>
              </div>
              <p className="mt-1 text-lg font-black leading-none text-[#20242d] [font-family:var(--classic-display-font)]">FINAL</p>
            </div>
          ) : (
            <span>vs</span>
          )}
        </div>
        <div className="flex justify-center">
          <TeamLogo sportKey={sportKey} teamCode={teams.homeTeam} teamName={homeName} />
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-center">
        <p className="text-[13px] font-black uppercase tracking-[0.02em] text-[#20232a]">{awayName}</p>
        <p className="text-[13px] font-black uppercase tracking-[0.02em] text-[#20232a]">{homeName}</p>
      </div>
    </button>
  );
}
