function normalizeTier(tier) {
  const t = String(tier || "").trim().toUpperCase();
  if (t === "ELITE" || t === "PREMIUM" || t === "STRONG" || t === "NORMAL") {
    return t;
  }
  if (t === "PASS") {
    return "NORMAL";
  }
  return "";
}

export function inferTierFromScore(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return "NORMAL";
  if (n >= 80) return "ELITE";
  if (n >= 65) return "PREMIUM";
  if (n >= 55) return "STRONG";
  return "NORMAL";
}

export function resolveEventTier(event) {
  const normalized = normalizeTier(event?.full_game_tier);
  if (normalized) return normalized;

  const score =
    event?.full_game_recommended_score ??
    event?.recommended_score ??
    event?.full_game_confidence ??
    event?.recommended_confidence;

  return inferTierFromScore(score);
}

export function tierClasses(tier) {
  switch (normalizeTier(tier)) {
    case "ELITE":
      return "bg-emerald-500/15 text-emerald-300 border-emerald-400/30";
    case "PREMIUM":
      return "bg-red-500/15 text-red-300 border-red-400/30";
    case "STRONG":
      return "bg-blue-500/15 text-blue-300 border-blue-400/30";
    case "NORMAL":
      return "bg-yellow-500/15 text-yellow-300 border-yellow-400/30";
    default:
      return "bg-white/10 text-white/65 border-white/15";
  }
}

export function tierLabel(tier) {
  switch (normalizeTier(tier)) {
    case "ELITE":
      return "ELITE PICK";
    case "PREMIUM":
      return "PREMIUM PICK";
    case "STRONG":
      return "STRONG PICK";
    case "NORMAL":
      return "NORMAL PICK";
    default:
      return "NORMAL PICK";
  }
}