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

export function resolveMarketTier(event, market) {
  const normalizedMarket = String(market || "").trim().toLowerCase();
  const tierKeysByMarket = {
    full_game: ["full_game_tier", "moneyline_tier"],
    spread: ["spread_tier"],
    total: ["total_tier"],
    q1: ["q1_tier"],
    h1: ["h1_tier"],
    btts: ["btts_tier"],
    f5: ["f5_tier"],
    home_over: ["home_over_tier"],
    corners: ["corners_tier"],
  };
  const scoreKeysByMarket = {
    full_game: [
      "full_game_recommended_score",
      "moneyline_recommended_score",
      "recommended_score",
      "full_game_confidence",
      "recommended_confidence",
    ],
    spread: ["spread_recommended_score", "spread_confidence"],
    total: ["total_recommended_score", "total_confidence"],
    q1: ["q1_recommended_score", "q1_confidence"],
    h1: ["h1_recommended_score", "h1_confidence"],
    btts: ["btts_recommended_score", "btts_confidence"],
    f5: ["extra_f5_recommended_score", "extra_f5_confidence"],
    home_over: ["home_over_recommended_score", "home_over_confidence"],
    corners: ["corners_recommended_score", "corners_confidence"],
  };

  for (const key of tierKeysByMarket[normalizedMarket] || []) {
    const normalized = normalizeTier(event?.[key]);
    if (normalized) return normalized;
  }

  for (const key of scoreKeysByMarket[normalizedMarket] || []) {
    const value = event?.[key];
    if (value !== undefined && value !== null && value !== "") {
      return inferTierFromScore(value);
    }
  }

  return "NORMAL";
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
