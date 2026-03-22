export const API_BASE = "/api";
const API_FALLBACK_BASE = "http://127.0.0.1:8010/api";

const ODDS_QUALITY_KEYS = [
  "closing_moneyline_odds",
  "home_moneyline_odds",
  "away_moneyline_odds",
  "closing_spread_odds",
  "closing_total_odds",
  "closing_spread_line",
  "closing_total_line",
];

async function fetchWithFallback(path, options = undefined) {
  try {
    const primary = await fetch(`${API_BASE}${path}`, options);
    if (primary.ok) return primary;
  } catch {
    // Try local fallback backend if primary is unavailable.
  }
  return fetch(`${API_FALLBACK_BASE}${path}`, options);
}

function normalizeEventsPayload(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.games)) return payload.games;
  return [];
}

function hasValidMetric(value) {
  const n = Number(value);
  return Number.isFinite(n) && n !== 0;
}

function scorePayloadQuality(events) {
  let score = 0;
  for (const event of events) {
    if (!event || typeof event !== "object") continue;
    for (const key of ODDS_QUALITY_KEYS) {
      if (hasValidMetric(event[key])) score += 1;
    }
  }
  return score;
}

async function fetchPredictionsWithSmartFallback(path) {
  const attempts = [];

  try {
    const res = await fetch(`${API_BASE}${path}`);
    if (res.ok) {
      const payload = await res.json();
      const events = normalizeEventsPayload(payload);
      attempts.push({ events, quality: scorePayloadQuality(events) });
    }
  } catch {
    // Ignore and continue to fallback.
  }

  try {
    const res = await fetch(`${API_FALLBACK_BASE}${path}`);
    if (res.ok) {
      const payload = await res.json();
      const events = normalizeEventsPayload(payload);
      attempts.push({ events, quality: scorePayloadQuality(events) });
    }
  } catch {
    // Both endpoints may fail; handled below.
  }

  if (attempts.length === 0) {
    throw new Error("No se pudieron cargar las predicciones de hoy.");
  }

  attempts.sort((a, b) => {
    if (b.quality !== a.quality) return b.quality - a.quality;
    return b.events.length - a.events.length;
  });
  return attempts[0].events;
}

export async function fetchTodayPredictions(sport) {
  return await fetchPredictionsWithSmartFallback(`/${sport}/predictions/today`);
}

export async function fetchPredictionsByDate(sport, dateStr) {
  try {
    return await fetchPredictionsWithSmartFallback(`/${sport}/predictions/${dateStr}`);
  } catch {
    throw new Error("No hay predicciones guardadas para esa fecha.");
  }
}

export async function fetchAvailableDates(sport) {
  const res = await fetchWithFallback(`/${sport}/available-dates`);
  if (!res.ok) {
    throw new Error("No se pudieron cargar las fechas disponibles.");
  }
  return await res.json();
}

export async function fetchInsightsSummary() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000);
  const res = await fetch(`${API_BASE}/insights/summary`, { signal: controller.signal }).finally(() => {
    clearTimeout(timeoutId);
  });
  if (!res.ok) {
    throw new Error("No se pudieron cargar los insights del sistema.");
  }
  return await res.json();
}

export async function fetchWeekdayScoringInsights() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000);
  const res = await fetch(`${API_BASE}/insights/weekday-scoring`, { signal: controller.signal }).finally(() => {
    clearTimeout(timeoutId);
  });
  if (!res.ok) {
    throw new Error("No se pudieron cargar los insights de scoring por dia.");
  }
  return await res.json();
}

export async function fetchBestPicksToday(topN = 25, rankingMode = "best_hit_rate") {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000);
  const res = await fetch(
    `${API_BASE}/insights/best-picks/today?top_n=${encodeURIComponent(topN)}&ranking_mode=${encodeURIComponent(rankingMode)}`,
    {
    signal: controller.signal,
    },
  ).finally(() => {
    clearTimeout(timeoutId);
  });

  if (!res.ok) {
    throw new Error("No se pudieron cargar los mejores picks del dia.");
  }
  return await res.json();
}

export async function fetchBestPicksByDate(dateStr, topN = 25, rankingMode = "best_hit_rate") {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000);
  const res = await fetch(
    `${API_BASE}/insights/best-picks/${encodeURIComponent(dateStr)}?top_n=${encodeURIComponent(topN)}&ranking_mode=${encodeURIComponent(rankingMode)}`,
    { signal: controller.signal },
  ).finally(() => {
    clearTimeout(timeoutId);
  });

  if (!res.ok) {
    throw new Error("No se pudieron cargar los mejores picks para esa fecha.");
  }
  return await res.json();
}

export async function fetchBestPicksAvailableDates() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);
  const res = await fetch(`${API_BASE}/insights/best-picks/available-dates`, {
    signal: controller.signal,
  }).finally(() => {
    clearTimeout(timeoutId);
  });

  if (!res.ok) {
    throw new Error("No se pudieron cargar las fechas de best picks.");
  }
  return await res.json();
}

export async function fetchTierPerformanceInsights() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);
  const res = await fetch(`${API_BASE}/insights/tier-performance`, { signal: controller.signal }).finally(() => {
    clearTimeout(timeoutId);
  });

  if (!res.ok) {
    throw new Error("No se pudieron cargar los insights por tier.");
  }
  return await res.json();
}