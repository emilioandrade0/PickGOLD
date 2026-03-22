const configuredApiBase = (import.meta.env.VITE_API_BASE || "/api").trim();
export const API_BASE = configuredApiBase.replace(/\/$/, "");
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJsonWithRetry(path, {
  timeoutMs = 30000,
  retries = 2,
  retryDelayMs = 2000,
  errorMessage = "No se pudo completar la solicitud.",
} = {}) {
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(`${API_BASE}${path}`, { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`${errorMessage} (status ${res.status})`);
      }
      return await res.json();
    } catch (err) {
      if (attempt < retries) {
        await sleep(retryDelayMs * (attempt + 1));
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }

  throw new Error(`${errorMessage} Intenta de nuevo en unos segundos.`);
}

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
  return await fetchJsonWithRetry("/insights/summary", {
    timeoutMs: 45000,
    retries: 2,
    errorMessage: "No se pudieron cargar los insights del sistema.",
  });
}

export async function fetchWeekdayScoringInsights() {
  return await fetchJsonWithRetry("/insights/weekday-scoring", {
    timeoutMs: 60000,
    retries: 2,
    errorMessage: "No se pudieron cargar los insights de scoring por dia.",
  });
}

export async function fetchBestPicksToday(topN = 25, rankingMode = "best_hit_rate") {
  return await fetchJsonWithRetry(
    `/insights/best-picks/today?top_n=${encodeURIComponent(topN)}&ranking_mode=${encodeURIComponent(rankingMode)}`,
    {
      timeoutMs: 90000,
      retries: 2,
      errorMessage: "No se pudieron cargar los mejores picks del dia.",
    },
  );

}

export async function fetchBestPicksByDate(dateStr, topN = 25, rankingMode = "best_hit_rate") {
  return await fetchJsonWithRetry(
    `/insights/best-picks/${encodeURIComponent(dateStr)}?top_n=${encodeURIComponent(topN)}&ranking_mode=${encodeURIComponent(rankingMode)}`,
    {
      timeoutMs: 90000,
      retries: 2,
      errorMessage: "No se pudieron cargar los mejores picks para esa fecha.",
    },
  });
}

export async function fetchBestPicksAvailableDates() {
  return await fetchJsonWithRetry("/insights/best-picks/available-dates", {
    timeoutMs: 30000,
    retries: 2,
    errorMessage: "No se pudieron cargar las fechas de best picks.",
  });
}

export async function fetchTierPerformanceInsights() {
  return await fetchJsonWithRetry("/insights/tier-performance", {
    timeoutMs: 60000,
    retries: 2,
    errorMessage: "No se pudieron cargar los insights por tier.",
  });
}