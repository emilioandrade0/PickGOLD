import { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import EventCard from "../components/EventCard.jsx";
import DetailModal from "../components/DetailModal.jsx";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { resolveEventTier } from "../utils/picks.js";
import {
  fetchAvailableDates,
  fetchPredictionsByDate,
  fetchTodayPredictions,
} from "../services/api.js";

function toYmdLocal(dateObj) {
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, "0");
  const d = String(dateObj.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

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

function formatDecimalOdds(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return null;
  if (n > 1 && n < 20) return n;
  if (n > 0) return 1 + (n / 100);
  if (n <= -100) return 1 + (100 / Math.abs(n));
  return null;
}

function normalizeSearchText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .trim();
}

export default function LeaguePage({ sportKey, sportLabel }) {
  const { socialMode } = useAppSettings();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [activeEvent, setActiveEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [modalOpen, setModalOpen] = useState(false);
  const [availableDates, setAvailableDates] = useState([]);
  const [eventSearch, setEventSearch] = useState("");

  function findNearestDate(targetDate, availableDates) {
    if (!targetDate || availableDates.length === 0) return "";

    const sorted = [...availableDates].sort();
    const sameOrAfter = sorted.find((d) => d >= targetDate);
    if (sameOrAfter) return sameOrAfter;
    return sorted[sorted.length - 1] || "";
  }

  const refreshAvailableDates = useCallback(async () => {
    try {
      const dates = await fetchAvailableDates(sportKey);
      setAvailableDates(Array.isArray(dates) ? dates : []);
    } catch {
      setAvailableDates([]);
    }
  }, [sportKey]);

  const loadToday = useCallback(async () => {
    try {
      setLoading(true);
      setError("");

      const localToday = toYmdLocal(new Date());

      let data = [];
      let effectiveDate = "";

      // Prefer the user's local date first to avoid UTC server drift showing tomorrow.
      try {
        const localRows = await fetchPredictionsByDate(sportKey, localToday);
        if (Array.isArray(localRows) && localRows.length > 0) {
          data = localRows;
          effectiveDate = localToday;
        }
      } catch {
        // Fallback to /today endpoint if direct date lookup fails.
      }

      if (data.length === 0) {
        data = await fetchTodayPredictions(sportKey);
        effectiveDate = data.length > 0 ? (data[0].date || localToday) : "";
      }

      // If today's slate is empty, jump to the nearest date that actually has games.
      if (data.length === 0) {
        const allDates = await fetchAvailableDates(sportKey);
        const todayStr = localToday;

        const futureOrToday = allDates.filter((d) => d >= todayStr).sort();
        const past = allDates.filter((d) => d < todayStr).sort().reverse();
        const candidates = [...futureOrToday, ...past].slice(0, 21);

        for (const dateStr of candidates) {
          try {
            const rows = await fetchPredictionsByDate(sportKey, dateStr);
            if (rows.length > 0) {
              data = rows;
              effectiveDate = dateStr;
              break;
            }
          } catch {
            // Ignore per-date errors and continue with next candidate.
          }
        }
      }

      // For sparse soccer slates, jump to the nearest upcoming date with a fuller board.
      if (sportKey === "laliga" && data.length <= 1) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const allDates = await fetchAvailableDates(sportKey);
        const candidateDates = allDates
          .filter((d) => {
            const dt = new Date(`${d}T00:00:00`);
            return !Number.isNaN(dt.getTime()) && dt >= today;
          })
          .slice(0, 8);

        if (candidateDates.length > 1) {
          let bestDate = effectiveDate;
          let bestData = data;

          for (const dateStr of candidateDates) {
            try {
              const rows = await fetchPredictionsByDate(sportKey, dateStr);
              if (rows.length > bestData.length) {
                bestData = rows;
                bestDate = dateStr;
              }
            } catch {
              // Ignore single-date errors and continue scanning next candidates.
            }
          }

          if (bestData.length > data.length) {
            data = bestData;
            effectiveDate = bestDate;
          }
        }
      }

      setEvents(data);

      if (data.length > 0) {
        const gameDate = effectiveDate || data[0].date;
        setSelectedDate(gameDate);
        setActiveEvent(data[0]);

        const d = new Date(`${gameDate}T00:00:00`);
        if (!Number.isNaN(d.getTime())) {
          setCalendarMonth(d);
        }

        navigate(`/${sportKey.replace("_", "-")}?date=${gameDate}`, { replace: true });
      } else {
        setActiveEvent(null);
      }
    } catch (err) {
      setError(err.message || "Error al cargar datos.");
      setEvents([]);
      setActiveEvent(null);
    } finally {
      setLoading(false);
    }
  }, [navigate, sportKey]);

  const loadByDate = useCallback(async (dateStr) => {
    try {
      setLoading(true);
      setError("");

      let data = await fetchPredictionsByDate(sportKey, dateStr);
      let effectiveDate = dateStr;

      // If a valid date returns an empty board, move to nearest date with games.
      if (data.length === 0) {
        const allDates = await fetchAvailableDates(sportKey);
        const sorted = [...allDates].sort();
        const neighbors = [
          ...sorted.filter((d) => d >= dateStr),
          ...sorted.filter((d) => d < dateStr).reverse(),
        ].slice(0, 21);

        for (const candidate of neighbors) {
          if (candidate === dateStr) continue;
          try {
            const rows = await fetchPredictionsByDate(sportKey, candidate);
            if (rows.length > 0) {
              data = rows;
              effectiveDate = candidate;
              setError(`No habia datos para ${dateStr}. Mostrando ${candidate}.`);
              break;
            }
          } catch {
            // Ignore per-date errors while scanning neighbors.
          }
        }
      }

      setEvents(data);
      setSelectedDate(effectiveDate);
      setActiveEvent(data[0] || null);

      const d = new Date(`${effectiveDate}T00:00:00`);
      if (!Number.isNaN(d.getTime())) {
        setCalendarMonth(d);
      }

      navigate(`/${sportKey.replace("_", "-")}?date=${effectiveDate}`, { replace: true });
    } catch (err) {
      // If a deep-link date has no data, jump to the nearest available date.
      try {
        const allDates = await fetchAvailableDates(sportKey);
        const fallbackDate = findNearestDate(dateStr, allDates);

        if (fallbackDate && fallbackDate !== dateStr) {
          const fallbackData = await fetchPredictionsByDate(sportKey, fallbackDate);
          setEvents(fallbackData);
          setSelectedDate(fallbackDate);
          setActiveEvent(fallbackData[0] || null);

          const d2 = new Date(`${fallbackDate}T00:00:00`);
          if (!Number.isNaN(d2.getTime())) {
            setCalendarMonth(d2);
          }

          navigate(`/${sportKey.replace("_", "-")}?date=${fallbackDate}`, { replace: true });
          setError(`No habia datos para ${dateStr}. Mostrando ${fallbackDate}.`);
          return;
        }
      } catch {
        // Keep original empty-state handling if fallback lookup fails.
      }

      setError(err.message || "Error al cargar la fecha.");
      setEvents([]);
      setActiveEvent(null);
      setSelectedDate(dateStr);

      const d = new Date(`${dateStr}T00:00:00`);
      if (!Number.isNaN(d.getTime())) {
        setCalendarMonth(d);
      }

      navigate(`/${sportKey.replace("_", "-")}?date=${dateStr}`, { replace: true });
    } finally {
      setLoading(false);
    }
  }, [navigate, sportKey]);

  const refreshCurrentBoardSilently = useCallback(async () => {
    const targetDate = selectedDate || toYmdLocal(new Date());
    try {
      const rows = await fetchPredictionsByDate(sportKey, targetDate);
      setEvents(rows);
      if (rows.length > 0) {
        setActiveEvent((current) => {
          if (!current) return rows[0];
          return rows.find((item) => String(item.game_id) === String(current.game_id)) || current;
        });
      }
    } catch {
      // Ignore silent refresh failures and keep the current board visible.
    }
  }, [selectedDate, sportKey]);

  useEffect(() => {
    const queryDate = searchParams.get("date");

    if (queryDate) {
      loadByDate(queryDate);
    } else {
      loadToday();
    }
  }, [loadByDate, loadToday, searchParams]);

  useEffect(() => {
    refreshAvailableDates();
  }, [refreshAvailableDates]);

  useEffect(() => {
    setEvents([]);
    setSelectedDate("");
    setActiveEvent(null);
    setError("");
    setLoading(true);
    setEventSearch("");
  }, [sportKey]);

  useEffect(() => {
    if (sportKey !== "nba") return undefined;
    const localToday = toYmdLocal(new Date());
    const hasLiveGames = events.some((event) => String(event.status_state || "").toLowerCase() === "in");
    if (!hasLiveGames && selectedDate !== localToday) return undefined;

    const intervalId = setInterval(() => {
      refreshCurrentBoardSilently();
    }, 25000);

    return () => clearInterval(intervalId);
  }, [events, refreshCurrentBoardSilently, selectedDate, sportKey]);

  function handleOpenEvent(event) {
    setActiveEvent(event);
    setModalOpen(true);
  }

  const settledEvents = events.filter((event) => toHitValue(event.full_game_hit) !== null);
  const wins = settledEvents.filter((event) => toHitValue(event.full_game_hit) === true).length;
  const totalTracked = settledEvents.length;
  const hitRate = totalTracked > 0 ? (wins / totalTracked) * 100 : 0;
  const averageOddsSource = settledEvents
    .map((event) => formatDecimalOdds(
      event.closing_moneyline_odds
      ?? event.home_moneyline_odds
      ?? event.away_moneyline_odds
      ?? event.moneyline_odds
    ))
    .filter((value) => Number.isFinite(value));
  const averageOdds = averageOddsSource.length > 0
    ? averageOddsSource.reduce((sum, value) => sum + value, 0) / averageOddsSource.length
    : null;
  const summaryCards = [
    { label: socialMode ? "Total analizado" : "Total", value: String(totalTracked), accent: "text-cyan-200" },
    { label: socialMode ? "Correctas" : "Ganadas", value: String(wins), accent: "text-emerald-300" },
    { label: socialMode ? "% correcto" : "% Acierto", value: `${hitRate.toFixed(1)}%`, accent: "text-amber-200" },
    { label: socialMode ? "Referencia media" : "Cuota media", value: averageOdds ? averageOdds.toFixed(2) : "-", accent: "text-fuchsia-200" },
  ];
  const upcomingCount = events.filter((event) => event.result_available !== true).length;
  const premiumCount = events.filter((event) => {
    const tier = resolveEventTier(event);
    return tier === "ELITE" || tier === "PREMIUM";
  }).length;
  const showNcaaSearch = sportKey === "ncaa_baseball";
  const normalizedEventSearch = normalizeSearchText(eventSearch);
  const filteredEvents = showNcaaSearch && normalizedEventSearch
    ? events.filter((event) => {
        const haystack = normalizeSearchText([
          event.away_team,
          event.home_team,
          event.game_name,
          event.full_game_pick,
          event.time,
        ].filter(Boolean).join(" "));
        return haystack.includes(normalizedEventSearch);
      })
    : events;

  return (
    <>
      <main className="mx-auto max-w-[1780px] px-4 py-8 xl:px-6 2xl:px-8">
        <div className="grid gap-6 xl:grid-cols-[290px_minmax(0,1fr)_300px] 2xl:grid-cols-[310px_minmax(0,1fr)_320px]">
          <SidebarCalendar
            calendarMonth={calendarMonth}
            setCalendarMonth={setCalendarMonth}
            selectedDate={selectedDate}
            loading={loading}
            error={error}
            onSelectDate={loadByDate}
            onLoadToday={loadToday}
            availableDates={availableDates}
          />

          <section className="min-w-0">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold">Eventos {sportLabel}</h2>
                <p className="text-sm text-white/60">
                  {socialMode
                    ? "Abre cada evento para ver la lectura completa del modelo."
                    : "Haz click en un juego para abrir su detalle completo."}
                </p>
              </div>

              <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/70">
                {filteredEvents.length} juegos
              </div>
            </div>

            {filteredEvents.length === 0 && !loading ? (
              <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/70">
                {showNcaaSearch && normalizedEventSearch
                  ? `No encontramos eventos de NCAA Baseball para "${eventSearch}" en esta fecha.`
                  : socialMode
                    ? `No hay proyecciones disponibles para ${sportLabel} en esta fecha.`
                    : `No hay predicciones disponibles para ${sportLabel} en esta fecha.`}
              </div>
            ) : (
              <>
                {showNcaaSearch && (
                  <div className="mb-5 rounded-[24px] border border-white/10 bg-white/[0.035] p-4">
                    <label htmlFor="ncaa-event-search" className="mb-2 block text-xs font-semibold uppercase tracking-[0.18em] text-white/45">
                      Buscar evento NCAA Baseball
                    </label>
                    <input
                      id="ncaa-event-search"
                      type="text"
                      value={eventSearch}
                      onChange={(e) => setEventSearch(e.target.value)}
                      placeholder="Escribe equipo, matchup u hora..."
                      className="w-full rounded-2xl border border-white/12 bg-black/25 px-4 py-3 text-sm text-white outline-none transition placeholder:text-white/30 focus:border-cyan-300/60 focus:bg-black/35"
                    />
                    <p className="mt-2 text-xs text-white/50">
                      Aqui mostramos todos los juegos cargados para NCAA Baseball, incluso si no tienen linea disponible todavia.
                    </p>
                  </div>
                )}

                <div className="grid items-start gap-5 sm:grid-cols-2 xl:grid-cols-3">
                  {filteredEvents.map((event) => (
                    <EventCard key={event.game_id} event={event} onOpen={handleOpenEvent} sportKey={sportKey} />
                  ))}
                </div>

                <div className="mt-8">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-white/55">
                      {socialMode ? "Resumen del modelo" : "Resumen del dia"}
                    </h3>
                    <span className="text-xs text-white/45">
                      {socialMode ? "Basado en el resultado de la proyeccion principal" : "Basado en resultados del pick principal"}
                    </span>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    {summaryCards.map((card) => (
                      <div
                        key={card.label}
                        className="rounded-2xl border border-white/10 bg-[#262424] px-5 py-4 shadow-lg shadow-black/20"
                      >
                        <p className="text-xs uppercase tracking-[0.16em] text-white/45">{card.label}</p>
                        <p className={`mt-2 text-3xl font-semibold ${card.accent}`}>{card.value}</p>
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 h-3 overflow-hidden rounded-full bg-white/8">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-lime-300 transition-all duration-700"
                      style={{ width: `${totalTracked > 0 ? Math.min(100, hitRate || 0) : 0}%` }}
                    />
                  </div>

                  <div className="mt-10">
                    <div className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(16,19,27,0.98))] p-6 shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Prueba social</p>
                          <h3 className="mt-2 text-2xl font-semibold text-white">
                            {socialMode ? "Por que esta lectura se comparte mejor" : "Por qué esta experiencia vende mejor"}
                          </h3>
                        </div>
                        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                          Conversion UX
                        </span>
                      </div>

                      <div className="mt-6 grid gap-4 md:grid-cols-3">
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Claridad</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            {socialMode
                              ? "La lectura del modelo se entiende rapido, sin lenguaje agresivo de apuestas."
                              : "El usuario entiende el pick, la confianza y el contexto sin esfuerzo."}
                          </p>
                        </div>
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Autoridad</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            {socialMode
                              ? "Los indicadores se leen como analitica clara, no como una promesa agresiva."
                              : "Las métricas y tiers elevan la percepción de calidad y control."}
                          </p>
                        </div>
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Deseo</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            {socialMode
                              ? "La presentacion se siente seria, limpia y apta para mostrarla en contenido publico."
                              : "El diseño hace que el producto se sienta premium antes de venderlo."}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>

          <aside className="self-start xl:sticky xl:top-6">
            <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.14),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(52,211,153,0.10),transparent_30%),linear-gradient(180deg,rgba(22,26,36,0.96),rgba(14,17,24,0.98))] p-5 shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
              <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-200/85">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" />
                Board de {sportLabel}
              </div>

              <h3 className="mt-4 text-2xl font-semibold leading-tight text-white">
                {socialMode ? "Resumen General del Modelo" : "Resumen General del Dia"}
                {!socialMode && "(Mercado MonyLine)."}
              </h3>

              <div className="mt-6 grid gap-3">
                <div className="rounded-[22px] border border-white/10 bg-white/[0.045] p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Eventos activos</p>
                  <p className="mt-3 text-4xl font-semibold text-white">{events.length}</p>
                  <p className="mt-2 text-sm text-white/55">Slate disponible para {selectedDate || "hoy"}</p>
                </div>
                <div className="rounded-[22px] border border-white/10 bg-white/[0.045] p-4">
                  <p className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/45">
                    Premium / Elite
                    <span className="group relative inline-flex">
                      <button
                        type="button"
                        aria-label="Info de calculo Premium/Elite"
                        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-white/20 bg-white/10 text-[10px] font-bold text-white/75 transition hover:border-cyan-300/40 hover:bg-cyan-300/12 hover:text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-300/40"
                      >
                        i
                      </button>
                      <span className="pointer-events-none absolute left-1/2 top-full z-30 mt-2 w-64 -translate-x-1/2 rounded-xl border border-cyan-300/28 bg-[linear-gradient(180deg,rgba(12,19,30,0.98),rgba(8,13,22,0.98))] px-3 py-2 text-[11px] normal-case tracking-normal text-cyan-100/95 opacity-0 shadow-[0_16px_32px_rgba(0,0,0,0.3)] transition duration-200 group-hover:opacity-100 group-focus-within:opacity-100">
                        Conteo conectado al mismo tier de las cards via <span className="font-semibold">resolveEventTier(event)</span>. Incluye solo picks <span className="font-semibold">ELITE</span> y <span className="font-semibold">PREMIUM</span> con fallback por score/confianza.
                      </span>
                    </span>
                  </p>
                  <p className="mt-3 text-4xl font-semibold text-amber-200">{premiumCount}</p>
                  <p className="mt-2 text-sm text-white/55">{socialMode ? "Proyecciones con narrativa de mayor valor" : "Picks con narrativa de mayor valor"}</p>
                </div>
                <div className="rounded-[22px] border border-white/10 bg-white/[0.045] p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{socialMode ? "% correcto" : "% acierto"}</p>
                  <p className="mt-3 text-4xl font-semibold text-emerald-300">{hitRate.toFixed(1)}%</p>
                  <p className="mt-2 text-sm text-white/55">{socialMode ? "Referencia social basada en resultados del modelo" : "Prueba social basada en resultado real"}</p>
                </div>
                <div className="rounded-[22px] border border-white/10 bg-white/[0.045] p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Pendientes</p>
                  <p className="mt-3 text-4xl font-semibold text-cyan-200">{upcomingCount}</p>
                  <p className="mt-2 text-sm text-white/55">Partidos listos para explorar o vender</p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>

      {modalOpen && activeEvent && (
        <DetailModal event={activeEvent} onClose={() => setModalOpen(false)} sportKey={sportKey} />
      )}
    </>
  );
}
