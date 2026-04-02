import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import EventCard from "../components/EventCard.jsx";
import DetailModal from "../components/DetailModal.jsx";
import {
  fetchAvailableDates,
  fetchSportUpdateStatus,
  fetchPredictionsByDate,
  fetchTodayPredictions,
  startSportUpdateAll,
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
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const supportsUpdatePipeline = ["nba", "mlb", "tennis", "kbo", "nhl", "liga_mx", "laliga", "euroleague", "ncaa_baseball"].includes(sportKey);

  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [activeEvent, setActiveEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [modalOpen, setModalOpen] = useState(false);
  const [availableDates, setAvailableDates] = useState([]);
  const [eventSearch, setEventSearch] = useState("");
  const [updateStatus, setUpdateStatus] = useState({
    status: "idle",
    percent: 0,
    message: `Listo para actualizar ${sportLabel}.`,
    completed_steps: 0,
    total_steps: 0,
    logs: [],
  });
  const previousUpdateStatusRef = useRef("idle");

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

  async function handleRunSportUpdate() {
    try {
      const nextStatus = await startSportUpdateAll(sportKey);
      previousUpdateStatusRef.current = nextStatus?.status || "idle";
      setUpdateStatus(nextStatus);
    } catch (err) {
      setUpdateStatus((current) => ({
        ...current,
        status: "failed",
        message: `No se pudo iniciar la actualizacion ${sportLabel}.`,
        error: err.message || "Error desconocido.",
      }));
    }
  }

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
    setUpdateStatus({
      status: "idle",
      percent: 0,
      message: `Listo para actualizar ${sportLabel}.`,
      completed_steps: 0,
      total_steps: 0,
      logs: [],
    });
    previousUpdateStatusRef.current = "idle";
  }, [sportKey, sportLabel]);

  useEffect(() => {
    if (!supportsUpdatePipeline) return undefined;

    let active = true;
    let timerId = null;

    const pollStatus = async () => {
      try {
        const status = await fetchSportUpdateStatus(sportKey);
        if (!active) return;
        setUpdateStatus(status);

        if (status?.status === "running") {
          timerId = setTimeout(pollStatus, 2000);
        } else if (
          previousUpdateStatusRef.current === "running" &&
          status?.status === "completed"
        ) {
          await refreshAvailableDates();
          if (selectedDate) {
            await loadByDate(selectedDate);
          } else {
            await loadToday();
          }
        }

        previousUpdateStatusRef.current = status?.status || "idle";
      } catch {
        if (!active) return;
        timerId = setTimeout(pollStatus, 4000);
      }
    };

    pollStatus();

    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [loadByDate, loadToday, refreshAvailableDates, selectedDate, sportKey, supportsUpdatePipeline, updateStatus.status]);

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
    { label: "Total", value: String(totalTracked), accent: "text-cyan-200" },
    { label: "Ganadas", value: String(wins), accent: "text-emerald-300" },
    { label: "% Acierto", value: `${hitRate.toFixed(1)}%`, accent: "text-amber-200" },
    { label: "Cuota media", value: averageOdds ? averageOdds.toFixed(2) : "-", accent: "text-fuchsia-200" },
  ];
  const upcomingCount = events.filter((event) => event.result_available !== true).length;
  const premiumCount = events.filter((event) => String(event.recommended_tier || "").toLowerCase().includes("elite") || String(event.recommended_tier || "").toLowerCase().includes("premium")).length;
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
      <main className="mx-auto max-w-7xl px-6 py-8">
        <section className="mb-8 overflow-hidden rounded-[32px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.16),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(52,211,153,0.12),transparent_26%),linear-gradient(180deg,rgba(22,26,36,0.96),rgba(14,17,24,0.98))] p-6 shadow-[0_24px_80px_rgba(0,0,0,0.24)]">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.3fr)_420px] xl:items-center">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/90">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" />
                Board premium de {sportLabel}
              </div>
              <h2 className="max-w-3xl text-3xl font-semibold leading-tight text-white sm:text-4xl">
                Una vista que hace sentir el valor de tus picks desde el primer segundo.
              </h2>
              <p className="mt-4 max-w-2xl text-base leading-7 text-white/68">
                Menos saturacion, mejor foco. Selecciona fecha, detecta picks fuertes y navega
                tu board como si fuera un producto premium listo para convertir usuarios en clientes.
              </p>

              <div className="mt-6 flex flex-wrap gap-3 text-sm text-white/70">
                <span className="rounded-full border border-white/10 bg-white/[0.045] px-4 py-2">Confianza visible</span>
                <span className="rounded-full border border-white/10 bg-white/[0.045] px-4 py-2">Mercados accionables</span>
                <span className="rounded-full border border-white/10 bg-white/[0.045] px-4 py-2">Lectura instantanea</span>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-[24px] border border-white/10 bg-white/[0.045] p-5">
                <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Eventos activos</p>
                <p className="mt-3 text-4xl font-semibold text-white">{events.length}</p>
                <p className="mt-2 text-sm text-white/55">Slate disponible para {selectedDate || "hoy"}</p>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.045] p-5">
                <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Premium / Elite</p>
                <p className="mt-3 text-4xl font-semibold text-amber-200">{premiumCount}</p>
                <p className="mt-2 text-sm text-white/55">Picks con narrativa de mayor valor</p>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.045] p-5">
                <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">% acierto</p>
                <p className="mt-3 text-4xl font-semibold text-emerald-300">{hitRate.toFixed(1)}%</p>
                <p className="mt-2 text-sm text-white/55">Prueba social basada en resultado real</p>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.045] p-5">
                <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Pendientes</p>
                <p className="mt-3 text-4xl font-semibold text-cyan-200">{upcomingCount}</p>
                <p className="mt-2 text-sm text-white/55">Partidos listos para explorar o vender</p>
              </div>
            </div>
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
          <SidebarCalendar
            calendarMonth={calendarMonth}
            setCalendarMonth={setCalendarMonth}
            selectedDate={selectedDate}
            loading={loading}
            error={error}
            onSelectDate={loadByDate}
            onLoadToday={loadToday}
            onRunUpdate={supportsUpdatePipeline ? handleRunSportUpdate : undefined}
            updateStatus={supportsUpdatePipeline ? updateStatus : undefined}
            availableDates={availableDates}
            updateActionLabel={`Actualizar ${sportLabel} ahora`}
            updateRunningLabel={`Actualizando ${sportLabel}...`}
          />

          <section className="min-w-0">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold">Eventos {sportLabel}</h2>
                <p className="text-sm text-white/60">
                  Haz click en un juego para abrir su detalle completo.
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
                      Resumen del dia
                    </h3>
                    <span className="text-xs text-white/45">
                      Basado en resultados del pick principal
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

                  <div className="mt-10 grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_420px]">
                    <div className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(16,19,27,0.98))] p-6 shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Prueba social</p>
                          <h3 className="mt-2 text-2xl font-semibold text-white">Por qué esta experiencia vende mejor</h3>
                        </div>
                        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                          Conversion UX
                        </span>
                      </div>

                      <div className="mt-6 grid gap-4 md:grid-cols-3">
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Claridad</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            El usuario entiende el pick, la confianza y el contexto sin esfuerzo.
                          </p>
                        </div>
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Autoridad</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            Las métricas y tiers elevan la percepción de calidad y control.
                          </p>
                        </div>
                        <div className="rounded-[22px] border border-white/8 bg-white/[0.04] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Deseo</p>
                          <p className="mt-3 text-sm leading-6 text-white/72">
                            El diseño hace que el producto se sienta premium antes de venderlo.
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[30px] border border-amber-300/22 bg-[linear-gradient(180deg,rgba(255,199,76,0.14),rgba(255,199,76,0.05))] p-6 shadow-[0_24px_60px_rgba(246,196,83,0.14)]">
                      <p className="text-[11px] uppercase tracking-[0.18em] text-amber-100/70">Oferta premium</p>
                      <h3 className="mt-2 text-2xl font-semibold text-white">Escala esto como un producto de suscripción</h3>
                      <p className="mt-4 text-sm leading-6 text-white/76">
                        Usa este board como escaparate premium: best picks, insights, scoring y mercados destacados.
                      </p>

                      <div className="mt-5 space-y-3 text-sm text-white/78">
                        <div className="flex items-center gap-3">
                          <span className="h-2 w-2 rounded-full bg-emerald-400" />
                          <span>Plan mensual con acceso al board diario</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="h-2 w-2 rounded-full bg-emerald-400" />
                          <span>Upsell a Best Picks e Insights avanzados</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="h-2 w-2 rounded-full bg-emerald-400" />
                          <span>Percepción de valor reforzada por diseño + métricas</span>
                        </div>
                      </div>

                      <div className="mt-6 rounded-[24px] border border-black/10 bg-black/18 p-4">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Anchor price</p>
                        <p className="mt-2 text-4xl font-semibold text-white">$79<span className="text-base font-medium text-white/55"> / mes</span></p>
                        <p className="mt-2 text-sm text-white/68">Best Picks + Insights + experiencia premium de picks.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      </main>

      {modalOpen && activeEvent && (
        <DetailModal event={activeEvent} onClose={() => setModalOpen(false)} sportKey={sportKey} />
      )}
    </>
  );
}
