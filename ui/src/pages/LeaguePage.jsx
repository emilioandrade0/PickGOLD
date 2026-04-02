import { useEffect, useRef, useState } from "react";
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

export default function LeaguePage({ sportKey, sportLabel }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const supportsUpdatePipeline = ["nba", "mlb"].includes(sportKey);

  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [activeEvent, setActiveEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [modalOpen, setModalOpen] = useState(false);
  const [availableDates, setAvailableDates] = useState([]);
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

  async function refreshAvailableDates() {
    try {
      const dates = await fetchAvailableDates(sportKey);
      setAvailableDates(Array.isArray(dates) ? dates : []);
    } catch {
      setAvailableDates([]);
    }
  }

  async function loadToday() {
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
  }

  async function loadByDate(dateStr) {
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
  }

  async function refreshCurrentBoardSilently() {
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
  }

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
  }, [sportKey]);

  useEffect(() => {
    refreshAvailableDates();
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
  }, [sportKey, selectedDate, updateStatus.status]);

  useEffect(() => {
    if (sportKey !== "nba") return undefined;
    const localToday = toYmdLocal(new Date());
    const hasLiveGames = events.some((event) => String(event.status_state || "").toLowerCase() === "in");
    if (!hasLiveGames && selectedDate !== localToday) return undefined;

    const intervalId = setInterval(() => {
      refreshCurrentBoardSilently();
    }, 25000);

    return () => clearInterval(intervalId);
  }, [sportKey, selectedDate, events]);

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

  return (
    <>
      <main className="mx-auto max-w-7xl px-6 py-6">
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
                {events.length} juegos
              </div>
            </div>

            {events.length === 0 && !loading ? (
              <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/70">
                No hay predicciones disponibles para {sportLabel} en esta fecha.
              </div>
            ) : (
              <>
                <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
                  {events.map((event) => (
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
