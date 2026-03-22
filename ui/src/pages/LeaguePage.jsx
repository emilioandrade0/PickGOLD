import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import EventCard from "../components/EventCard.jsx";
import DetailModal from "../components/DetailModal.jsx";
import { fetchAvailableDates, fetchPredictionsByDate, fetchTodayPredictions } from "../services/api.js";

function toYmdLocal(dateObj) {
  const y = dateObj.getFullYear();
  const m = String(dateObj.getMonth() + 1).padStart(2, "0");
  const d = String(dateObj.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export default function LeaguePage({ sportKey, sportLabel }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [activeEvent, setActiveEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [modalOpen, setModalOpen] = useState(false);

  function findNearestDate(targetDate, availableDates) {
    if (!targetDate || availableDates.length === 0) return "";

    const sorted = [...availableDates].sort();
    const sameOrAfter = sorted.find((d) => d >= targetDate);
    if (sameOrAfter) return sameOrAfter;
    return sorted[sorted.length - 1] || "";
  }

  async function loadToday() {
    try {
      setLoading(true);
      setError("");

      let data = await fetchTodayPredictions(sportKey);
      let effectiveDate = data.length > 0 ? data[0].date : "";

      // If today's slate is empty, jump to the nearest date that actually has games.
      if (data.length === 0) {
        const allDates = await fetchAvailableDates(sportKey);
        const todayStr = toYmdLocal(new Date());

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

  useEffect(() => {
    const queryDate = searchParams.get("date");

    if (queryDate) {
      loadByDate(queryDate);
    } else {
      loadToday();
    }
  }, [sportKey]);

  function handleOpenEvent(event) {
    setActiveEvent(event);
    setModalOpen(true);
  }

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
              <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
                {events.map((event) => (
                  <EventCard key={event.game_id} event={event} onOpen={handleOpenEvent} sportKey={sportKey} />
                ))}
              </div>
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