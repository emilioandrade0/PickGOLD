import { buildCalendarDays, dateToYMDLocal, formatDateInput } from "../utils/date.js";

export default function SidebarCalendar({
  calendarMonth,
  setCalendarMonth,
  selectedDate,
  loading,
  error,
  onSelectDate,
  onLoadToday,
  availableDates,
  title = "Calendario",
  subtitle = "Consulta predicciones históricas por fecha.",
  todayButtonLabel = "Cargar hoy",
}) {
  const calendarDays = buildCalendarDays(calendarMonth);
  const enabledDateSet = new Set(Array.isArray(availableDates) ? availableDates : []);

  const monthLabel = calendarMonth.toLocaleDateString("es-MX", {
    month: "long",
    year: "numeric",
  });

  return (
    <aside className="self-start rounded-3xl border border-white/10 bg-[#171a21]/85 p-5 shadow-2xl shadow-black/40 backdrop-blur-sm lg:sticky lg:top-6">
      <div className="mb-5">
        <h2 className="text-xl font-semibold">{title}</h2>
        <p className="text-sm text-white/60">{subtitle}</p>
      </div>

      <div className="rounded-2xl border border-white/10 bg-black/25 p-4">
        <div className="mb-4 flex items-center justify-between text-sm text-white/70">
          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))
            }
            className="rounded-lg bg-white/5 px-3 py-1 hover:bg-white/10"
          >
            ←
          </button>

          <span className="capitalize">{monthLabel}</span>

          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))
            }
            className="rounded-lg bg-white/5 px-3 py-1 hover:bg-white/10"
          >
            →
          </button>
        </div>

        <div className="mb-2 grid grid-cols-7 text-center text-[11px] uppercase text-white/40">
          {["L", "M", "M", "J", "V", "S", "D"].map((d, i) => (
            <div key={`${d}-${i}`} className="py-2">
              {d}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-7 gap-1">
          {calendarDays.map((dateObj, index) => {
            if (!dateObj) {
              return <div key={`empty-${index}`} className="h-9" />;
            }

            const day = dateObj.getDate();
            const dateStr = dateToYMDLocal(dateObj);
            const active = dateStr === selectedDate;
            const dateEnabled = enabledDateSet.size === 0 || enabledDateSet.has(dateStr);

            return (
              <button
                key={dateStr}
                disabled={!dateEnabled}
                onClick={() => onSelectDate(dateStr)}
                className={`rounded-lg py-2 text-sm transition ${
                  active
                    ? "bg-amber-300 font-semibold text-black"
                    : dateEnabled
                      ? "bg-white/5 text-white/70 hover:bg-cyan-300/10 hover:text-cyan-100"
                      : "cursor-not-allowed bg-white/5 text-white/25"
                }`}
              >
                {day}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-5 space-y-3">
        <button
          onClick={onLoadToday}
          className="w-full rounded-xl border border-amber-300/50 bg-amber-300/12 px-6 py-3 text-base font-medium text-amber-200 shadow-lg shadow-black/30"
        >
          {todayButtonLabel}
        </button>

        <div className="rounded-2xl bg-black/15 p-4">
          <p className="text-sm text-white/50">Fecha seleccionada</p>
          <p className="mt-1 text-lg font-semibold">
            {selectedDate ? formatDateInput(selectedDate) : "Sin fecha"}
          </p>
        </div>

        {loading && (
          <div className="rounded-2xl bg-black/15 p-4 text-sm text-white/70">
            Cargando...
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">
            {error}
          </div>
        )}
      </div>
    </aside>
  );
}