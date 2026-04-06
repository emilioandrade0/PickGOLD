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
  subtitle = "Consulta el historial y cambia de fecha en segundos.",
  todayButtonLabel = "Cargar hoy",
}) {
  const calendarDays = buildCalendarDays(calendarMonth);
  const enabledDateSet = new Set(Array.isArray(availableDates) ? availableDates : []);

  const monthLabel = calendarMonth.toLocaleDateString("es-MX", {
    month: "long",
    year: "numeric",
  });


  return (
    <aside className="self-start rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(15,18,26,0.98))] p-5 shadow-[0_24px_60px_rgba(0,0,0,0.28)] backdrop-blur-sm lg:sticky lg:top-6">
      <div className="mb-5">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-200/85">
          <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" />
          Centro de control
        </div>
        <h2 className="text-2xl font-semibold">{title}</h2>
        <p className="mt-2 text-sm leading-6 text-white/60">{subtitle}</p>
      </div>

      <div className="rounded-[26px] border border-white/10 bg-black/22 p-4">
        <div className="mb-4 flex items-center justify-between text-sm text-white/70">
          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))
            }
            className="rounded-xl border border-white/8 bg-white/[0.05] px-3 py-1.5 transition hover:bg-white/[0.08]"
          >
            ‹
          </button>

          <span className="capitalize">{monthLabel}</span>

          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))
            }
            className="rounded-xl border border-white/8 bg-white/[0.05] px-3 py-1.5 transition hover:bg-white/[0.08]"
          >
            ›
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
                    ? "bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] font-semibold text-[#131821] shadow-[0_8px_20px_rgba(246,196,83,0.25)]"
                    : dateEnabled
                      ? "bg-white/[0.05] text-white/70 hover:bg-cyan-300/10 hover:text-cyan-100"
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
          className="w-full rounded-2xl border border-amber-300/28 bg-[linear-gradient(180deg,rgba(255,199,76,0.16),rgba(255,199,76,0.08))] px-6 py-3 text-base font-semibold text-amber-100 shadow-[0_16px_34px_rgba(0,0,0,0.18)] transition hover:-translate-y-0.5 hover:border-amber-300/40"
        >
          {todayButtonLabel}
        </button>


        <div className="rounded-[24px] border border-white/8 bg-black/18 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Fecha seleccionada</p>
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
