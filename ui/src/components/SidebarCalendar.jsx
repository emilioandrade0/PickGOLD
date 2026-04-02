import { buildCalendarDays, dateToYMDLocal, formatDateInput } from "../utils/date.js";

export default function SidebarCalendar({
  calendarMonth,
  setCalendarMonth,
  selectedDate,
  loading,
  error,
  onSelectDate,
  onLoadToday,
  onRunUpdate,
  updateStatus,
  availableDates,
  title = "Calendario",
  subtitle = "Consulta predicciones historicas por fecha.",
  todayButtonLabel = "Cargar hoy",
  updateActionLabel = "Actualizar ahora",
  updateRunningLabel = "Actualizando...",
}) {
  const calendarDays = buildCalendarDays(calendarMonth);
  const enabledDateSet = new Set(Array.isArray(availableDates) ? availableDates : []);

  const monthLabel = calendarMonth.toLocaleDateString("es-MX", {
    month: "long",
    year: "numeric",
  });
  const updatePercent = Number(updateStatus?.percent);
  const safeUpdatePercent = Number.isFinite(updatePercent)
    ? Math.max(0, Math.min(100, updatePercent))
    : 0;
  const updateIsRunning = updateStatus?.status === "running";
  const updateIsDone = updateStatus?.status === "completed";
  const updateIsFailed = updateStatus?.status === "failed";
  const showUpdatePanel = typeof onRunUpdate === "function";
  const updateLabel = updateStatus?.current_step_label || updateStatus?.message || "Listo para actualizar.";
  const updateLogs = Array.isArray(updateStatus?.logs) ? updateStatus.logs : [];

  return (
    <aside className="self-start rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(15,18,26,0.98))] p-5 shadow-[0_24px_60px_rgba(0,0,0,0.28)] backdrop-blur-sm lg:sticky lg:top-6">
      <div className="mb-5">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-200/85">
          <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" />
          Control panel
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
            ?
          </button>

          <span className="capitalize">{monthLabel}</span>

          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))
            }
            className="rounded-xl border border-white/8 bg-white/[0.05] px-3 py-1.5 transition hover:bg-white/[0.08]"
          >
            ?
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

        {showUpdatePanel && (
          <div className="rounded-[26px] border border-cyan-400/20 bg-[linear-gradient(180deg,rgba(25,40,52,0.86),rgba(22,22,31,0.92))] p-4 shadow-[0_18px_40px_rgba(0,0,0,0.22)]">
            <button
              onClick={onRunUpdate}
              disabled={updateIsRunning}
              className={`w-full rounded-xl border px-4 py-3 text-sm font-semibold transition ${
                updateIsRunning
                  ? "cursor-not-allowed border-cyan-300/20 bg-cyan-300/10 text-cyan-100/70"
                  : "border-cyan-300/40 bg-cyan-300/12 text-cyan-100 hover:border-cyan-200/60 hover:bg-cyan-300/18"
              }`}
            >
              {updateIsRunning ? updateRunningLabel : updateActionLabel}
            </button>

            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between text-xs text-white/65">
                <span className="flex items-center gap-2">
                  {updateIsRunning && (
                    <>
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-cyan-200/25 border-t-cyan-200" />
                      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-cyan-300 shadow-[0_0_12px_rgba(103,232,249,0.95)]" />
                    </>
                  )}
                  <span>{updateIsDone ? "Completado" : updateIsFailed ? "Con error" : "Progreso"}</span>
                </span>
                <span>{safeUpdatePercent}%</span>
              </div>
              <div className="relative h-3 overflow-hidden rounded-full bg-white/8">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    updateIsFailed
                      ? "bg-gradient-to-r from-rose-500 to-amber-400"
                      : "bg-gradient-to-r from-cyan-400 via-sky-400 to-amber-300"
                  }`}
                  style={{ width: `${safeUpdatePercent}%` }}
                />
                {updateIsRunning && (
                  <div
                    className="absolute inset-y-0 -translate-x-full animate-[pulse_1.6s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-white/25 to-transparent"
                    style={{ width: "30%" }}
                  />
                )}
              </div>
            </div>

            <div className="mt-3 space-y-1">
              <p className="text-sm font-medium text-white/85">{updateLabel}</p>
              <p className="text-xs text-white/55">
                Paso {updateStatus?.completed_steps ?? 0} de {updateStatus?.total_steps ?? 0}
              </p>
              {updateIsFailed && updateStatus?.error && (
                <p className="text-xs text-rose-200">{updateStatus.error}</p>
              )}
            </div>

            {updateLogs.length > 0 && (
              <div className="mt-3 rounded-xl bg-black/20 p-3 text-[11px] text-white/60">
                {updateLogs.slice(-3).map((line, index) => (
                  <p key={`${index}-${line}`} className="truncate">
                    {line}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}

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
