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
            ?
          </button>

          <span className="capitalize">{monthLabel}</span>

          <button
            onClick={() =>
              setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))
            }
            className="rounded-lg bg-white/5 px-3 py-1 hover:bg-white/10"
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

        {showUpdatePanel && (
          <div className="rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-500/10 via-slate-900/80 to-amber-400/10 p-4 shadow-lg shadow-black/25">
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
