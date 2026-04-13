import { useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { buildCalendarDays, dateToYMDLocal, formatDateInput } from "../utils/date.js";

function SidebarLink({ to, label, icon, compact = false }) {
  return (
    <NavLink
      to={to}
      title={label}
      className={({ isActive }) =>
        `group flex items-center rounded-xl border transition ${
          compact
            ? "justify-center px-0 py-2.5"
            : "gap-2.5 px-3 py-2.5"
        } ${
          isActive
            ? "border-white/18 bg-white/[0.08] text-white"
            : "border-transparent text-white/72 hover:border-white/14 hover:bg-white/[0.05] hover:text-white"
        }`
      }
    >
      <span className="text-base leading-none">{icon}</span>
      {!compact && <span className="text-sm font-medium">{label}</span>}
    </NavLink>
  );
}

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
  sportLabel = "",
  compactMode = false,
  onCompactChange,
}) {
  const { uiTheme } = useAppSettings();
  const isDashboardPro = uiTheme === "dashboard_pro";
  const [sportsCollapsed, setSportsCollapsed] = useState(false);

  const calendarDays = buildCalendarDays(calendarMonth);
  const enabledDateSet = new Set(Array.isArray(availableDates) ? availableDates : []);

  const monthLabel = calendarMonth.toLocaleDateString("es-MX", {
    month: "long",
    year: "numeric",
  });

  const sportLinks = useMemo(() => ([
    { label: "NBA", path: "/nba" },
    { label: "MLB", path: "/mlb" },
    { label: "NHL", path: "/nhl" },
    { label: "Liga MX", path: "/liga-mx" },
    { label: "LaLiga", path: "/laliga" },
    { label: "Bundesliga", path: "/bundesliga" },
    { label: "Ligue 1", path: "/ligue1" },
    { label: "EuroLeague Basquetball", path: "/euroleague" },
    { label: "KBO", path: "/kbo" },
    { label: "Tennis WTA", path: "/tennis" },
    { label: "NCAA Triple-A", path: "/triple-a" },
  ]), []);

  if (!isDashboardPro) {
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
              onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))}
              className="rounded-xl border border-white/8 bg-white/[0.05] px-3 py-1.5 transition hover:bg-white/[0.08]"
            >
              ‹
            </button>

            <span className="capitalize">{monthLabel}</span>

            <button
              onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))}
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
              if (!dateObj) return <div key={`empty-${index}`} className="h-9" />;

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
            <p className="mt-1 text-lg font-semibold">{selectedDate ? formatDateInput(selectedDate) : "Sin fecha"}</p>
          </div>

          {loading && <div className="rounded-2xl bg-black/15 p-4 text-sm text-white/70">Cargando...</div>}
          {error && <div className="rounded-2xl border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">{error}</div>}
        </div>
      </aside>
    );
  }

  const compact = Boolean(compactMode);

  return (
    <aside className={`self-start overflow-hidden rounded-[26px] border border-white/10 bg-[linear-gradient(180deg,rgba(13,16,24,0.98),rgba(8,11,18,0.99))] shadow-[0_20px_50px_rgba(0,0,0,0.35)] lg:sticky lg:top-6 ${compact ? "px-2 py-3" : "px-4 py-4"}`}>
      <div className="mb-3 flex items-center justify-between gap-2">
        {!compact && (
          <div>
            <p className="text-[11px] uppercase tracking-[0.2em] text-amber-200/80">{sportLabel || "PickGold"}</p>
            <p className="mt-1 text-base font-semibold text-white">Panel</p>
          </div>
        )}
        <button
          type="button"
          onClick={() => onCompactChange?.(!compact)}
          className="rounded-lg border border-white/12 bg-white/[0.05] px-2 py-1 text-xs font-semibold text-white/72 hover:text-white"
          title={compact ? "Expandir" : "Compactar"}
        >
          {compact ? "»" : "«"}
        </button>
      </div>

      <div className="space-y-1 border-b border-white/10 pb-3">
        <SidebarLink to="/live" label="En Vivo" icon="▦" compact={compact} />
        <SidebarLink to="/resumen-dia" label="Eventos Del Dia" icon="◉" compact={compact} />
        <SidebarLink to="/best-picks" label="Picks Del Dia" icon="◌" compact={compact} />
        <SidebarLink to="/estadisticas" label="Rachas Por Equipo" icon="⌁" compact={compact} />
        <SidebarLink to="/insights" label="Insights" icon="◍" compact={compact} />
        <SidebarLink to="/admin/approve-users" label="Configuración" icon="⚙" compact={compact} />
      </div>

      <div className="mt-3 border-b border-white/10 pb-3">
        <div className="mb-2 flex items-center justify-between">
          {!compact && <p className="text-[10px] uppercase tracking-[0.16em] text-white/45">Sports</p>}
          <button
            type="button"
            onClick={() => setSportsCollapsed((prev) => !prev)}
            className="rounded-md border border-white/12 bg-white/[0.04] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-white/60 hover:text-white/85"
          >
            {sportsCollapsed ? "+" : "−"}
          </button>
        </div>

        {!sportsCollapsed && (
          <div className={`grid gap-1 ${compact ? "grid-cols-1" : "grid-cols-2"}`}>
            {sportLinks.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                title={item.label}
                className={({ isActive }) =>
                  `rounded-lg border px-2 py-2 text-xs font-semibold transition ${
                    isActive
                      ? "border-amber-300/60 bg-amber-300/14 text-amber-100"
                      : "border-white/10 bg-white/[0.03] text-white/70 hover:border-white/18 hover:bg-white/[0.06] hover:text-white"
                  } ${compact ? "text-center" : ""}`
                }
              >
                {compact ? item.label.slice(0, 3).toUpperCase() : item.label}
              </NavLink>
            ))}
          </div>
        )}
      </div>

      {!compact && (
        <>
          <div className="mt-3 rounded-[18px] border border-white/10 bg-black/24 p-3">
            <p className="mb-2 text-[10px] uppercase tracking-[0.16em] text-white/45">Calendario</p>
            <div className="mb-3 flex items-center justify-between text-xs text-white/70">
              <button onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1))} className="rounded-md border border-white/12 bg-white/[0.05] px-2 py-1">‹</button>
              <span className="capitalize">{monthLabel}</span>
              <button onClick={() => setCalendarMonth(new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1))} className="rounded-md border border-white/12 bg-white/[0.05] px-2 py-1">›</button>
            </div>
            <div className="mb-1 grid grid-cols-7 text-center text-[10px] uppercase text-white/35">
              {["L", "M", "M", "J", "V", "S", "D"].map((d, i) => <div key={`${d}-${i}`}>{d}</div>)}
            </div>
            <div className="grid grid-cols-7 gap-1">
              {calendarDays.map((dateObj, index) => {
                if (!dateObj) return <div key={`empty-${index}`} className="h-7" />;
                const day = dateObj.getDate();
                const dateStr = dateToYMDLocal(dateObj);
                const active = dateStr === selectedDate;
                const dateEnabled = enabledDateSet.size === 0 || enabledDateSet.has(dateStr);
                return (
                  <button
                    key={dateStr}
                    disabled={!dateEnabled}
                    onClick={() => onSelectDate(dateStr)}
                    className={`rounded-md py-1.5 text-[11px] transition ${active ? "bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] font-semibold text-[#131821]" : dateEnabled ? "bg-white/[0.05] text-white/70 hover:bg-cyan-300/10 hover:text-cyan-100" : "cursor-not-allowed bg-white/5 text-white/25"}`}
                  >
                    {day}
                  </button>
                );
              })}
            </div>
          </div>

          <button onClick={onLoadToday} className="mt-3 w-full rounded-xl border border-amber-300/28 bg-[linear-gradient(180deg,rgba(255,199,76,0.16),rgba(255,199,76,0.08))] px-4 py-2.5 text-sm font-semibold text-amber-100 transition hover:border-amber-300/40">{todayButtonLabel}</button>

          <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
            <p className="text-[10px] uppercase tracking-[0.16em] text-white/45">Fecha</p>
            <p className="mt-1 text-sm font-semibold text-white">{selectedDate ? formatDateInput(selectedDate) : "Sin fecha"}</p>
          </div>

          {loading && <div className="mt-2 rounded-xl bg-black/20 p-3 text-xs text-white/70">Cargando...</div>}
          {error && <div className="mt-2 rounded-xl border border-red-400/20 bg-red-500/10 p-3 text-xs text-red-200">{error}</div>}
        </>
      )}
    </aside>
  );
}
