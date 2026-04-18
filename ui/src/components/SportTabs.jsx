import { NavLink } from "react-router-dom";
import { useAppSettings } from "../context/AppSettingsContext.jsx";

const SPORT_TABS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "wnba", label: "WNBA", path: "/wnba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "lmb", label: "LMB", path: "/lmb" },
  { key: "triple_a", label: "Triple-A", path: "/triple-a" },
  { key: "tennis", label: "Tennis", path: "/tennis" },
  { key: "kbo", label: "KBO", path: "/kbo" },
  { key: "nhl", label: "NHL", path: "/nhl" },
  { key: "euroleague", label: "EuroLeague", path: "/euroleague" },
  { key: "liga_mx", label: "Liga MX", path: "/liga-mx" },
  { key: "laliga", label: "LaLiga", path: "/laliga" },
  { key: "bundesliga", label: "Bundesliga", path: "/bundesliga" },
  { key: "ligue1", label: "Ligue 1", path: "/ligue1" },
];

const MODULE_TABS = [
  { key: "live", label: "En Vivo", path: "/live" },
  { key: "resumen_dia", label: "Resumen del día", path: "/resumen-dia" },
  { key: "best_picks", label: "Best Picks", path: "/best-picks" },
  { key: "estadisticas", label: "Estadisticas", path: "/estadisticas" },
];

const ADMIN_TABS = [
  { key: "admin_insights", label: "Insights", path: "/insights" },
  { key: "admin_weekday_scoring", label: "Weekday Scoring", path: "/weekday-scoring" },
  { key: "admin_panel", label: "Configuracion", path: "/admin/approve-users" },
];

function mapTabLabel(tab, socialMode) {
  if (!socialMode) return tab;
  if (tab.key === "best_picks") return { ...tab, label: "Radar" };
  if (tab.key === "admin_weekday_scoring") return { ...tab, label: "Ritmo" };
  if (tab.key === "admin_insights") return { ...tab, label: "Modelo" };
  if (tab.key === "resumen_dia") return { ...tab, label: "Resumen" };
  if (tab.key === "estadisticas") return { ...tab, label: "Metricas" };
  return tab;
}

function TabPill({ tab }) {
  return (
    <NavLink
      key={tab.key}
      to={tab.path}
      className={({ isActive }) =>
        `rounded-xl border px-4 py-2.5 text-sm font-semibold tracking-[0.01em] transition ${
          isActive
            ? "border-amber-300/70 bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#0f1218] shadow-[0_10px_24px_rgba(246,196,83,0.18)]"
            : "border-white/10 bg-white/[0.03] text-white/78 hover:border-white/20 hover:bg-white/[0.08] hover:text-white"
        }`
      }
    >
      {tab.label}
    </NavLink>
  );
}

function TabGroup({ title, tabs }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-black/12 p-2.5">
      <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-white/45">{title}</p>
      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <TabPill key={tab.key} tab={tab} />
        ))}
      </div>
    </div>
  );
}

export default function SportTabs({ userRole = null }) {
  const { socialMode, uiTheme } = useAppSettings();
  const isDashboardPro = uiTheme === "dashboard_pro";
  const isClassicLight = uiTheme === "classic_light";
  const sports = SPORT_TABS.map((tab) => mapTabLabel(tab, socialMode));
  const modules = MODULE_TABS.map((tab) => mapTabLabel(tab, socialMode));
  const adminTabs = userRole === "admin" ? ADMIN_TABS.map((tab) => mapTabLabel(tab, socialMode)) : [];

  if (isClassicLight) {
    return (
      <div className="classic-light-surface classic-light-tabs overflow-x-auto">
        <div className="flex min-w-max items-center gap-2 whitespace-nowrap">
          {sports.map((tab, index) => (
            <div key={tab.key} className="flex items-center gap-2">
              <NavLink
                to={tab.path}
                className={({ isActive }) =>
                  `rounded-md px-1.5 py-1 text-sm font-semibold uppercase tracking-[0.06em] transition ${
                    isActive
                      ? "bg-[#ececf0] text-[#232730]"
                      : "text-[#3f4452] hover:bg-[#ececf0] hover:text-[#20242d]"
                  }`
                }
              >
                {tab.label}
              </NavLink>
              {index < sports.length - 1 && <span className="text-[#656a77]">■</span>}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className={isDashboardPro ? "" : "rounded-2xl border border-white/8 bg-white/[0.02] p-2"}>
        <TabGroup title={socialMode ? "Ligas" : "Deportes"} tabs={sports} />
      </div>
      <div className={isDashboardPro ? "" : "rounded-2xl border border-white/8 bg-white/[0.02] p-2"}>
        <TabGroup title="En vivo y cierre" tabs={modules} />
      </div>
      {adminTabs.length > 0 ? (
        <div className={isDashboardPro ? "" : "rounded-2xl border border-white/8 bg-white/[0.02] p-2"}>
          <TabGroup title="Admin" tabs={adminTabs} />
        </div>
      ) : null}
    </div>
  );
}
