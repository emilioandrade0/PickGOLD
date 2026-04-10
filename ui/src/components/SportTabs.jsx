import { NavLink } from "react-router-dom";
import { useAppSettings } from "../context/AppSettingsContext.jsx";

const SPORTS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "triple_a", label: "Triple-A", path: "/triple-a" },
  { key: "tennis", label: "Tennis", path: "/tennis" },
  { key: "kbo", label: "KBO", path: "/kbo" },
  { key: "nhl", label: "NHL", path: "/nhl" },
  { key: "ncaa_baseball", label: "NCAA Baseball", path: "/ncaa-baseball" },
  { key: "euroleague", label: "EuroLeague", path: "/euroleague" },
  { key: "liga_mx", label: "Liga MX", path: "/liga-mx" },
  { key: "laliga", label: "LaLiga", path: "/laliga" },
  { key: "bundesliga", label: "Bundesliga", path: "/bundesliga" },
  { key: "live", label: "En Vivo", path: "/live" },
  { key: "resumen_dia", label: "Resumen del dia", path: "/resumen-dia" },
  { key: "estadisticas", label: "Estadisticas", path: "/estadisticas" },
  { key: "metricas", label: "Insights", path: "/insights" },
  { key: "weekday_scoring", label: "Weekday Scoring", path: "/weekday-scoring" },
  { key: "best_picks", label: "Best Picks", path: "/best-picks" },
];

export default function SportTabs() {
  const { socialMode } = useAppSettings();
  const sports = SPORTS.map((sport) => {
    if (!socialMode) return sport;
    if (sport.key === "best_picks") return { ...sport, label: "Radar" };
    if (sport.key === "weekday_scoring") return { ...sport, label: "Ritmo" };
    if (sport.key === "insights") return { ...sport, label: "Modelo" };
    if (sport.key === "resumen_dia") return { ...sport, label: "Resumen" };
    if (sport.key === "estadisticas") return { ...sport, label: "Metricas" };
    return sport;
  });

  return (
    <div className="flex flex-wrap gap-2">
      {sports.map((sport) => (
        <NavLink
          key={sport.key}
          to={sport.path}
          className={({ isActive }) =>
            `rounded-xl border px-4 py-2.5 text-sm font-semibold tracking-[0.01em] transition ${
              isActive
                ? "border-amber-300/70 bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#0f1218] shadow-[0_10px_24px_rgba(246,196,83,0.18)]"
                : "border-white/8 bg-white/[0.025] text-white/72 hover:border-white/16 hover:bg-white/[0.055] hover:text-white"
            }`
          }
        >
          {sport.label}
        </NavLink>
      ))}
    </div>
  );
}
