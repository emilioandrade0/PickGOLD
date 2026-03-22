import { NavLink } from "react-router-dom";

const SPORTS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "kbo", label: "KBO", path: "/kbo" },
  { key: "nhl", label: "NHL", path: "/nhl" },
  { key: "ncaa_baseball", label: "NCAA Baseball", path: "/ncaa-baseball" },
  { key: "euroleague", label: "EuroLeague", path: "/euroleague" },
  { key: "liga_mx", label: "Liga MX", path: "/liga-mx" },
  { key: "laliga", label: "LaLiga", path: "/laliga" },
  { key: "insights", label: "Insights", path: "/insights" },
  { key: "weekday_scoring", label: "Weekday Scoring", path: "/weekday-scoring" },
  { key: "best_picks", label: "Best Picks", path: "/best-picks" },
];

export default function SportTabs() {
  return (
    <div className="flex flex-wrap gap-3">
      {SPORTS.map((sport) => (
        <NavLink
          key={sport.key}
          to={sport.path}
          className={({ isActive }) =>
            `rounded-xl border px-5 py-2 text-sm font-semibold transition ${
              isActive
                ? "border-amber-300 bg-amber-300/90 text-black shadow-lg shadow-amber-300/20"
                : "border-white/15 bg-white/5 text-white/80 hover:border-cyan-300/45 hover:bg-cyan-300/10 hover:text-cyan-100"
            }`
          }
        >
          {sport.label}
        </NavLink>
      ))}
    </div>
  );
}