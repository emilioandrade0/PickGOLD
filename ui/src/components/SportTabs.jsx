import { NavLink } from "react-router-dom";

const SPORTS = [
  { key: "nba", label: "NBA", path: "/nba" },
  { key: "mlb", label: "MLB", path: "/mlb" },
  { key: "tennis", label: "Tennis", path: "/tennis" },
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
    <div className="flex flex-wrap gap-2">
      {SPORTS.map((sport) => (
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
