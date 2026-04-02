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
    <div className="flex flex-wrap gap-3">
      {SPORTS.map((sport) => (
        <NavLink
          key={sport.key}
          to={sport.path}
          className={({ isActive }) =>
            `rounded-2xl border px-5 py-2.5 text-sm font-semibold tracking-[0.01em] transition ${
              isActive
                ? "border-amber-300 bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#0f1218] shadow-[0_10px_24px_rgba(246,196,83,0.28)]"
                : "border-white/10 bg-white/[0.045] text-white/78 hover:border-cyan-300/30 hover:bg-cyan-300/10 hover:text-cyan-100"
            }`
          }
        >
          {sport.label}
        </NavLink>
      ))}
    </div>
  );
}
