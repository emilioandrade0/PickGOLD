import { useNavigate } from "react-router-dom";
import PlanCheckoutActions from "../components/PlanCheckoutActions.jsx";
import { getActiveSession } from "../services/auth.js";
import { PLANS } from "../config/plans.js";

const SIGNALS = ["Best Picks diarios", "Multi-deporte", "Historial claro"];
const ACCESS_MATRIX = [
  { label: "Picks del dia", starter: true, pro: true, vip: true },
  { label: "Multi-deporte", starter: false, pro: true, vip: true },
  { label: "Picks premium", starter: false, pro: true, vip: true },
  { label: "Insights", starter: false, pro: true, vip: true },
];

const TELEGRAM_URL = (import.meta.env.VITE_TELEGRAM_URL || "").trim();

function PlanCard({ plan, onSelect }) {
  const rolePillClass = plan.featured
    ? "border-amber-300/30 bg-amber-300/12 text-amber-200"
    : "border-white/10 bg-white/[0.04] text-white/55";

  return (
    <div
      className={`rounded-[26px] border p-5 transition ${
        plan.featured
          ? "border-amber-300/28 bg-[linear-gradient(180deg,rgba(255,199,76,0.13),rgba(255,199,76,0.05))] shadow-[0_20px_50px_rgba(246,196,83,0.12)]"
          : "border-white/10 bg-white/[0.03]"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-white/38">{plan.note}</p>
          <h3 className="mt-2 text-[32px] font-semibold tracking-tight text-white">{plan.name}</h3>
        </div>
        <span className={`rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${rolePillClass}`}>
          {plan.role}
        </span>
      </div>

      <div className="mt-5 flex items-end gap-2">
        <p className="text-5xl font-semibold tracking-tight text-white">{plan.priceLabel}</p>
        <span className="pb-2 text-sm text-white/42">/ mes</span>
      </div>

      <div className="mt-5 space-y-3 text-sm text-white/70">
        {plan.features.map((feature) => (
          <div key={feature} className="flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span>{feature}</span>
          </div>
        ))}
      </div>

      <PlanCheckoutActions
        plan={plan}
        telegramUrl={TELEGRAM_URL}
        secondaryLabel={`Crear cuenta ${plan.name}`}
        onSecondary={onSelect}
      />
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const session = getActiveSession();

  return (
    <main className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.14),transparent_24%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.08),transparent_20%),linear-gradient(180deg,#07090f,#0c1017)] text-white">
      <div className="pointer-events-none absolute inset-0 opacity-90">
        <div className="absolute left-[-100px] top-[-80px] h-[360px] w-[360px] rounded-full bg-amber-300/8 blur-3xl" />
        <div className="absolute right-[-100px] top-4 h-[320px] w-[320px] rounded-full bg-cyan-300/6 blur-3xl" />
      </div>

      <section className="relative mx-auto max-w-6xl px-6 pb-16 pt-8">
        <div className="flex items-center justify-between gap-4">
          <button type="button" onClick={() => navigate("/")} className="text-3xl font-light tracking-tight text-white">
            PICK<span className="font-semibold text-amber-300">GOLD</span>
          </button>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-white/78 transition hover:border-white/18 hover:bg-white/[0.06]"
            >
              Acceder
            </button>
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-4 py-2 text-sm font-bold text-[#121722] shadow-[0_10px_26px_rgba(246,196,83,0.2)] transition hover:-translate-y-0.5 hover:brightness-105"
            >
              Registrate
            </button>
          </div>
        </div>

        <div className="mx-auto max-w-3xl pt-16 text-center">
          <h1 className="text-5xl font-light leading-[0.95] tracking-tight text-white sm:text-6xl lg:text-7xl">
            Picks premium.
          </h1>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-white/60 sm:text-lg">
            Reduce incertidumbre, obten ventaja y desbloquea picks con una progresion clara.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
            {SIGNALS.map((item) => (
              <span
                key={item}
                className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs font-medium text-white/68 sm:text-sm"
              >
                {item}
              </span>
            ))}
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-6 py-3.5 text-sm font-bold text-[#131821] shadow-[0_16px_38px_rgba(246,196,83,0.22)] transition hover:-translate-y-0.5 hover:brightness-105"
            >
              Registrate
            </button>
            <button
              type="button"
              onClick={() => navigate("/auth")}
              className="rounded-2xl border border-white/10 bg-white/[0.04] px-6 py-3.5 text-sm font-semibold text-white/84 transition hover:border-white/20 hover:bg-white/[0.07]"
            >
              Ver demo
            </button>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-6xl px-6 pb-24">
        <div className="mx-auto max-w-3xl text-center">
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/65">Membresias</p>
          <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Tres planes. Una progresion que convierte.
          </h2>
          <p className="mt-3 text-sm leading-6 text-white/56 sm:text-base">
            Starter para probar, Pro como ancla de conversion y VIP para quien va en serio. El cobro y la activacion se coordinan manualmente por Telegram.
          </p>
        </div>

        <div className="mt-10 grid gap-5 lg:grid-cols-3">
          {PLANS.map((plan) => (
            <PlanCard
              key={plan.key}
              plan={plan}
              onSelect={() => navigate(`/auth?plan=${encodeURIComponent(plan.key)}`)}
            />
          ))}
        </div>

        <div className="mt-12 rounded-[28px] border border-white/10 bg-white/[0.03] p-5 sm:p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/65">Planes</p>
              <h3 className="mt-3 text-2xl font-semibold text-white">
                Elige tu plan.
              </h3>
            </div>
            <p className="max-w-md text-sm leading-6 text-white/46">
              PickGold
            </p>
          </div>

          <div className="mt-8 overflow-hidden rounded-[22px] border border-white/8">
            <div className="grid grid-cols-[1.4fr_repeat(3,1fr)] bg-white/[0.03] text-sm font-semibold text-white/72">
              <div className="px-4 py-4">Feature</div>
              <div className="px-4 py-4 text-center">Starter</div>
              <div className="bg-amber-300/[0.08] px-4 py-4 text-center text-amber-100">Pro</div>
              <div className="px-4 py-4 text-center">VIP</div>
            </div>

            {ACCESS_MATRIX.map((row) => (
              <div key={row.label} className="grid grid-cols-[1.4fr_repeat(3,1fr)] border-t border-white/8 text-sm text-white/68">
                <div className="px-4 py-4">{row.label}</div>
                <div className="px-4 py-4 text-center">{row.starter ? "Si" : "No"}</div>
                <div className="bg-amber-300/[0.05] px-4 py-4 text-center text-amber-100">{row.pro ? "Si" : "No"}</div>
                <div className="px-4 py-4 text-center">{row.vip ? "Si" : "No"}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
