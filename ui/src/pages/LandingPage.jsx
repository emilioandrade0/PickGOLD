import { useNavigate } from "react-router-dom";
import { getActiveSession } from "../services/auth.js";

const SELLING_POINTS = [
  "Best Picks con foco de conversión",
  "Insights visuales para vender mejor el valor",
  "Dashboard premium multi-deporte",
];

const FEATURE_COLUMNS = [
  {
    title: "Lectura inmediata",
    text: "El usuario entiende picks, tiers y confianza en segundos. Menos friccion, mas deseo de compra.",
  },
  {
    title: "Autoridad visual",
    text: "Métricas, jerarquía y contraste premium para que el producto transmita control y valor.",
  },
  {
    title: "Pensado para monetizar",
    text: "La experiencia completa ayuda a empaquetar mejor suscripciones, upsells y acceso premium.",
  },
];

const PLAN_CARDS = [
  {
    name: "Starter",
    price: "$29",
    caption: "Entrada rápida",
    points: ["Board diario", "Calendario histórico", "Picks principales"],
  },
  {
    name: "Pro",
    price: "$79",
    caption: "Más vendido",
    points: ["Best Picks", "Insights avanzados", "Experiencia premium"],
    featured: true,
  },
  {
    name: "VIP",
    price: "$149",
    caption: "Escala completa",
    points: ["Todos los deportes", "Scoring + insights", "Mayor percepción de valor"],
  },
];

export default function LandingPage() {
  const navigate = useNavigate();
  const session = getActiveSession();

  return (
    <main className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.18),transparent_24%),radial-gradient(circle_at_top_right,rgba(45,212,191,0.16),transparent_22%),linear-gradient(180deg,#07090f,#11141d)] text-white">
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="absolute left-[-80px] top-[-20px] h-96 w-96 rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-[-70px] top-16 h-80 w-80 rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <section className="relative mx-auto max-w-7xl px-6 pb-16 pt-8">
        <div className="flex items-center justify-between gap-4">
          <div className="text-4xl font-light tracking-tight">
            NBA <span className="font-semibold text-amber-300">GOLD</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-2xl border border-white/12 bg-white/[0.05] px-5 py-3 text-sm font-semibold text-white/85 transition hover:border-white/25 hover:bg-white/[0.08]"
            >
              {session ? "Entrar al dashboard" : "Login"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/auth")}
              className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-5 py-3 text-sm font-bold text-[#121722] shadow-[0_12px_30px_rgba(246,196,83,0.28)] transition hover:-translate-y-0.5 hover:brightness-105"
            >
              Empezar ahora
            </button>
          </div>
        </div>

        <div className="mt-16 grid gap-10 xl:grid-cols-[minmax(0,1.15fr)_460px] xl:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/90">
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
              Premium sports intelligence
            </div>

            <h1 className="mt-6 max-w-5xl text-5xl font-light leading-none tracking-tight sm:text-6xl lg:text-7xl">
              Convierte predicciones deportivas en una experiencia
              <span className="block font-semibold text-amber-300"> premium, clara y lista para vender.</span>
            </h1>

            <p className="mt-6 max-w-3xl text-lg leading-8 text-white/72">
               Picks, historial, insights y jerarquía visual pensados para elevar percepción de valor.
               Tu producto deja de verse como dashboard y empieza a sentirse como suscripción premium.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              {SELLING_POINTS.map((pill) => (
                <span
                  key={pill}
                  className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-medium text-white/76"
                >
                  {pill}
                </span>
              ))}
            </div>

            <div className="mt-8 flex flex-wrap gap-4">
              <button
                type="button"
                onClick={() => navigate(session ? "/nba" : "/auth")}
                className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-6 py-4 text-base font-bold text-[#131821] shadow-[0_16px_38px_rgba(246,196,83,0.26)] transition hover:-translate-y-0.5 hover:brightness-105"
              >
                {session ? "Abrir dashboard premium" : "Crear acceso"}
              </button>
              <button
                type="button"
                onClick={() => navigate("/auth")}
                className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 px-6 py-4 text-base font-semibold text-cyan-100 transition hover:border-cyan-300/35 hover:bg-cyan-300/14"
              >
                Ver demo privada
              </button>
            </div>
          </div>

          <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(15,18,26,0.98))] p-6 shadow-[0_26px_70px_rgba(0,0,0,0.30)]">
            <div className="flex items-center justify-between">
              <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/60">
                Panel de conversión
              </span>
              <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                77.8% hit rate
              </span>
            </div>

            <div className="mt-5 grid grid-cols-3 gap-3">
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Premium</p>
                <p className="mt-3 text-3xl font-semibold text-amber-200">A+</p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mercados</p>
                <p className="mt-3 text-3xl font-semibold text-white">12</p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-4">
                <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Board</p>
                <p className="mt-3 text-3xl font-semibold text-cyan-200">Live</p>
              </div>
            </div>

            <div className="mt-4 rounded-[24px] border border-amber-300/18 bg-amber-300/10 p-5">
              <p className="text-[11px] uppercase tracking-[0.16em] text-amber-100/70">Narrativa de valor</p>
              <p className="mt-3 text-sm leading-6 text-white/78">
                 Menos ruido, más autoridad. Una interfaz que ayuda a cobrar mejor porque hace que el
                 producto se sienta premium desde el primer scroll.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-6 py-14">
        <div className="grid gap-5 lg:grid-cols-3">
          {FEATURE_COLUMNS.map((item) => (
            <div key={item.title} className="rounded-[28px] border border-white/10 bg-white/[0.04] p-6 shadow-[0_16px_40px_rgba(0,0,0,0.18)]">
              <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{item.title}</p>
              <h3 className="mt-3 text-2xl font-semibold text-white">{item.title}</h3>
              <p className="mt-4 text-sm leading-7 text-white/70">{item.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-6 pb-20 pt-6">
        <div className="mb-8 max-w-3xl">
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/75">Pricing psychology</p>
          <h2 className="mt-3 text-4xl font-semibold text-white">Estructura el producto para subir percepción y ticket.</h2>
          <p className="mt-4 text-base leading-7 text-white/70">
            Una landing pública separada te deja vender mejor el acceso. El dashboard privado hace el resto:
            retención, autoridad y upsell.
          </p>
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          {PLAN_CARDS.map((plan) => (
            <div
              key={plan.name}
              className={`rounded-[30px] border p-6 ${
                plan.featured
                  ? "border-amber-300/30 bg-[linear-gradient(180deg,rgba(255,199,76,0.14),rgba(255,199,76,0.05))] shadow-[0_20px_50px_rgba(246,196,83,0.18)]"
                  : "border-white/10 bg-white/[0.04]"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{plan.caption}</p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">{plan.name}</h3>
                </div>
                {plan.featured && (
                  <span className="rounded-full border border-amber-300/35 bg-amber-300/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-200">
                    Popular
                  </span>
                )}
              </div>

              <p className="mt-5 text-5xl font-semibold text-white">
                {plan.price}
                <span className="text-base font-medium text-white/52"> / mes</span>
              </p>

              <div className="mt-6 space-y-3 text-sm text-white/74">
                {plan.points.map((feature) => (
                  <div key={feature} className="flex items-center gap-3">
                    <span className="h-2 w-2 rounded-full bg-emerald-400" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>

              <button
                type="button"
                onClick={() => navigate("/auth")}
                className={`mt-8 w-full rounded-2xl px-5 py-3 text-sm font-bold transition ${
                  plan.featured
                    ? "bg-[#131821] text-amber-200 hover:bg-[#171d27]"
                    : "border border-white/12 bg-white/[0.05] text-white/88 hover:bg-white/[0.08]"
                }`}
              >
                Elegir {plan.name}
              </button>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
