import { useState } from "react";
import { loginUser, registerUser, setActiveSession } from "../services/auth.js";

const VALUE_PILLS = [
  "Best Picks listos para vender",
  "Insights y scoring por día",
  "Multi-deporte, multi-mercado",
];

const SOCIAL_PROOF = [
  { label: "Hit rate visual", value: "77.8%" },
  { label: "Mercados activos", value: "12" },
  { label: "Lectura premium", value: "A+" },
];

const PLANS = [
  {
    name: "Starter",
    price: "$29",
    caption: "Entrada rápida",
    features: ["Board diario", "Historial por fecha", "Picks principales"],
    accent: "border-white/10 bg-white/[0.04]",
  },
  {
    name: "Pro",
    price: "$79",
    caption: "Más vendido",
    features: ["Best Picks", "Insights avanzados", "Jerarquía premium de picks"],
    accent: "border-amber-300/35 bg-[linear-gradient(180deg,rgba(255,199,76,0.14),rgba(255,199,76,0.05))] shadow-[0_18px_40px_rgba(246,196,83,0.18)]",
    featured: true,
  },
  {
    name: "VIP",
    price: "$149",
    caption: "Alta percepción de valor",
    features: ["Todos los deportes", "Paneles premium", "Experiencia orientada a conversión"],
    accent: "border-cyan-300/20 bg-cyan-300/[0.06]",
  },
];

export default function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [loading, setLoading] = useState(false);

  const isLogin = mode === "login";

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setInfo("");
    setLoading(true);
    let action;
    if (isLogin) {
      action = await loginUser({ email, password });
    } else {
      action = await registerUser({ name, email, password });
    }
    setLoading(false);
    if (!action.ok) {
      setError(action.error || "No se pudo completar la acción.");
      return;
    }

    if (action.pending) {
      setInfo(action.message || "Tu cuenta sigue pendiente de aprobación.");
      return;
    }

    if (!action.token) {
      setInfo(action.message || "Solicitud procesada correctamente.");
      return;
    }

    setActiveSession(action);
    onAuthenticated?.(action);
  }

  return (
    <div className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.18),transparent_26%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.14),transparent_24%),linear-gradient(180deg,#090b11,#11141c)] text-white">
      <div className="pointer-events-none absolute inset-0 opacity-80">
        <div className="absolute left-[-80px] top-[-20px] h-80 w-80 rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-[-40px] top-20 h-72 w-72 rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl items-center px-6 py-12">
        <div className="grid w-full gap-10 xl:grid-cols-[minmax(0,1.15fr)_460px] xl:items-center">
          <section>
            <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/90">
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
              Sports intelligence reimagined
            </div>

            <h1 className="mt-6 max-w-4xl text-5xl font-light leading-none tracking-tight sm:text-6xl lg:text-7xl">
              NBA <span className="font-semibold text-amber-300">GOLD</span>
              <span className="mt-3 block text-white/96">haz que tus picks se sientan premium, claros y vendibles.</span>
            </h1>

            <p className="mt-6 max-w-3xl text-lg leading-8 text-white/72">
              No es solo una app de predicciones. Es una experiencia diseñada para elevar percepción,
              confianza y conversión: picks premium, historial limpio, insights accionables y una interfaz
              que transmite valor desde el primer segundo.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              {VALUE_PILLS.map((pill) => (
                <span
                  key={pill}
                  className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-medium text-white/78"
                >
                  {pill}
                </span>
              ))}
            </div>

            <div className="mt-8 grid gap-4 md:grid-cols-3">
              {SOCIAL_PROOF.map((item) => (
                <div key={item.label} className="rounded-[24px] border border-white/10 bg-white/[0.04] p-5 shadow-[0_16px_40px_rgba(0,0,0,0.18)]">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{item.label}</p>
                  <p className="mt-3 text-4xl font-semibold text-white">{item.value}</p>
                </div>
              ))}
            </div>

            <div className="mt-8 grid gap-4 lg:grid-cols-3">
              {PLANS.map((plan) => (
                <div key={plan.name} className={`rounded-[28px] border p-5 ${plan.accent}`}>
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
                  <p className="mt-4 text-4xl font-semibold text-white">{plan.price}<span className="text-base font-medium text-white/52"> / mes</span></p>
                  <div className="mt-5 space-y-3 text-sm text-white/72">
                    {plan.features.map((feature) => (
                      <div key={feature} className="flex items-center gap-3">
                        <span className="h-2 w-2 rounded-full bg-emerald-400" />
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(16,19,27,0.98))] p-6 shadow-[0_26px_70px_rgba(0,0,0,0.28)] backdrop-blur-sm">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Acceso privado</p>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight">Entra a tu panel premium</h2>
              </div>
              <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                Live access
              </div>
            </div>

            <p className="mt-3 text-sm leading-6 text-white/68">
              Regístrate una vez, entra rápido y accede al board de picks, insights y mejores selecciones.
            </p>

            <div className="mt-5 grid grid-cols-2 gap-2 rounded-2xl bg-black/20 p-1.5">
              <button
                type="button"
                onClick={() => setMode("login")}
                className={`rounded-xl px-3 py-2.5 text-sm font-semibold transition ${
                  isLogin ? "bg-amber-300 text-[#131821] shadow-[0_10px_22px_rgba(246,196,83,0.22)]" : "text-white/65 hover:text-white"
                }`}
              >
                Login
              </button>
              <button
                type="button"
                onClick={() => setMode("register")}
                className={`rounded-xl px-3 py-2.5 text-sm font-semibold transition ${
                  !isLogin ? "bg-cyan-300 text-[#0c1620] shadow-[0_10px_22px_rgba(92,200,232,0.20)]" : "text-white/65 hover:text-white"
                }`}
              >
                Registro
              </button>
            </div>

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              {!isLogin && (
                <label className="block text-sm">
                  <span className="mb-2 block text-white/78">Nombre</span>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/55 focus:bg-black/24"
                    placeholder="Tu nombre"
                    required
                  />
                </label>
              )}

              <label className="block text-sm">
                <span className="mb-2 block text-white/78">Email</span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/55 focus:bg-black/24"
                  placeholder="correo@ejemplo.com"
                  required
                />
              </label>

              <label className="block text-sm">
                <span className="mb-2 block text-white/78">Password</span>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/55 focus:bg-black/24"
                  placeholder="********"
                  required
                />
              </label>

              {error && <p className="text-sm text-rose-300">{error}</p>}
              {info && <p className="text-sm text-amber-200">{info}</p>}

              <button
                type="submit"
                className="w-full rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-4 py-3 font-bold text-[#141821] shadow-[0_18px_34px_rgba(246,196,83,0.24)] transition hover:-translate-y-0.5 hover:brightness-105 disabled:opacity-60"
                disabled={loading}
              >
                {loading ? "Procesando..." : isLogin ? "Entrar al panel" : "Crear cuenta premium"}
              </button>
            </form>

            <div className="mt-6 rounded-[24px] border border-white/8 bg-white/[0.035] p-4 text-sm text-white/66">
              Tu acceso abre la puerta a una interfaz diseñada para que los picks se entiendan rápido,
              se vean premium y generen mayor percepción de valor.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
