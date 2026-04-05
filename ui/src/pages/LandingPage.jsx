import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PlanCheckoutActions from "../components/PlanCheckoutActions.jsx";
import { getActiveSession } from "../services/auth.js";
import { PLANS } from "../config/plans.js";
import { createPaypalOrder } from "../services/payments.js";

const SIGNALS = ["Best Picks diarios", "Multi-deporte", "Historial claro"];
const ACCESS_MATRIX = [
  { label: "Picks del dia", starter: true, pro: true, vip: true },
  { label: "Multi-deporte", starter: false, pro: true, vip: true },
  { label: "Picks premium", starter: false, pro: false, vip: true },
  { label: "Insights", starter: false, pro: true, vip: true },
];

const TELEGRAM_URL = (import.meta.env.VITE_TELEGRAM_URL || "").trim();

function PlanCard({ plan, onSelect, onPaypalCheckout, loadingProvider }) {
  return (
    <div
      className={`rounded-[32px] border p-6 transition ${
        plan.featured
          ? "border-amber-300/30 bg-[linear-gradient(180deg,rgba(255,199,76,0.12),rgba(255,199,76,0.04))] shadow-[0_24px_60px_rgba(246,196,83,0.14)]"
          : "border-white/10 bg-white/[0.035]"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-white/42">{plan.note}</p>
          <h3 className="mt-2 text-2xl font-semibold text-white">{plan.name}</h3>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
            plan.featured
              ? "border-amber-300/35 bg-amber-300/12 text-amber-200"
              : "border-white/12 bg-white/[0.04] text-white/60"
          }`}
        >
          {plan.role}
        </span>
      </div>

      <p className="mt-6 text-5xl font-semibold text-white">
        {plan.priceLabel}
        <span className="text-base font-medium text-white/45"> / mes</span>
      </p>

      <div className="mt-6 space-y-3 text-sm text-white/72">
        {plan.features.map((feature) => (
          <div key={feature} className="flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span>{feature}</span>
          </div>
        ))}
      </div>

      <PlanCheckoutActions
        plan={plan}
        loadingProvider={loadingProvider}
        onPaypal={onPaypalCheckout}
        telegramUrl={TELEGRAM_URL}
        secondaryLabel={`Ver acceso ${plan.name}`}
        onSecondary={onSelect}
      />
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const session = getActiveSession();
  const [loadingKey, setLoadingKey] = useState("");
  const [paymentError, setPaymentError] = useState("");

  async function handleCheckout(planKey, provider) {
    try {
      setPaymentError("");
      setLoadingKey(`${planKey}:${provider}`);
      const response = await createPaypalOrder(planKey);
      window.location.href = response.url;
    } catch (error) {
      setPaymentError(error.message || "No se pudo iniciar el checkout.");
      setLoadingKey("");
    }
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.12),transparent_22%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.07),transparent_20%),linear-gradient(180deg,#06080d,#0d1118)] text-white">
      <div className="pointer-events-none absolute inset-0 opacity-90">
        <div className="absolute left-[-120px] top-[-80px] h-[420px] w-[420px] rounded-full bg-amber-300/8 blur-3xl" />
        <div className="absolute right-[-120px] top-10 h-[360px] w-[360px] rounded-full bg-cyan-300/6 blur-3xl" />
      </div>

      <section className="relative mx-auto max-w-7xl px-6 pb-20 pt-8">
        <div className="flex items-center justify-between gap-4">
          <button type="button" onClick={() => navigate("/")} className="text-4xl font-light tracking-tight">
            NBA <span className="font-semibold text-amber-300">GOLD</span>
          </button>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-2xl border border-white/12 bg-white/[0.04] px-5 py-3 text-sm font-semibold text-white/82 transition hover:border-white/22 hover:bg-white/[0.07]"
            >
              {session ? "Entrar" : "Login"}
            </button>
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-5 py-3 text-sm font-bold text-[#121722] shadow-[0_12px_30px_rgba(246,196,83,0.24)] transition hover:-translate-y-0.5 hover:brightness-105"
            >
              Empezar
            </button>
          </div>
        </div>

        <div className="pt-24 text-center">
          <div className="mx-auto inline-flex items-center gap-2 rounded-full border border-amber-300/18 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/85">
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
            Picks premium
          </div>

          <h1 className="mx-auto mt-8 max-w-5xl text-5xl font-light leading-[0.95] tracking-tight text-white sm:text-6xl lg:text-7xl">
            Picks premium.
            <span className="block text-amber-300">Claros. Vendibles.</span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-white/66">
            Reduce incertidumbre, vende ventaja y desbloquea picks con una progresion clara.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            {SIGNALS.map((item) => (
              <span
                key={item}
                className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-medium text-white/74"
              >
                {item}
              </span>
            ))}
          </div>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <button
              type="button"
              onClick={() => navigate(session ? "/nba" : "/auth")}
              className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-7 py-4 text-base font-bold text-[#131821] shadow-[0_16px_38px_rgba(246,196,83,0.22)] transition hover:-translate-y-0.5 hover:brightness-105"
            >
              {session ? "Abrir dashboard" : "Crear acceso"}
            </button>
            <button
              type="button"
              onClick={() => navigate("/auth")}
              className="rounded-2xl border border-white/12 bg-white/[0.04] px-7 py-4 text-base font-semibold text-white/86 transition hover:border-white/22 hover:bg-white/[0.07]"
            >
              Ver demo
            </button>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-6 pb-24 pt-10">
        <div className="mx-auto mb-10 max-w-3xl text-center">
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/70">Membresias</p>
          <h2 className="mt-4 text-4xl font-semibold text-white">Tres planes. Una progresion que convierte.</h2>
          <p className="mt-4 text-base leading-7 text-white/64">
            Starter para probar, Pro como ancla de conversion y VIP para quien va en serio.
          </p>
          <p className="mt-3 text-sm leading-6 text-white/48">
            Precios mostrados en pesos mexicanos. PayPal es la ruta principal y Telegram queda como contacto alterno.
          </p>
          {paymentError ? <p className="mt-4 text-sm text-rose-300">{paymentError}</p> : null}
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          {PLANS.map((plan) => (
            <PlanCard
              key={plan.key}
              plan={plan}
              loadingProvider={loadingKey === `${plan.key}:paypal` ? "paypal" : ""}
              onPaypalCheckout={() => handleCheckout(plan.key, "paypal")}
              onSelect={() => navigate(`/auth?plan=${encodeURIComponent(plan.key)}`)}
            />
          ))}
        </div>

        <div className="mt-12 rounded-[32px] border border-white/10 bg-white/[0.035] p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/70">Que desbloquea cada plan</p>
              <h3 className="mt-3 text-2xl font-semibold text-white">El usuario entiende rapido por que subir de nivel.</h3>
            </div>
            <p className="max-w-md text-sm leading-6 text-white/50">
              Sin promesas falsas: solo acceso claro, ventaja visible y una escalera logica de valor.
            </p>
          </div>

          <div className="mt-8 overflow-hidden rounded-[24px] border border-white/8">
            <div className="grid grid-cols-[1.3fr_repeat(3,1fr)] bg-white/[0.03] text-sm font-semibold text-white/72">
              <div className="px-4 py-4">Feature</div>
              <div className="px-4 py-4 text-center">Starter</div>
              <div className="bg-amber-300/10 px-4 py-4 text-center text-amber-100">Pro</div>
              <div className="px-4 py-4 text-center">VIP</div>
            </div>

            {ACCESS_MATRIX.map((row) => (
              <div key={row.label} className="grid grid-cols-[1.3fr_repeat(3,1fr)] border-t border-white/8 text-sm text-white/70">
                <div className="px-4 py-4">{row.label}</div>
                <div className="px-4 py-4 text-center">{row.starter ? "Si" : "No"}</div>
                <div className="bg-amber-300/[0.06] px-4 py-4 text-center text-amber-100">{row.pro ? "Si" : "No"}</div>
                <div className="px-4 py-4 text-center">{row.vip ? "Si" : "No"}</div>
              </div>
            ))}
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            <div className="rounded-[24px] border border-white/8 bg-black/18 p-4">
              <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Escasez</p>
              <p className="mt-2 text-sm leading-6 text-white/74">Los picks premium se presentan limitados por nivel para empujar el upgrade sin saturar la pantalla.</p>
            </div>
            <div className="rounded-[24px] border border-amber-300/20 bg-amber-300/10 p-4">
              <p className="text-[11px] uppercase tracking-[0.16em] text-amber-100/70">Ancla</p>
              <p className="mt-2 text-sm leading-6 text-white/78">Pro queda al centro y con badge para dirigir la mayoria de conversiones hacia el ticket que mas te conviene.</p>
            </div>
            <div className="rounded-[24px] border border-cyan-300/18 bg-cyan-300/[0.05] p-4">
              <p className="text-[11px] uppercase tracking-[0.16em] text-cyan-100/70">Retencion</p>
              <p className="mt-2 text-sm leading-6 text-white/74">El retorno del checkout lleva directo a registro para que el acceso se active sin cambiar tu control manual actual.</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
