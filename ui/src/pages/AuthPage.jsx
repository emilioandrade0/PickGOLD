import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import PlanCheckoutActions from "../components/PlanCheckoutActions.jsx";
import { loginUser, registerUser, setActiveSession } from "../services/auth.js";
import { getPlanByKey, PLANS } from "../config/plans.js";
import {
  capturePaypalOrder,
  createPaypalOrder,
} from "../services/payments.js";

const VALUE_PILLS = ["Best Picks listos para vender", "Insights y scoring por dia", "Multi-deporte, multi-mercado"];
const TELEGRAM_URL = (import.meta.env.VITE_TELEGRAM_URL || "").trim();

export default function AuthPage({ onAuthenticated }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [loading, setLoading] = useState(false);
  const [paymentLoadingKey, setPaymentLoadingKey] = useState("");

  const isLogin = mode === "login";
  const selectedPlanKey = searchParams.get("plan") || "starter";
  const selectedPlan = useMemo(() => getPlanByKey(selectedPlanKey) || PLANS[0], [selectedPlanKey]);
  const handledPaymentRef = useRef("");

  function handlePlanSelect(planKey) {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("plan", planKey);
    setSearchParams(nextParams, { replace: true });
  }

  async function handleCheckout(planKey, provider) {
    try {
      setError("");
      setInfo("");
      setPaymentLoadingKey(`${planKey}:${provider}`);
      const response = await createPaypalOrder(planKey);
      window.location.href = response.url;
    } catch (checkoutError) {
      setError(checkoutError.message || "No se pudo iniciar el checkout.");
      setPaymentLoadingKey("");
    }
  }

  useEffect(() => {
    const provider = searchParams.get("provider");
    const paymentStatus = searchParams.get("payment");
    const planKey = searchParams.get("plan") || "starter";
    const paypalOrderId = searchParams.get("token");

    if (!provider || !paymentStatus) return;

    const paymentKey = `${provider}:${paymentStatus}:${paypalOrderId || planKey}`;
    if (handledPaymentRef.current === paymentKey) return;
    handledPaymentRef.current = paymentKey;

    async function syncPaymentStatus() {
      try {
        setError("");
        if (provider !== "paypal") {
          setInfo("Stripe quedo desactivado por ahora. Usa PayPal o Telegram para cerrar el cobro.");
          setMode("register");
          return;
        }

        if (paymentStatus === "cancel") {
          setInfo("El pago de PayPal fue cancelado. Puedes intentarlo otra vez.");
          setMode("register");
          return;
        }

        if (provider === "paypal" && paypalOrderId) {
          const result = await capturePaypalOrder(paypalOrderId);
          setInfo(
            result.paid
              ? `Pago confirmado con PayPal para ${result.plan_name}. Ahora registra al cliente para activacion manual.`
              : "PayPal devolvio una orden sin captura confirmada todavia."
          );
          setMode("register");
          return;
        }

        setInfo("Regresaste del checkout. Si ya se cobro, puedes continuar con el registro.");
        setMode("register");
      } catch (paymentError) {
        setError(paymentError.message || "No se pudo verificar el pago.");
        setMode("register");
      } finally {
        setPaymentLoadingKey("");
      }
    }

    syncPaymentStatus();
  }, [searchParams]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setInfo("");
    setLoading(true);
    const action = isLogin
      ? await loginUser({ email, password })
      : await registerUser({ name, email, password });
    setLoading(false);

    if (!action.ok) {
      setError(action.error || "No se pudo completar la accion.");
      return;
    }

    if (action.pending) {
      setInfo(action.message || "Tu cuenta sigue pendiente de aprobacion.");
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
        <div className="grid w-full gap-12 xl:grid-cols-[minmax(0,1fr)_460px] xl:items-center">
          <section>
            <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-300/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-200/90">
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
              Acceso premium
            </div>

            <h1 className="mt-6 max-w-4xl text-5xl font-light leading-[0.95] tracking-tight sm:text-6xl lg:text-7xl">
              NBA <span className="font-semibold text-amber-300">GOLD</span>
              <span className="mt-3 block text-white/96">compra confianza. entra con ventaja.</span>
            </h1>

            <p className="mt-6 max-w-2xl text-lg leading-8 text-white/68">
              Elige tu nivel, cobra con PayPal y usa Telegram como ruta directa para cerrar ventas manuales.
            </p>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-amber-100/70">
              Pro es el plan ancla de conversion. Stripe queda fuera por ahora para simplificar la operacion.
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

            <div className="mt-10 grid gap-4 lg:grid-cols-3">
              {PLANS.map((plan) => (
                <div
                  key={plan.key}
                  className={`rounded-[28px] border p-5 transition ${plan.accent} ${
                    selectedPlan.key === plan.key ? "ring-2 ring-amber-300/60" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{plan.caption}</p>
                      <h3 className="mt-2 text-2xl font-semibold text-white">{plan.name}</h3>
                    </div>
                    {plan.featured && (
                      <span className="rounded-full border border-amber-300/35 bg-amber-300/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-200">
                        Mas elegido
                      </span>
                    )}
                  </div>
                  <div className="mt-4 flex items-center justify-between">
                    <p className="text-4xl font-semibold text-white">
                      {plan.priceLabel}
                      <span className="text-base font-medium text-white/52"> / mes</span>
                    </p>
                    <span className="rounded-full border border-white/10 bg-black/18 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/60">
                      {plan.role}
                    </span>
                  </div>
                  <div className="mt-5 space-y-3 text-sm text-white/72">
                    {plan.features.map((feature) => (
                      <div key={feature} className="flex items-center gap-3">
                        <span className="h-2 w-2 rounded-full bg-emerald-400" />
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>

                  <PlanCheckoutActions
                    plan={plan}
                    loadingProvider={paymentLoadingKey === `${plan.key}:paypal` ? "paypal" : ""}
                    onPaypal={() => handleCheckout(plan.key, "paypal")}
                    telegramUrl={TELEGRAM_URL}
                    secondaryLabel={selectedPlan.key === plan.key ? "Plan seleccionado" : `Elegir ${plan.name}`}
                    onSecondary={() => {
                      handlePlanSelect(plan.key);
                      setMode("register");
                    }}
                  />
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

            <p className="mt-3 max-w-md text-sm leading-6 text-white/68">
              Si ya pagaste, registra al cliente aqui para dejar la cuenta lista. Si ya tiene acceso, solo inicia sesion.
            </p>
            <div className="mt-4 rounded-[24px] border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100/88">
              Plan seleccionado: <span className="font-semibold">{selectedPlan.name}</span> ({selectedPlan.priceLabel} MXN al mes).
            </div>

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
              Acceso privado con aprobacion manual para mantener el producto limpio y premium.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
