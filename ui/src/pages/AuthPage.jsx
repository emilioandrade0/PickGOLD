import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useSearchParams } from "react-router-dom";
import PlanCheckoutActions from "../components/PlanCheckoutActions.jsx";
import { loginUser, registerUser, setActiveSession } from "../services/auth.js";
import { getPlanByKey, PLANS } from "../config/plans.js";
import {
  capturePaypalOrder,
  createPaypalOrder,
} from "../services/payments.js";

const TELEGRAM_URL = (import.meta.env.VITE_TELEGRAM_URL || "").trim();

function AuthPlanCard({
  plan,
  selectedPlanKey,
  paymentLoadingKey,
  onPaypal,
  onSelect,
}) {
  const isSelected = selectedPlanKey === plan.key;
  const wrapperClass = plan.featured
    ? "border-amber-300/35 bg-[linear-gradient(180deg,rgba(255,199,76,0.14),rgba(255,199,76,0.05))] shadow-[0_20px_46px_rgba(246,196,83,0.14)]"
    : plan.key === "vip"
      ? "border-cyan-300/28 bg-cyan-300/[0.05]"
      : "border-white/10 bg-white/[0.03]";

  return (
    <div
      className={`rounded-[28px] border p-5 transition ${wrapperClass} ${isSelected ? "ring-1 ring-amber-300/55" : ""}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">{plan.caption}</p>
          <h3 className="mt-2 text-[18px] font-semibold text-white">{plan.name}</h3>
        </div>
        {plan.featured ? (
          <span className="rounded-full border border-amber-300/35 bg-amber-300/12 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-amber-200">
            Mas elegido
          </span>
        ) : (
          <span className="rounded-full border border-white/10 bg-black/18 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-white/60">
            {plan.role}
          </span>
        )}
      </div>

      <div className="mt-5 flex items-end gap-2">
        <p className="text-5xl font-semibold tracking-tight text-white">{plan.priceLabel}</p>
        <span className="pb-2 text-sm text-white/45">/ mes</span>
      </div>

      <div className="mt-6 space-y-4 text-sm text-white/74">
        {plan.features.map((feature) => (
          <div key={feature} className="flex items-center gap-3">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
            <span>{feature}</span>
          </div>
        ))}
      </div>

      <PlanCheckoutActions
        plan={plan}
        loadingProvider={paymentLoadingKey === `${plan.key}:paypal` ? "paypal" : ""}
        onPaypal={onPaypal}
        telegramUrl={TELEGRAM_URL}
        secondaryLabel={isSelected ? "Plan seleccionado" : `Elegir ${plan.name}`}
        onSecondary={onSelect}
      />
    </div>
  );
}

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
  const [acceptedLegal, setAcceptedLegal] = useState(false);

  const isLogin = mode === "login";
  const selectedPlanKey = searchParams.get("plan") || "starter";
  const selectedPlan = useMemo(() => getPlanByKey(selectedPlanKey) || PLANS[0], [selectedPlanKey]);
  const handledPaymentRef = useRef("");

  function handlePlanSelect(planKey) {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("plan", planKey);
    setSearchParams(nextParams, { replace: true });
  }

  async function handleCheckout(planKey) {
    try {
      setError("");
      setInfo("");
      setPaymentLoadingKey(`${planKey}:paypal`);
      const response = await createPaypalOrder(planKey);
      window.location.href = response.url;
    } catch (checkoutError) {
      setError(checkoutError.message || "No se pudo iniciar el pago.");
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
          setInfo("Por ahora el acceso premium se activa por PayPal o contacto directo.");
          setMode("register");
          return;
        }

        if (paymentStatus === "cancel") {
          setInfo("El pago fue cancelado. Puedes intentarlo otra vez.");
          setMode("register");
          return;
        }

        if (provider === "paypal" && paypalOrderId) {
          const result = await capturePaypalOrder(paypalOrderId);
          setInfo(
            result.paid
              ? `Pago confirmado para ${result.plan_name}. Ahora registra la cuenta para activacion manual.`
              : "La orden de PayPal aun no aparece como capturada."
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

    if (!isLogin && !acceptedLegal) {
      setError("Debes aceptar los términos y el aviso de privacidad para continuar.");
      return;
    }

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
    <div className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.16),transparent_24%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.08),transparent_18%),linear-gradient(180deg,#090b11,#0f141d)] text-white">
      <div className="pointer-events-none absolute inset-0 opacity-80">
        <div className="absolute left-[-90px] top-[-30px] h-[380px] w-[380px] rounded-full bg-amber-300/10 blur-3xl" />
        <div className="absolute right-[-50px] top-14 h-[320px] w-[320px] rounded-full bg-cyan-300/10 blur-3xl" />
      </div>

      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl items-center px-6 py-12">
        <div className="grid w-full gap-12 xl:grid-cols-[minmax(0,1fr)_460px] xl:items-start">
          <section>
            <div className="text-3xl font-light tracking-tight text-white sm:text-4xl">
              PICK<span className="font-semibold text-amber-300">GOLD</span>
            </div>

            <h1 className="mt-8 max-w-4xl text-5xl font-light leading-[0.95] tracking-tight text-white sm:text-6xl lg:text-7xl">
              compra confianza. entra
              <span className="block">con ventaja.</span>
            </h1>

            <p className="mt-6 text-2xl font-light text-white/84">
              Elige tu nivel.
            </p>

            <div className="mt-10 grid gap-4 lg:grid-cols-3">
              {PLANS.map((plan) => (
                <AuthPlanCard
                  key={plan.key}
                  plan={plan}
                  selectedPlanKey={selectedPlan.key}
                  paymentLoadingKey={paymentLoadingKey}
                  onPaypal={() => handleCheckout(plan.key)}
                  onSelect={() => {
                    handlePlanSelect(plan.key);
                    setMode("register");
                  }}
                />
              ))}
            </div>
          </section>

          <section className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(24,28,38,0.96),rgba(16,19,27,0.98))] p-6 shadow-[0_26px_70px_rgba(0,0,0,0.28)] backdrop-blur-sm">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-white/45">Acceso privado</p>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white">Entra a tu panel premium</h2>
            </div>

            <div className="mt-5 rounded-[24px] border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100/88">
              Plan seleccionado: <span className="font-semibold">{selectedPlan.name}</span> ({selectedPlan.priceLabel} MXN al mes).
            </div>

            <div className="mt-5 grid grid-cols-2 gap-2 rounded-2xl bg-black/20 p-1.5">
              <button
                type="button"
                onClick={() => setMode("login")}
                className={`rounded-xl px-3 py-3 text-sm font-semibold transition ${
                  isLogin ? "bg-amber-300 text-[#131821] shadow-[0_10px_22px_rgba(246,196,83,0.22)]" : "text-white/65 hover:text-white"
                }`}
              >
                Login
              </button>
              <button
                type="button"
                onClick={() => setMode("register")}
                className={`rounded-xl px-3 py-3 text-sm font-semibold transition ${
                  !isLogin ? "bg-white/[0.06] text-white" : "text-white/65 hover:text-white"
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
                    className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/45 focus:bg-black/22"
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
                  className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/45 focus:bg-black/22"
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
                  className="w-full rounded-2xl border border-white/12 bg-black/16 px-4 py-3 outline-none transition focus:border-cyan-300/45 focus:bg-black/22"
                  placeholder="********"
                  required
                />
              </label>

              {!isLogin && (
                <label className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-white/72">
                  <input
                    type="checkbox"
                    checked={acceptedLegal}
                    onChange={(e) => setAcceptedLegal(e.target.checked)}
                    className="mt-1 h-4 w-4 rounded border-white/20 bg-black/20"
                  />
                  <span>
                    Acepto los{" "}
                    <Link to="/terms" className="text-amber-200 underline underline-offset-4">
                      términos y condiciones
                    </Link>{" "}
                    y el{" "}
                    <Link to="/privacy" className="text-cyan-200 underline underline-offset-4">
                      aviso de privacidad
                    </Link>.
                  </span>
                </label>
              )}

              {error && <p className="text-sm text-rose-300">{error}</p>}
              {info && <p className="text-sm text-amber-200">{info}</p>}

              <button
                type="submit"
                className="w-full rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-4 py-3.5 font-bold text-[#141821] shadow-[0_18px_34px_rgba(246,196,83,0.24)] transition hover:-translate-y-0.5 hover:brightness-105 disabled:opacity-60"
                disabled={loading}
              >
                {loading ? "Procesando..." : isLogin ? "Entrar al panel" : "Crear cuenta premium"}
              </button>
            </form>

            <p className="mt-5 text-xs leading-6 text-white/45">
              No garantizamos resultados. Uso bajo responsabilidad.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
