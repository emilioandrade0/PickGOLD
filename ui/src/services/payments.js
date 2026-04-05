const PAYMENTS_BASE = (
  import.meta.env.VITE_AUTH_API_BASE ||
  import.meta.env.VITE_API_BASE ||
  "/api"
).trim().replace(/\/$/, "");

const PAYMENTS_FALLBACK_BASE = "http://127.0.0.1:8010/api";

function shouldUseLocalFallback() {
  if (!PAYMENTS_BASE) return true;
  return PAYMENTS_BASE.startsWith("/") || PAYMENTS_BASE.includes("127.0.0.1") || PAYMENTS_BASE.includes("localhost");
}

async function fetchWithFallback(path, options = undefined) {
  try {
    return await fetch(`${PAYMENTS_BASE}${path}`, options);
  } catch (error) {
    if (!shouldUseLocalFallback()) {
      throw error;
    }
  }
  return await fetch(`${PAYMENTS_FALLBACK_BASE}${path}`, options);
}

async function postJson(path, body = {}) {
  const res = await fetchWithFallback(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const payload = await res.json().catch(() => ({}));
  if (!res.ok || !payload?.ok) {
    throw new Error(payload?.error || "No se pudo iniciar el pago.");
  }
  return payload;
}

async function getJson(path) {
  const res = await fetchWithFallback(path);
  const payload = await res.json().catch(() => ({}));
  if (!res.ok || !payload?.ok) {
    throw new Error(payload?.error || "No se pudo verificar el pago.");
  }
  return payload;
}

export async function createStripeCheckout(planKey) {
  return await postJson("/payments/stripe/checkout-session", { plan_key: planKey });
}

export async function fetchStripeCheckoutStatus(sessionId) {
  return await getJson(`/payments/stripe/session-status?session_id=${encodeURIComponent(sessionId)}`);
}

export async function createPaypalOrder(planKey) {
  return await postJson("/payments/paypal/order", { plan_key: planKey });
}

export async function capturePaypalOrder(orderId) {
  return await postJson("/payments/paypal/capture-order", { order_id: orderId });
}
