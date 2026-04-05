const mxnFormatter = new Intl.NumberFormat("es-MX", {
  style: "currency",
  currency: "MXN",
  maximumFractionDigits: 0,
});

function formatPriceMXN(amount) {
  return mxnFormatter.format(amount);
}

const rawPlans = [
  {
    key: "starter",
    name: "Starter",
    role: "member",
    priceMxn: 99,
    caption: "Empieza a ganar con picks claros",
    note: "Para empezar",
    features: ["1 deporte", "Picks diarios basicos", "Historial por fecha"],
    checkoutLabel: "Probar ahora",
    accent: "border-white/10 bg-white/[0.04]",
  },
  {
    key: "pro",
    name: "Pro",
    role: "vip",
    priceMxn: 249,
    caption: "Donde empiezan los resultados reales",
    note: "Mas elegido",
    features: ["Multi-deporte", "Picks fuertes del dia", "Insights + scoring"],
    checkoutLabel: "Desbloquear picks",
    accent:
      "border-amber-300/35 bg-[linear-gradient(180deg,rgba(255,199,76,0.14),rgba(255,199,76,0.05))] shadow-[0_18px_40px_rgba(246,196,83,0.18)]",
    featured: true,
  },
  {
    key: "vip",
    name: "VIP",
    role: "capper",
    priceMxn: 499,
    caption: "Para usuarios que van en serio",
    note: "Acceso total",
    features: ["Todos los deportes", "Picks premium", "Mayor valor esperado"],
    checkoutLabel: "Acceso total",
    accent: "border-cyan-300/20 bg-cyan-300/[0.06]",
  },
];

export const PLANS = rawPlans.map((plan) => ({
  ...plan,
  priceLabel: formatPriceMXN(plan.priceMxn),
}));

export function getPlanByKey(planKey) {
  return PLANS.find((plan) => plan.key === planKey) || null;
}
