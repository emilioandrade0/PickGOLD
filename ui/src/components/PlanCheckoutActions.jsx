export default function PlanCheckoutActions({
  plan,
  loadingProvider,
  onPaypal,
  telegramUrl,
  secondaryLabel,
  onSecondary,
}) {
  return (
    <div className="mt-5 space-y-3">
      <button
        type="button"
        onClick={onPaypal}
        disabled={loadingProvider === "paypal"}
        className={`w-full rounded-2xl px-4 py-3 text-sm font-bold transition disabled:cursor-not-allowed disabled:opacity-60 ${
          plan.featured
            ? "bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#131821] shadow-[0_16px_32px_rgba(246,196,83,0.22)] hover:brightness-105"
            : "bg-[#131821] text-amber-200 hover:bg-[#171d27]"
        }`}
      >
        {loadingProvider === "paypal" ? "Abriendo PayPal..." : plan.checkoutLabel || "Pagar con PayPal"}
      </button>

      <div className="grid gap-3 sm:grid-cols-2">
        <a
          href={telegramUrl || undefined}
          target={telegramUrl ? "_blank" : undefined}
          rel={telegramUrl ? "noreferrer" : undefined}
          aria-disabled={!telegramUrl}
          className={`rounded-2xl border border-[#0070ba]/35 bg-[#0070ba]/12 px-4 py-3 text-sm font-bold text-[#b9e3ff] transition ${
            telegramUrl ? "hover:bg-[#0070ba]/18" : "cursor-not-allowed opacity-60"
          }`}
        >
          {telegramUrl ? "Pagar por Telegram" : "Configurar Telegram"}
        </a>
      </div>

      {secondaryLabel && onSecondary ? (
        <button
          type="button"
          onClick={onSecondary}
          className="w-full rounded-2xl border border-white/10 bg-black/16 px-4 py-3 text-sm font-semibold text-white/76 transition hover:border-white/18 hover:bg-black/22"
        >
          {secondaryLabel}
        </button>
      ) : null}

      <p className="text-center text-[11px] leading-5 text-white/42">
        No garantizamos resultados. Uso bajo responsabilidad.
      </p>
    </div>
  );
}
