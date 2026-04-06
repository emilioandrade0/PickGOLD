export default function PlanCheckoutActions({
  plan,
  telegramUrl,
  secondaryLabel,
  onSecondary,
}) {
  return (
    <div className="mt-5 space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <a
          href={telegramUrl || undefined}
          target={telegramUrl ? "_blank" : undefined}
          rel={telegramUrl ? "noreferrer" : undefined}
          aria-disabled={!telegramUrl}
          className={`rounded-2xl px-4 py-3 text-center text-sm font-bold transition ${
            plan.featured
              ? "bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#131821] shadow-[0_16px_32px_rgba(246,196,83,0.22)]"
              : "bg-[#131821] text-amber-200"
          } ${telegramUrl ? "hover:brightness-105" : "cursor-not-allowed opacity-60"}`}
        >
          {telegramUrl ? plan.checkoutLabel || "Contactar por Telegram" : "Configurar Telegram"}
        </a>

        {secondaryLabel && onSecondary ? (
          <button
            type="button"
            onClick={onSecondary}
            className="rounded-2xl border border-white/10 bg-black/16 px-4 py-3 text-sm font-semibold text-white/76 transition hover:border-white/18 hover:bg-black/22"
          >
            {secondaryLabel}
          </button>
        ) : (
          <div
            className={`rounded-2xl border border-[#0070ba]/35 bg-[#0070ba]/12 px-4 py-3 text-center text-sm font-bold text-[#b9e3ff] transition ${
              telegramUrl ? "hover:bg-[#0070ba]/18" : "cursor-not-allowed opacity-60"
            }`}
          >
            {telegramUrl ? "Hablar por Telegram" : "Configurar Telegram"}
          </div>
        )}
      </div>

      <div
        className={`rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-white/62 ${
          plan.featured ? "border-amber-300/18 bg-amber-300/[0.04]" : ""
          }`}
      >
        El alta de cuentas se revisa manualmente despues del contacto por Telegram.
      </div>

      <p className="text-center text-[11px] leading-5 text-white/42">
        No garantizamos resultados. Uso bajo responsabilidad.
      </p>
    </div>
  );
}
