import { useNavigate } from "react-router-dom";

const sections = [
  {
    title: "1. Uso del servicio",
    paragraphs: [
      "PickGold ofrece información, visualización de picks, historiales y herramientas de análisis deportivo con fines informativos.",
      "El usuario reconoce que cualquier decisión tomada con base en esta información es bajo su propia responsabilidad.",
    ],
  },
  {
    title: "2. Naturaleza del servicio",
    paragraphs: [
      "PickGold no es una casa de apuestas ni un asesor financiero.",
      "No se garantiza ningún resultado, rendimiento o beneficio económico. Las métricas, porcentajes y predicciones son estimaciones basadas en modelos y datos históricos.",
    ],
  },
  {
    title: "3. Acceso y membresías",
    paragraphs: [
      "El acceso puede depender de validación, plan activo y cumplimiento de pago.",
      "PickGold se reserva el derecho de suspender o cancelar cuentas por uso indebido, fraude o incumplimiento de estos términos.",
    ],
  },
  {
    title: "4. Pagos y suscripciones",
    paragraphs: [
      "Los pagos se procesan mediante plataformas externas como Stripe o PayPal.",
      "Las suscripciones pueden ser recurrentes.",
      "El usuario es responsable de cancelar su suscripción antes del siguiente ciclo de facturación.",
      "Salvo disposición contraria, no se realizan reembolsos una vez activado el acceso.",
    ],
  },
  {
    title: "Validación de pagos manuales",
    paragraphs: [
      "Los pagos realizados mediante transferencia bancaria están sujetos a verificación manual.",
      "El acceso a la plataforma se activará únicamente una vez confirmado el pago.",
      "PickGold no se hace responsable por errores en transferencias, datos incorrectos o pagos no identificados correctamente por el usuario.",
    ],
  },
  {
    title: "5. Activación del servicio",
    paragraphs: [
      "La activación puede no ser inmediata y puede requerir validación de datos.",
      "El usuario debe asegurarse de que su información coincida con la utilizada en el pago.",
    ],
  },
  {
    title: "6. Uso permitido",
    paragraphs: ["Queda prohibido:"],
    bullets: [
      "Compartir cuentas",
      "Revender acceso",
      "Automatizar extracción de datos",
      "Intentar vulnerar la plataforma",
    ],
    closing: "El incumplimiento puede resultar en suspensión sin previo aviso.",
  },
  {
    title: "7. Edad mínima",
    paragraphs: [
      "El usuario declara ser mayor de edad según la legislación aplicable en su país.",
    ],
  },
  {
    title: "8. Limitación de responsabilidad",
    paragraphs: [
      "PickGold no será responsable por pérdidas económicas, decisiones de apuesta o cualquier daño derivado del uso de la plataforma.",
    ],
  },
  {
    title: "9. Cambios en el servicio",
    paragraphs: [
      "PickGold puede modificar funciones, precios, contenido o condiciones en cualquier momento sin previo aviso.",
    ],
  },
  {
    title: "10. Jurisdicción",
    paragraphs: [
      "Estos términos se rigen por las leyes de México.",
    ],
  },
];

export default function TermsPage() {
  const navigate = useNavigate();

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(255,198,79,0.12),transparent_22%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.08),transparent_20%),linear-gradient(180deg,#080a10,#0d1118)] px-6 py-12 text-white">
      <div className="mx-auto max-w-4xl">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="mb-8 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-white/80 transition hover:border-white/18 hover:bg-white/[0.07]"
        >
          Volver
        </button>

        <div className="rounded-[32px] border border-white/10 bg-white/[0.03] p-8 shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
          <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-200/80">
            Términos y condiciones
          </div>

          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-white">
            Términos y condiciones de uso
          </h1>
          <p className="mt-4 text-base leading-7 text-white/62">
            Al utilizar PickGold, el usuario acepta las siguientes condiciones.
          </p>

          <div className="mt-10 space-y-8">
            {sections.map((section) => (
              <section key={section.title}>
                <h2 className="text-xl font-semibold text-white">{section.title}</h2>

                <div className="mt-3 space-y-3 text-sm leading-7 text-white/70">
                  {section.paragraphs.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}

                  {section.bullets ? (
                    <ul className="list-disc space-y-1 pl-6">
                      {section.bullets.map((bullet) => (
                        <li key={bullet}>{bullet}</li>
                      ))}
                    </ul>
                  ) : null}

                  {section.closing ? <p>{section.closing}</p> : null}
                </div>
              </section>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
