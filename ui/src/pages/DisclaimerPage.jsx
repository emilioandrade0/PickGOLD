import { useNavigate } from "react-router-dom";

const sections = [
  {
    title: "Naturaleza del servicio",
    paragraphs: [
      "PickGold es una plataforma de análisis deportivo que proporciona información, predicciones y herramientas de interpretación de eventos deportivos.",
    ],
    bullets: [
      "No es una casa de apuestas",
      "No acepta dinero para apostar",
      "No gestiona apuestas de ningún tipo",
    ],
  },
  {
    title: "Uso bajo responsabilidad del usuario",
    paragraphs: [
      "El usuario reconoce que cualquier decisión basada en la información proporcionada por PickGold se realiza bajo su propio criterio y riesgo.",
    ],
  },
  {
    title: "Sin garantía de resultados",
    paragraphs: [
      "PickGold no garantiza ganancias, rentabilidad ni aciertos en predicciones.",
      "Los resultados pueden variar y dependen de múltiples factores fuera del control de la plataforma.",
    ],
  },
  {
    title: "Riesgo en apuestas",
    paragraphs: [
      "Las apuestas deportivas implican riesgo financiero.",
      "El usuario acepta que puede perder dinero al utilizar información basada en predicciones.",
    ],
  },
  {
    title: "No asesoría financiera",
    paragraphs: [
      "El contenido de PickGold no constituye asesoría financiera, legal o de inversión.",
    ],
  },
  {
    title: "Uso responsable",
    paragraphs: ["Se recomienda al usuario:"],
    bullets: [
      "Apostar de manera responsable",
      "No utilizar dinero que no pueda permitirse perder",
    ],
  },
  {
    title: "Limitación de responsabilidad",
    paragraphs: ["PickGold no será responsable por:"],
    bullets: [
      "Pérdidas económicas",
      "Decisiones de apuesta",
      "Daños directos o indirectos derivados del uso de la plataforma",
    ],
  },
  {
    title: "Edad mínima",
    paragraphs: [
      "El usuario declara ser mayor de edad conforme a la legislación aplicable.",
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
];

export default function DisclaimerPage() {
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
          <div className="inline-flex items-center gap-2 rounded-full border border-rose-300/20 bg-rose-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-rose-100/85">
            Disclaimer legal
          </div>

          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-white">
            Apuestas y responsabilidad
          </h1>
          <p className="mt-4 text-base leading-7 text-white/62">
            Este aviso aclara el alcance del producto y los riesgos asociados al uso de información deportiva para toma de decisiones.
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
                </div>
              </section>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
