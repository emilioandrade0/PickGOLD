import { useNavigate } from "react-router-dom";

const CONTACT_EMAIL = import.meta.env.VITE_CONTACT_EMAIL || "pickgoldapp@gmail.com";

const sections = [
  {
    title: "1. Responsable del tratamiento de datos",
    paragraphs: [
      "PickGold es responsable del tratamiento de los datos personales que se recaben a través de la plataforma.",
      `Para cualquier asunto relacionado con privacidad, el usuario puede contactar a: ${CONTACT_EMAIL}`,
    ],
  },
  {
    title: "2. Datos personales que se recaban",
    paragraphs: ["Podemos recopilar los siguientes datos:"],
    bullets: [
      "Nombre o alias",
      "Correo electrónico",
      "Datos de pago",
      "Datos de uso dentro de la plataforma",
      "Dirección IP y datos técnicos del dispositivo",
    ],
  },
  {
    title: "Datos de pago",
    paragraphs: [
      "PickGold no almacena directamente información bancaria o de tarjetas.",
      "Los pagos con tarjeta se procesan mediante plataformas externas como Stripe o PayPal, las cuales gestionan de forma segura la información financiera del usuario.",
      "En caso de pagos mediante transferencia bancaria u otros medios manuales, el usuario podrá compartir comprobantes de pago a través de canales como Telegram u otros medios de contacto.",
      "Estos comprobantes serán utilizados únicamente para validar el acceso al servicio y no serán almacenados con fines distintos.",
    ],
  },
  {
    title: "3. Finalidad del uso de datos",
    paragraphs: ["Los datos se utilizan para:"],
    bullets: [
      "Gestionar el acceso a la plataforma",
      "Procesar pagos y suscripciones",
      "Mostrar contenido personalizado",
      "Mejorar la experiencia del usuario",
      "Enviar notificaciones relevantes (picks, actualizaciones, etc.)",
    ],
  },
  {
    title: "4. Transferencia de datos",
    paragraphs: ["PickGold puede compartir datos únicamente con:"],
    bullets: [
      "Proveedores de pago",
      "Servicios tecnológicos necesarios para operar la plataforma",
    ],
    closing: "No vendemos ni comercializamos datos personales.",
  },
  {
    title: "5. Derechos ARCO",
    paragraphs: ["El usuario tiene derecho a:"],
    bullets: [
      "Acceder a sus datos",
      "Rectificarlos",
      "Cancelarlos",
      "Oponerse a su uso",
    ],
    closing: `Para ejercer estos derechos, enviar solicitud a: ${CONTACT_EMAIL}`,
  },
  {
    title: "6. Seguridad de la información",
    paragraphs: [
      "Se implementan medidas razonables de seguridad para proteger los datos.",
      "Sin embargo, ningún sistema es completamente invulnerable.",
    ],
  },
  {
    title: "7. Uso de cookies",
    paragraphs: ["La plataforma puede utilizar cookies para:"],
    bullets: [
      "Mejorar navegación",
      "Analizar comportamiento",
      "Personalizar contenido",
    ],
    closing: "El usuario puede deshabilitarlas desde su navegador.",
  },
  {
    title: "8. Cambios al aviso",
    paragraphs: [
      "PickGold puede modificar este aviso en cualquier momento.",
      "Las actualizaciones se publicarán dentro de la plataforma.",
    ],
  },
  {
    title: "9. Legislación aplicable",
    paragraphs: ["Este aviso se rige por las leyes de México."],
  },
];

export default function PrivacyPage() {
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
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-100/80">
            Aviso de privacidad
          </div>

          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-white">
            Aviso de privacidad – PickGold
          </h1>
          <p className="mt-4 text-base leading-7 text-white/62">
            Este aviso explica cómo PickGold recopila, utiliza y protege la información personal de los usuarios.
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
