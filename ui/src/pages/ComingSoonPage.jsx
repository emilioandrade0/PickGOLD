export default function ComingSoonPage({ sportName }) {
  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      <section className="rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl">
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex rounded-full border border-yellow-400/30 bg-yellow-400/10 px-4 py-2 text-sm font-medium text-yellow-300">
            Próximamente
          </div>

          <h2 className="mt-6 text-4xl font-semibold">{sportName} GOLD</h2>

          <p className="mt-4 text-base text-white/70">
            Esta sección ya quedó preparada con ruta propia. El siguiente paso será conectar su backend,
            calendario, predicciones históricas y tarjetas de partidos.
          </p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl bg-black/15 p-5 text-left">
              <p className="text-sm text-white/50">Estado</p>
              <p className="mt-2 text-lg font-semibold">UI preparada</p>
            </div>

            <div className="rounded-2xl bg-black/15 p-5 text-left">
              <p className="text-sm text-white/50">Backend</p>
              <p className="mt-2 text-lg font-semibold">Pendiente</p>
            </div>

            <div className="rounded-2xl bg-black/15 p-5 text-left">
              <p className="text-sm text-white/50">Modelos</p>
              <p className="mt-2 text-lg font-semibold">Pendiente</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}