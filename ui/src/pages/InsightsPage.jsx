import { useEffect, useState } from "react";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { fetchInsightsSummary, fetchTierPerformanceInsights } from "../services/api.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export default function InsightsPage() {
  const { socialMode } = useAppSettings();
  const [data, setData] = useState(null);
  const [tierData, setTierData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function run() {
      try {
        setLoading(true);
        setError("");
        const [payload, tierPayload] = await Promise.all([
          fetchInsightsSummary(),
          fetchTierPerformanceInsights(),
        ]);
        setData(payload);
        setTierData(tierPayload);
      } catch (err) {
        if (err?.name === "AbortError") {
          setError("La carga de insights tardó demasiado. Intenta refrescar en unos segundos.");
        } else {
          setError(err.message || "No se pudieron cargar los insights.");
        }
      } finally {
        setLoading(false);
      }
    }

    run();
  }, []);

  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      <section className="mb-6 rounded-3xl border border-white/10 bg-white/5 p-6">
        <h2 className="text-2xl font-semibold">{socialMode ? "Insights del modelo" : "Performance Insights"}</h2>
        <p className="mt-2 text-sm text-white/70">
          Equipos con mayor precisión en pick de ganador y mercados con mejor acertividad por deporte.
        </p>
        {data?.generated_at && (
          <p className="mt-2 text-xs text-white/50">Actualizado: {data.generated_at}</p>
        )}
      </section>

      {loading && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/75">
          {socialMode ? "Cargando insights del modelo..." : "Cargando insights..."}
        </div>
      )}

      {!loading && error && (
        <div className="rounded-3xl border border-rose-400/50 bg-rose-500/10 p-8 text-center text-rose-100">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="space-y-6">
          <article className="rounded-3xl border border-white/10 bg-black/20 p-5">
            <div className="mb-4 flex items-center justify-between gap-4">
              <h3 className="text-xl font-semibold">{socialMode ? "Performance por nivel del modelo" : "Performance por Tier (Full Game)"}</h3>
              {tierData?.generated_at && (
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs text-white/75">
                  Actualizado: {tierData.generated_at}
                </span>
              )}
            </div>

            <div className="grid gap-4 xl:grid-cols-3">
              {(tierData?.tiers || []).map((row) => (
                <section
                  key={row.tier}
                  className="rounded-2xl border border-white/10 bg-white/5 p-4"
                >
                  <h4 className="mb-3 text-base font-semibold text-yellow-300">{row.tier}</h4>
                  <div className="space-y-1 text-sm text-white/80">
                    <p>{socialMode ? `Casos: ${row.sample_size}` : `Muestra: ${row.sample_size}`}</p>
                    <p>{socialMode ? `Correcto: ${pct(row.accuracy)}` : `Acierto: ${pct(row.accuracy)}`}</p>
                    <p>{socialMode ? `Error modelo: ${pct(row.error_rate)}` : `Error: ${pct(row.error_rate)}`}</p>
                    <p>IC 95%: {pct(row.ci95_low)} - {pct(row.ci95_high)}</p>
                  </div>
                </section>
              ))}
            </div>

            {!!tierData?.by_sport?.length && (
              <div className="mt-4 overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-4">
                <table className="w-full text-left text-sm">
                  <thead className="text-white/60">
                    <tr>
                      <th className="pb-2">Tier</th>
                      <th className="pb-2">Deporte</th>
                      <th className="pb-2">Muestra</th>
                      <th className="pb-2">{socialMode ? "Correcto" : "Accuracy"}</th>
                      <th className="pb-2">Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tierData.by_sport.map((row) => (
                      <tr key={`${row.tier}-${row.sport}`} className="border-t border-white/10">
                        <td className="py-2 font-semibold">{row.tier}</td>
                        <td className="py-2">{row.label}</td>
                        <td className="py-2">{row.sample_size}</td>
                        <td className="py-2">{pct(row.accuracy)}</td>
                        <td className="py-2">{pct(row.error_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>

          {(data?.sports || []).map((sport) => (
            <article key={sport.sport} className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <div className="mb-4 flex items-center justify-between gap-4">
                <h3 className="text-xl font-semibold">{sport.label}</h3>
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs text-white/75">
                  Juegos evaluados: {sport.total_events}
                </span>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <section className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <h4 className="mb-3 text-sm font-semibold text-white/85">{socialMode ? "Top equipos mejor modelados" : "Top Equipos Mejor Predichos"}</h4>
                  {sport.team_insights?.length ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="text-white/60">
                          <tr>
                            <th className="pb-2">Equipo</th>
                            <th className="pb-2">Picks</th>
                            <th className="pb-2">{socialMode ? "Correctos" : "Aciertos"}</th>
                            <th className="pb-2">{socialMode ? "Correcto" : "Accuracy"}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sport.team_insights.map((row) => (
                            <tr key={`${sport.sport}-${row.team}`} className="border-t border-white/10">
                              <td className="py-2 font-semibold">{getTeamDisplayName(sport.sport, row.team)}</td>
                              <td className="py-2">{row.picks}</td>
                              <td className="py-2">{row.hits}</td>
                              <td className="py-2">{pct(row.accuracy)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-sm text-white/60">No hay suficiente muestra de equipos todavía.</p>
                  )}
                </section>

                <section className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <h4 className="mb-3 text-sm font-semibold text-white/85">{socialMode ? "Mercados con mejor consistencia" : "Mercados con Mejor Acertividad"}</h4>
                  {sport.market_insights?.length ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="text-white/60">
                          <tr>
                            <th className="pb-2">Mercado</th>
                            <th className="pb-2">Picks</th>
                            <th className="pb-2">{socialMode ? "Correctos" : "Aciertos"}</th>
                            <th className="pb-2">{socialMode ? "Correcto" : "Accuracy"}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sport.market_insights.map((row) => (
                            <tr key={`${sport.sport}-${row.market}`} className="border-t border-white/10">
                              <td className="py-2 font-semibold">{row.market}</td>
                              <td className="py-2">{row.picks}</td>
                              <td className="py-2">{row.hits}</td>
                              <td className="py-2">{pct(row.accuracy)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-sm text-white/60">No se encontraron mercados evaluables todavía.</p>
                  )}
                </section>
              </div>
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
