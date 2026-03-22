import { useEffect, useState } from "react";
import { fetchWeekdayScoringInsights } from "../services/api.js";

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export default function WeekdayScoringPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function run() {
      try {
        setLoading(true);
        setError("");
        const payload = await fetchWeekdayScoringInsights();
        setData(payload);
      } catch (err) {
        if (err?.name === "AbortError") {
          setError("La carga tardó demasiado. Intenta de nuevo en unos segundos.");
        } else {
          setError(err.message || "No se pudieron cargar los insights de scoring por dia.");
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
        <h2 className="text-2xl font-semibold">Altas y Bajas por Dia</h2>
        <p className="mt-2 text-sm text-white/70">
          Ranking por deporte para identificar que dias de la semana tienden a tener mayor o menor scoring.
        </p>
        {data?.generated_at && (
          <p className="mt-2 text-xs text-white/50">Actualizado: {data.generated_at}</p>
        )}
      </section>

      {loading && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/75">
          Cargando scoring por dia...
        </div>
      )}

      {!loading && error && (
        <div className="rounded-3xl border border-rose-400/50 bg-rose-500/10 p-8 text-center text-rose-100">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="space-y-6">
          {(data?.sports || []).map((sport) => (
            <article key={sport.sport} className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <div className="mb-4 grid gap-3 md:grid-cols-3">
                <div>
                  <h3 className="text-xl font-semibold">{sport.label}</h3>
                  <p className="text-xs text-white/60">Metrica: {sport.metric_label}</p>
                </div>
                <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 px-4 py-3 text-sm">
                  <div className="text-white/70">Dia mas alto</div>
                  <div className="text-base font-semibold text-emerald-200">
                    {sport.highest_day ? `${sport.highest_day.weekday} (${fmt(sport.highest_day.avg_total)})` : "-"}
                  </div>
                </div>
                <div className="rounded-2xl border border-sky-400/30 bg-sky-500/10 px-4 py-3 text-sm">
                  <div className="text-white/70">Dia mas bajo</div>
                  <div className="text-base font-semibold text-sky-200">
                    {sport.lowest_day ? `${sport.lowest_day.weekday} (${fmt(sport.lowest_day.avg_total)})` : "-"}
                  </div>
                </div>
              </div>

              <div className="mb-4 flex flex-wrap gap-3 text-xs text-white/70">
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">
                  Juegos: {sport.total_games}
                </span>
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">
                  Promedio global: {fmt(sport.overall_avg_total)}
                </span>
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">
                  Mediana global: {fmt(sport.overall_median_total)}
                </span>
              </div>

              {sport.weekday_breakdown?.length ? (
                <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-4">
                  <table className="w-full text-left text-sm">
                    <thead className="text-white/60">
                      <tr>
                        <th className="pb-2">Dia</th>
                        <th className="pb-2">Juegos</th>
                        <th className="pb-2">Promedio</th>
                        <th className="pb-2">Mediana</th>
                        <th className="pb-2">Alta %</th>
                        <th className="pb-2">Baja %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sport.weekday_breakdown.map((row) => (
                        <tr key={`${sport.sport}-${row.weekday}`} className="border-t border-white/10">
                          <td className="py-2 font-semibold">{row.weekday}</td>
                          <td className="py-2">{row.games}</td>
                          <td className="py-2">{fmt(row.avg_total)}</td>
                          <td className="py-2">{fmt(row.median_total)}</td>
                          <td className="py-2">{pct(row.high_rate)}</td>
                          <td className="py-2">{pct(row.low_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/65">
                  No hay datos suficientes para este deporte.
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
