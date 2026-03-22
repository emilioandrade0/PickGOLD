import { useEffect, useState } from "react";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import { dateToYMDLocal, formatDateInput } from "../utils/date.js";
import { fetchBestPicksAvailableDates, fetchBestPicksByDate, fetchBestPicksToday } from "../services/api.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function tierClass(tier) {
  const t = String(tier || "").toUpperCase();
  if (t === "ELITE") return "border-emerald-400/40 bg-emerald-500/15 text-emerald-200";
  if (t === "PREMIUM") return "border-sky-400/40 bg-sky-500/15 text-sky-200";
  if (t === "STRONG") return "border-amber-400/40 bg-amber-500/15 text-amber-200";
  return "border-white/25 bg-white/10 text-white/80";
}

function diagnosticItems(diag) {
  if (!diag || typeof diag !== "object") return [];
  return Object.entries(diag)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 10);
}

export default function BestPicksPage() {
  const [hitRateData, setHitRateData] = useState(null);
  const [valueData, setValueData] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  function syncCalendarMonth(dateStr) {
    const d = new Date(`${dateStr}T00:00:00`);
    if (!Number.isNaN(d.getTime())) {
      setCalendarMonth(d);
    }
  }

  async function loadToday() {
    const [hitPayload, valuePayload] = await Promise.all([
      fetchBestPicksToday(30, "best_hit_rate"),
      fetchBestPicksToday(30, "best_ev_real_only"),
    ]);
    setHitRateData(hitPayload);
    setValueData(valuePayload);
    const snapDate = hitPayload?.snapshot_date || valuePayload?.snapshot_date || dateToYMDLocal(new Date());
    setSelectedDate(snapDate);
    syncCalendarMonth(snapDate);
    return snapDate;
  }

  async function loadByDate(dateStr) {
    const [hitPayload, valuePayload] = await Promise.all([
      fetchBestPicksByDate(dateStr, 30, "best_hit_rate"),
      fetchBestPicksByDate(dateStr, 30, "best_ev_real_only"),
    ]);
    setHitRateData(hitPayload);
    setValueData(valuePayload);
    const snapDate = hitPayload?.snapshot_date || valuePayload?.snapshot_date || dateStr;
    setSelectedDate(snapDate);
    syncCalendarMonth(snapDate);
  }

  useEffect(() => {
    async function run() {
      try {
        setLoading(true);
        setError("");

        const [hitPayload, valuePayload, dates] = await Promise.all([
          fetchBestPicksToday(30, "best_hit_rate"),
          fetchBestPicksToday(30, "best_ev_real_only"),
          fetchBestPicksAvailableDates(),
        ]);
        setHitRateData(hitPayload);
        setValueData(valuePayload);
        const todayDate = hitPayload?.snapshot_date || valuePayload?.snapshot_date || dateToYMDLocal(new Date());
        setSelectedDate(todayDate);
        syncCalendarMonth(todayDate);
        const normalized = Array.isArray(dates) ? [...new Set([...dates, todayDate])].sort() : [todayDate];
        setAvailableDates(normalized);
      } catch (err) {
        if (err?.name === "AbortError") {
          setError("La carga de mejores picks tardó demasiado. Intenta de nuevo.");
        } else {
          setError(err.message || "No se pudieron cargar los mejores picks del dia.");
        }
      } finally {
        setLoading(false);
      }
    }

    run();
  }, []);

  async function handleSelectDate(dateStr) {
    try {
      setLoading(true);
      setError("");
      await loadByDate(dateStr);
    } catch (err) {
      setError(err.message || "No se pudieron cargar los mejores picks para la fecha.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadToday() {
    try {
      setLoading(true);
      setError("");
      const todayDate = await loadToday();
      if (!availableDates.includes(todayDate)) {
        setAvailableDates((prev) => [...new Set([...(prev || []), todayDate])].sort());
      }
    } catch (err) {
      setError(err.message || "No se pudieron cargar los mejores picks del dia.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      <section className="mb-6 rounded-3xl border border-white/10 bg-white/5 p-6">
        <h2 className="text-2xl font-semibold">Best Picks Históricos</h2>
        <p className="mt-2 text-sm text-white/70">
          Snapshot diario congelado de best picks. Puedes revisar por fecha qué picks ganaron o perdieron.
        </p>
        {selectedDate && (
          <p className="mt-2 text-xs text-white/60">Fecha consultada: {formatDateInput(selectedDate)}</p>
        )}
        {hitRateData?.snapshot_generated_at && (
          <p className="mt-1 text-xs text-white/50">Snapshot creado: {hitRateData.snapshot_generated_at}</p>
        )}
      </section>

      {loading && (
        <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/75">
          Cargando mejores picks...
        </div>
      )}

      {!loading && error && (
        <div className="rounded-3xl border border-rose-400/50 bg-rose-500/10 p-8 text-center text-rose-100">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
          <SidebarCalendar
            calendarMonth={calendarMonth}
            setCalendarMonth={setCalendarMonth}
            selectedDate={selectedDate}
            loading={loading}
            error={error}
            onSelectDate={handleSelectDate}
            onLoadToday={handleLoadToday}
            availableDates={availableDates}
            title="Calendario Best Picks"
            subtitle="Consulta snapshots diarios congelados de picks."
            todayButtonLabel="Cargar snapshot de hoy"
          />

          <section className="min-w-0">
            <section className="mb-6 rounded-3xl border border-white/10 bg-black/20 p-5">
              <div className="mb-4 flex flex-wrap gap-3 text-xs text-white/70">
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">
                  Hit Rate picks: {hitRateData?.top_n ?? 0}
                </span>
                <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">
                  Value picks: {valueData?.top_n ?? 0}
                </span>
                <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-emerald-200">
                  Hit Rate: {hitRateData?.ranking_model || "v1"}
                </span>
                <span className="rounded-full border border-amber-400/40 bg-amber-500/10 px-3 py-1 text-amber-200">
                  Value: {valueData?.ranking_model || "v1"}
                </span>
              </div>

              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {(hitRateData?.sports_summary || []).map((s) => (
                  <div key={s.sport} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                    <div className="text-sm font-semibold">{s.label}</div>
                    <div className="mt-1 text-xs text-white/65">Picks en top: {s.count}</div>
                    <div className="text-xs text-white/65">Score max: {fmt(s.max_score)}</div>
                    <div className="text-xs text-white/65">Score medio: {fmt(s.avg_score)}</div>
                    <div className="text-xs text-white/65">Final rank max: {fmt(s.max_final_rank_score)}</div>
                    <div className="text-xs text-white/65">Final rank medio: {fmt(s.avg_final_rank_score)}</div>
                  </div>
                ))}
              </div>

              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="mb-2 text-sm font-semibold text-emerald-200">Diagnóstico Gates: Hit Rate</div>
                  <div className="grid gap-2 text-xs text-white/70">
                    {diagnosticItems(hitRateData?.gate_diagnostics).map(([k, v]) => (
                      <div key={`hit-${k}`} className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-1.5">
                        <span>{k}</span>
                        <span className="font-semibold text-white">{v}</span>
                      </div>
                    ))}
                    {!diagnosticItems(hitRateData?.gate_diagnostics).length && <div>Sin datos de diagnóstico.</div>}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="mb-2 text-sm font-semibold text-amber-200">Diagnóstico Gates: Value</div>
                  <div className="grid gap-2 text-xs text-white/70">
                    {diagnosticItems(valueData?.gate_diagnostics).map(([k, v]) => (
                      <div key={`val-${k}`} className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-1.5">
                        <span>{k}</span>
                        <span className="font-semibold text-white">{v}</span>
                      </div>
                    ))}
                    {!diagnosticItems(valueData?.gate_diagnostics).length && <div>Sin datos de diagnóstico.</div>}
                  </div>
                </div>
              </div>
            </section>

            <section className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <h3 className="mb-3 text-lg font-semibold">A) Best Picks Hit Rate</h3>
              {hitRateData?.picks?.length ? (
                <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-4">
                  <table className="w-full text-left text-sm">
                    <thead className="text-white/60">
                      <tr>
                        <th className="pb-2">#</th>
                        <th className="pb-2">Deporte</th>
                        <th className="pb-2">Juego</th>
                        <th className="pb-2">Mercado</th>
                        <th className="pb-2">Tier</th>
                        <th className="pb-2">Pick</th>
                        <th className="pb-2">Score</th>
                        <th className="pb-2">Final Rank</th>
                        <th className="pb-2">Edge</th>
                        <th className="pb-2">Riesgo</th>
                        <th className="pb-2">Correlación</th>
                        <th className="pb-2">Resultado</th>
                        <th className="pb-2">Estado</th>
                      </tr>
                    </thead>
                    <tbody>
                      {hitRateData.picks.map((row, idx) => {
                        const sportKey = String(row.sport || "").toLowerCase();
                        const awayName = getTeamDisplayName(sportKey, row.away_team);
                        const homeName = getTeamDisplayName(sportKey, row.home_team);
                        const fullGameName = row.game_name || `${awayName} @ ${homeName}`;
                        const pickName = getTeamDisplayName(sportKey, row.pick);
                        const resultClass =
                          row.result_hit === true
                            ? "text-emerald-300"
                            : row.result_hit === false
                              ? "text-rose-300"
                              : "text-white/70";

                        return (
                          <tr key={`${row.sport}-${row.market}-${row.game_id}-${idx}`} className="border-t border-white/10">
                            <td className="py-2 font-semibold">{idx + 1}</td>
                            <td className="py-2">{row.sport_label}</td>
                            <td className="py-2">
                              <div className="font-semibold">{fullGameName}</div>
                              <div className="text-xs text-white/60">
                                {row.date || "-"} {row.time && row.time !== "nan" ? `- ${row.time}` : ""}
                              </div>
                            </td>
                            <td className="py-2">{row.market_label}</td>
                            <td className="py-2">
                              <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${tierClass(row.tier)}`}>
                                {row.tier || "NORMAL"}
                              </span>
                            </td>
                            <td className="py-2 font-semibold text-yellow-300">{pickName}</td>
                            <td className="py-2">{fmt(row.score)}</td>
                            <td className="py-2 font-semibold text-emerald-300">{fmt(row.final_rank_score)}</td>
                            <td className="py-2">{pct(row.edge_proxy)}</td>
                            <td className="py-2">x{fmt(row.stability_factor)}</td>
                            <td className="py-2">x{fmt(row.correlation_penalty)}</td>
                            <td className={`py-2 font-semibold ${resultClass}`}>{row.result_label || "PENDIENTE"}</td>
                            <td className="py-2">{row.status_description || row.status_state || "Programado"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/65">
                  No hay picks elegibles para Hit Rate en esta fecha.
                </div>
              )}

              <h3 className="mb-3 mt-8 text-lg font-semibold">B) Best Picks Value</h3>
              {valueData?.picks?.length ? (
                <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-4">
                  <table className="w-full text-left text-sm">
                    <thead className="text-white/60">
                      <tr>
                        <th className="pb-2">#</th>
                        <th className="pb-2">Deporte</th>
                        <th className="pb-2">Juego</th>
                        <th className="pb-2">Mercado</th>
                        <th className="pb-2">Tier</th>
                        <th className="pb-2">Pick</th>
                        <th className="pb-2">Score</th>
                        <th className="pb-2">Final Rank</th>
                        <th className="pb-2">Edge</th>
                        <th className="pb-2">Riesgo</th>
                        <th className="pb-2">Correlación</th>
                        <th className="pb-2">Resultado</th>
                        <th className="pb-2">Estado</th>
                      </tr>
                    </thead>
                    <tbody>
                      {valueData.picks.map((row, idx) => {
                        const sportKey = String(row.sport || "").toLowerCase();
                        const awayName = getTeamDisplayName(sportKey, row.away_team);
                        const homeName = getTeamDisplayName(sportKey, row.home_team);
                        const fullGameName = row.game_name || `${awayName} @ ${homeName}`;
                        const pickName = getTeamDisplayName(sportKey, row.pick);
                        const resultClass =
                          row.result_hit === true
                            ? "text-emerald-300"
                            : row.result_hit === false
                              ? "text-rose-300"
                              : "text-white/70";

                        return (
                          <tr key={`${row.sport}-${row.market}-${row.game_id}-${idx}`} className="border-t border-white/10">
                            <td className="py-2 font-semibold">{idx + 1}</td>
                            <td className="py-2">{row.sport_label}</td>
                            <td className="py-2">
                              <div className="font-semibold">{fullGameName}</div>
                              <div className="text-xs text-white/60">
                                {row.date || "-"} {row.time && row.time !== "nan" ? `- ${row.time}` : ""}
                              </div>
                            </td>
                            <td className="py-2">{row.market_label}</td>
                            <td className="py-2">
                              <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${tierClass(row.tier)}`}>
                                {row.tier || "NORMAL"}
                              </span>
                            </td>
                            <td className="py-2 font-semibold text-yellow-300">{pickName}</td>
                            <td className="py-2">{fmt(row.score)}</td>
                            <td className="py-2 font-semibold text-amber-300">{fmt(row.final_rank_score)}</td>
                            <td className="py-2">{pct(row.edge_proxy)}</td>
                            <td className="py-2">x{fmt(row.stability_factor)}</td>
                            <td className="py-2">x{fmt(row.correlation_penalty)}</td>
                            <td className={`py-2 font-semibold ${resultClass}`}>{row.result_label || "PENDIENTE"}</td>
                            <td className="py-2">{row.status_description || row.status_state || "Programado"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/65">
                  No hay picks elegibles para Value en esta fecha.
                </div>
              )}
            </section>
          </section>
        </div>
      )}
    </main>
  );
}
