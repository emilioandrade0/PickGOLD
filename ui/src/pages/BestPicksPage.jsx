import { useEffect, useMemo, useState } from "react";
import SidebarCalendar from "../components/SidebarCalendar.jsx";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { dateToYMDLocal, formatDateInput } from "../utils/date.js";
import { fetchBestPicksAvailableDates, fetchBestPicksByDate, fetchBestPicksToday } from "../services/api.js";
import { getTeamDisplayName } from "../utils/teamNames.js";

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function tierClass(tier) {
  const t = String(tier || "").toUpperCase();
  if (t === "ELITE") return "border-emerald-400/40 bg-emerald-500/15 text-emerald-200";
  if (t === "PREMIUM") return "border-rose-400/40 bg-rose-500/15 text-rose-200";
  if (t === "STRONG") return "border-sky-400/40 bg-sky-500/15 text-sky-200";
  return "border-amber-400/40 bg-amber-500/15 text-amber-200";
}

function resultDotClass(resultLabel) {
  const label = String(resultLabel || "").toUpperCase();
  if (["ACIERTO", "CORRECTO"].includes(label)) return "bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.65)]";
  if (["FALLO", "INCORRECTO"].includes(label)) return "bg-rose-400 shadow-[0_0_16px_rgba(251,113,133,0.65)]";
  return "bg-amber-300 shadow-[0_0_14px_rgba(252,211,77,0.45)]";
}

function diagnosticItems(diag) {
  if (!diag || typeof diag !== "object") return [];
  return Object.entries(diag)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 6);
}

function PickCard({ row, compact = false, socialMode = false }) {
  const sportKey = String(row.sport || "").toLowerCase();
  const awayName = getTeamDisplayName(sportKey, row.away_team);
  const homeName = getTeamDisplayName(sportKey, row.home_team);
  const pickName = getTeamDisplayName(sportKey, row.pick);
  const gameName = row.game_name || `${awayName} vs ${homeName}`;
  const rawResultLabel = String(row.result_label || "PENDIENTE").toUpperCase();
  const resultLabel =
    rawResultLabel === "ACIERTO" && socialMode
      ? "Correcto"
      : rawResultLabel === "FALLO" && socialMode
        ? "Incorrecto"
        : row.result_label || "PENDIENTE";

  return (
    <article className="rounded-3xl border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))] p-4 shadow-[0_18px_60px_rgba(0,0,0,0.22)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.28em] text-white/45">{row.sport_label}</div>
          <div className="mt-1 text-sm text-white/60">{row.date || "-"}{row.time && row.time !== "nan" ? ` ? ${row.time}` : ""}</div>
        </div>
        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold ${tierClass(row.tier)}`}>
          {row.tier || "NORMAL"}
        </span>
      </div>

      <div className="mt-4">
        <div className="text-lg font-semibold text-white">{gameName}</div>
        <div className="mt-1 text-sm text-white/55">{socialMode ? "Modelo" : "Mercado"}: {row.market_label}</div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <div>
          <div className="text-[11px] uppercase tracking-[0.24em] text-white/40">{socialMode ? "Proyeccion destacada" : "Pick destacado"}</div>
          <div className="mt-1 text-xl font-semibold text-yellow-300">{pickName}</div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-right">
          <div className="text-[10px] uppercase tracking-[0.24em] text-white/40">{socialMode ? "Ventaja modelo" : "Edge"}</div>
          <div className="text-lg font-semibold text-white">{pct(row.edge_proxy)}</div>
        </div>
      </div>

      <div className={`mt-4 grid gap-3 ${compact ? "sm:grid-cols-2" : "sm:grid-cols-4"}`}>
        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/40">Score</div>
          <div className="mt-1 text-base font-semibold text-white">{fmt(row.score)}</div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/40">Rank final</div>
          <div className="mt-1 text-base font-semibold text-emerald-300">{fmt(row.final_rank_score, 3)}</div>
        </div>
        {!compact && (
          <>
            <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-white/40">{socialMode ? "Probabilidad modelo" : "Probabilidad"}</div>
              <div className="mt-1 text-base font-semibold text-white">{pct(row.model_probability)}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-white/40">{socialMode ? "Solidez" : "Confiabilidad"}</div>
              <div className="mt-1 text-base font-semibold text-white">x{fmt(row.reliability_factor)}</div>
            </div>
          </>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-black/20 px-3 py-2.5">
        <div className="flex items-center gap-2 text-sm text-white/75">
          <span className={`h-2.5 w-2.5 rounded-full animate-[pulse_1.8s_ease-in-out_infinite] ${resultDotClass(resultLabel)}`} />
          <span>{resultLabel}</span>
        </div>
        <div className="text-sm text-white/55">{row.status_description || row.status_state || (socialMode ? "Disponible" : "Programado")}</div>
      </div>
    </article>
  );
}

function SportSection({ section, socialMode }) {
  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.03] p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h4 className="text-xl font-semibold text-white">Mejores {section.sport_label}</h4>
          <p className="mt-1 text-sm text-white/60">{socialMode ? "Top curado del modelo para esa liga en la fecha seleccionada." : "Top curado de ese deporte para la fecha seleccionada."}</p>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="rounded-full border border-white/15 bg-white/5 px-3 py-1 text-white/70">{section.count} candidatos</span>
          <span className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-emerald-200">rank top {fmt(section.top_final_rank_score, 3)}</span>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {(section.picks || []).map((row, idx) => (
          <PickCard key={`${section.sport}-${row.market}-${row.game_id}-${idx}`} row={row} compact socialMode={socialMode} />
        ))}
      </div>
    </section>
  );
}

export default function BestPicksPage() {
  const { socialMode } = useAppSettings();
  const [bestPicksData, setBestPicksData] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const todayDate = useMemo(() => dateToYMDLocal(new Date()), []);

  function syncCalendarMonth(dateStr) {
    const d = new Date(`${dateStr}T00:00:00`);
    if (!Number.isNaN(d.getTime())) setCalendarMonth(d);
  }

  async function loadToday(forceRefresh = true) {
    const payload = await fetchBestPicksToday(30, "best_hit_rate", forceRefresh);
    setBestPicksData(payload);
    const snapDate = payload?.snapshot_date || todayDate;
    setSelectedDate(snapDate);
    syncCalendarMonth(snapDate);
    return snapDate;
  }

  async function loadByDate(dateStr) {
    const forceRefresh = dateStr === todayDate;
    const payload = await fetchBestPicksByDate(dateStr, 30, "best_hit_rate", forceRefresh);
    setBestPicksData(payload);
    const snapDate = payload?.snapshot_date || dateStr;
    setSelectedDate(snapDate);
    syncCalendarMonth(snapDate);
  }

  useEffect(() => {
    async function run() {
      try {
        setLoading(true);
        setError("");
        const [payload, dates] = await Promise.all([
          fetchBestPicksToday(30, "best_hit_rate", true),
          fetchBestPicksAvailableDates(),
        ]);
        setBestPicksData(payload);
        const snapDate = payload?.snapshot_date || todayDate;
        setSelectedDate(snapDate);
        syncCalendarMonth(snapDate);
        const normalized = Array.isArray(dates) ? [...new Set([...dates, snapDate])].sort() : [snapDate];
        setAvailableDates(normalized);
      } catch (err) {
        if (err?.name === "AbortError") {
          setError("La carga de Best Picks tardo demasiado. Intenta de nuevo.");
        } else {
          setError(err.message || "No se pudieron cargar los Best Picks del dia.");
        }
      } finally {
        setLoading(false);
      }
    }

    run();
  }, [todayDate]);

  async function handleSelectDate(dateStr) {
    try {
      setLoading(true);
      setError("");
      await loadByDate(dateStr);
    } catch (err) {
      setError(err.message || "No se pudieron cargar los Best Picks para esa fecha.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadToday() {
    try {
      setLoading(true);
      setError("");
      const snapDate = await loadToday(true);
      if (!availableDates.includes(snapDate)) {
        setAvailableDates((prev) => [...new Set([...(prev || []), snapDate])].sort());
      }
    } catch (err) {
      setError(err.message || "No se pudieron cargar los Best Picks del dia.");
    } finally {
      setLoading(false);
    }
  }

  const globalPicks = bestPicksData?.picks || [];
  const sportSections = bestPicksData?.top_by_sport || [];

  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      <section className="mb-6 rounded-[2rem] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(250,204,21,0.10),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02))] p-6 shadow-[0_30px_120px_rgba(0,0,0,0.35)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <div className="inline-flex items-center rounded-full border border-emerald-400/25 bg-emerald-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.28em] text-emerald-200">
              {socialMode ? "Radar multideporte" : "Best Picks multideporte"}
            </div>
            <h2 className="mt-4 text-3xl font-semibold text-white">{socialMode ? "Las proyecciones mas fuertes del dia, en una sola vista" : "Los picks mas fuertes del dia, en una sola vista"}</h2>
            <p className="mt-3 text-sm leading-7 text-white/68">
              {socialMode
                ? "Aqui encuentras primero el top global del modelo y luego el mejor bloque por deporte para decidir rapido."
                : "Aqui encuentras primero el top global entre todos los deportes y mercados, y luego el mejor bloque de cada deporte para decidir rapido."}
            </p>
          </div>
          <div className="grid min-w-[220px] gap-3 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.24em] text-white/45">{socialMode ? "Proyecciones globales" : "Picks globales"}</div>
              <div className="mt-1 text-2xl font-semibold text-white">{globalPicks.length}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.24em] text-white/45">Deportes activos</div>
              <div className="mt-1 text-2xl font-semibold text-white">{sportSections.length}</div>
            </div>
          </div>
        </div>
        {selectedDate && <p className="mt-4 text-xs text-white/50">Fecha consultada: {formatDateInput(selectedDate)}</p>}
      </section>

      {loading && <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-center text-white/75">{socialMode ? "Cargando radar del modelo..." : "Cargando Best Picks..."}</div>}
      {!loading && error && <div className="rounded-3xl border border-rose-400/50 bg-rose-500/10 p-8 text-center text-rose-100">{error}</div>}

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
            subtitle="Consulta snapshots diarios del top global y del top por deporte."
            todayButtonLabel="Recargar snapshot de hoy"
          />

          <section className="min-w-0 space-y-6">
            <section className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-xl font-semibold text-white">{socialMode ? "Top global del modelo" : "Top global del dia"}</h3>
                  <p className="mt-1 text-sm text-white/60">{socialMode ? "La seleccion mas fuerte del modelo entre todos los deportes disponibles." : "La seleccion mas fuerte entre todos los deportes disponibles."}</p>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full border border-white/15 bg-white/5 px-3 py-1 text-white/70">modelo {bestPicksData?.ranking_model || "v1"}</span>
                  <span className="rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-amber-200">{bestPicksData?.total_candidates ?? 0} candidatos</span>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                {globalPicks.map((row, idx) => (
                  <PickCard key={`${row.sport}-${row.market}-${row.game_id}-${idx}`} row={row} socialMode={socialMode} />
                ))}
              </div>

              {!globalPicks.length && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/65">
                  {socialMode ? "No encontramos proyecciones elegibles en esta fecha." : "No encontramos picks elegibles en esta fecha."}
                </div>
              )}
            </section>

            <section className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-xl font-semibold text-white">{socialMode ? "Mejores proyecciones por deporte" : "Mejores picks por deporte"}</h3>
                  <p className="mt-1 text-sm text-white/60">{socialMode ? "Cada bloque muestra la mejor seleccion del modelo para esa liga en la fecha elegida." : "Cada bloque muestra la mejor seleccion disponible de esa liga para la fecha elegida."}</p>
                </div>
              </div>
              <div className="space-y-5">
                {sportSections.map((section) => (
                  <SportSection key={section.sport} section={section} socialMode={socialMode} />
                ))}
              </div>
              {!sportSections.length && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/65">
                  {socialMode ? "Todavia no hay deportes con proyecciones elegibles para esta fecha." : "Todavia no hay deportes con picks elegibles para esta fecha."}
                </div>
              )}
            </section>

            <section className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <h3 className="text-lg font-semibold text-white">{socialMode ? "Diagnostico del modelo" : "Diagnostico de filtros"}</h3>
              <p className="mt-1 text-sm text-white/60">{socialMode ? "Aqui vemos cuando el filtro del modelo deja fuera demasiados deportes o selecciones." : "Aqui vemos cuando el filtro esta dejando fuera demasiados mercados o deportes."}</p>
              <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {diagnosticItems(bestPicksData?.gate_diagnostics).map(([key, value]) => (
                  <div key={key} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                    <div className="text-xs uppercase tracking-[0.22em] text-white/40">{key}</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
                  </div>
                ))}
              </div>
            </section>
          </section>
        </div>
      )}
    </main>
  );
}
