
import { useEffect, useMemo, useState } from "react";
import {
  analyzeProgoldTicket,
  extractProgoldCapture,
  fetchProgoldOcrStatus,
} from "../services/api.js";

const MAX_MATCHES = 14;

const DUMMY_ROWS = [
  { local: "America", pct_local: "44", pct_empate: "29", pct_visita: "27", visitante: "Tigres" },
  { local: "Pumas", pct_local: "35", pct_empate: "32", pct_visita: "33", visitante: "Monterrey" },
  { local: "Cruz Azul", pct_local: "51", pct_empate: "24", pct_visita: "25", visitante: "Toluca" },
  { local: "Guadalajara", pct_local: "39", pct_empate: "30", pct_visita: "31", visitante: "Pachuca" },
  { local: "Santos", pct_local: "33", pct_empate: "28", pct_visita: "39", visitante: "Leon" },
  { local: "Atlas", pct_local: "31", pct_empate: "34", pct_visita: "35", visitante: "Necaxa" },
  { local: "Juarez", pct_local: "42", pct_empate: "27", pct_visita: "31", visitante: "Queretaro" },
  { local: "Mazatlan", pct_local: "28", pct_empate: "30", pct_visita: "42", visitante: "San Luis" },
  { local: "Tijuana", pct_local: "37", pct_empate: "31", pct_visita: "32", visitante: "Puebla" },
  { local: "Lazio", pct_local: "41", pct_empate: "29", pct_visita: "30", visitante: "Roma" },
  { local: "Sevilla", pct_local: "29", pct_empate: "31", pct_visita: "40", visitante: "Barcelona" },
  { local: "Lille", pct_local: "35", pct_empate: "33", pct_visita: "32", visitante: "Lyon" },
  { local: "Napoli", pct_local: "46", pct_empate: "25", pct_visita: "29", visitante: "Milan" },
  { local: "Dortmund", pct_local: "43", pct_empate: "26", pct_visita: "31", visitante: "Leverkusen" },
];

const PATTERN_GLOSSARY = [
  { name: "partido_caotico", desc: "Tres porcentajes muy juntos.", tip: "Prefiere cobertura." },
  { name: "empate_vivo", desc: "La X compite real.", tip: "No descartes empate." },
  { name: "empate_ignorado_por_masa", desc: "La masa castiga la X.", tip: "Lectura contrarian." },
  { name: "favorito_sobrejugado", desc: "Favorito inflado por masa.", tip: "Evita all-in." },
  { name: "local_sobrepopular", desc: "Local cargado sin dominio.", tip: "Revisa 1X/X2." },
  { name: "visita_sobrecomprada_moderada", desc: "Visita lidera sin cerrar.", tip: "X2 mas estable." },
];

function createEmptyRows() {
  return Array.from({ length: MAX_MATCHES }, (_, idx) => ({
    partido: idx + 1,
    local: "",
    pct_local: "",
    pct_empate: "",
    pct_visita: "",
    visitante: "",
  }));
}

function cleanText(value) {
  return String(value || "").trim();
}

function toNullableNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return numeric;
}

function safeFixed(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return numeric.toFixed(2);
}

function semaforoClass(value) {
  if (value === "verde") return "border-emerald-300/40 bg-emerald-300/10 text-emerald-200";
  if (value === "amarillo") return "border-amber-300/45 bg-amber-300/10 text-amber-200";
  if (value === "rojo") return "border-rose-300/45 bg-rose-300/10 text-rose-200";
  return "border-white/15 bg-white/[0.05] text-white/70";
}

function cardClassBySemaforo(value) {
  if (value === "verde") return "border-emerald-300/35";
  if (value === "amarillo") return "border-amber-300/35";
  if (value === "rojo") return "border-rose-300/35";
  return "border-white/12";
}

function pickToneClass(value) {
  if (value === "alta") return "text-emerald-200";
  if (value === "media") return "text-amber-200";
  return "text-rose-200";
}

function scrollToSection(id) {
  const node = document.getElementById(id);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
}

function toCsvCell(value) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes("\n") || text.includes('"')) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function downloadText(filename, content, mimeType = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

function buildReportsCsv(reports) {
  const headers = ["Partido", "Local", "% Local", "% Empate", "% Visita", "Visitante", "Pick", "Doble", "Confianza", "Semaforo", "Tipo", "Alerta"];
  const lines = [headers.join(",")];
  for (const row of reports) {
    lines.push([
      row.partido,
      row.local,
      row.pct_local,
      row.pct_empate,
      row.pct_visita,
      row.visitante,
      `${row.pick_symbol || "-"} - ${row.recomendacion || "-"}`,
      row.doble_oportunidad || "-",
      row.confianza || "-",
      row.semaforo || "-",
      row.tipo_partido || "-",
      row.alerta || "-",
    ].map(toCsvCell).join(","));
  }
  return lines.join("\n");
}

function buildSessionJson({ jornadaNombre, rows, debugMode, viewMode }) {
  const payload = {
    jornada: jornadaNombre,
    view_mode: viewMode,
    debug_mode: debugMode,
    matches: rows.map((row) => ({
      partido: row.partido,
      local: cleanText(row.local),
      pct_local: toNullableNumber(row.pct_local),
      pct_empate: toNullableNumber(row.pct_empate),
      pct_visita: toNullableNumber(row.pct_visita),
      visitante: cleanText(row.visitante),
    })),
  };
  return JSON.stringify(payload, null, 2);
}

function normalizeImportedRows(matches) {
  const base = createEmptyRows();
  if (!Array.isArray(matches)) return base;
  matches.slice(0, MAX_MATCHES).forEach((row, idx) => {
    base[idx] = {
      partido: idx + 1,
      local: cleanText(row?.local),
      pct_local: row?.pct_local === null || row?.pct_local === undefined ? "" : String(row.pct_local),
      pct_empate: row?.pct_empate === null || row?.pct_empate === undefined ? "" : String(row.pct_empate),
      pct_visita: row?.pct_visita === null || row?.pct_visita === undefined ? "" : String(row.pct_visita),
      visitante: cleanText(row?.visitante),
    };
  });
  return base;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("No se pudo leer la captura."));
    reader.readAsDataURL(file);
  });
}

function rowHasAnyData(row) {
  if (!row || typeof row !== "object") return false;
  return Boolean(
    cleanText(row.local) ||
      cleanText(row.visitante) ||
      toNullableNumber(row.pct_local) !== null ||
      toNullableNumber(row.pct_empate) !== null ||
      toNullableNumber(row.pct_visita) !== null,
  );
}

function applyExtractedRows(sourceRows, extractedRows, startPartido = 1, onlyEmptyRows = false) {
  const next = Array.isArray(sourceRows)
    ? sourceRows.map((row, idx) => ({
      partido: idx + 1,
      local: cleanText(row?.local),
      pct_local: row?.pct_local === null || row?.pct_local === undefined ? "" : String(row.pct_local),
      pct_empate: row?.pct_empate === null || row?.pct_empate === undefined ? "" : String(row.pct_empate),
      pct_visita: row?.pct_visita === null || row?.pct_visita === undefined ? "" : String(row.pct_visita),
      visitante: cleanText(row?.visitante),
    }))
    : createEmptyRows();

  let cursor = Math.max(0, Math.min(MAX_MATCHES - 1, Number(startPartido || 1) - 1));
  let inserted = 0;

  for (const extracted of Array.isArray(extractedRows) ? extractedRows : []) {
    while (cursor < MAX_MATCHES && onlyEmptyRows && rowHasAnyData(next[cursor])) {
      cursor += 1;
    }
    if (cursor >= MAX_MATCHES) break;

    next[cursor] = {
      partido: cursor + 1,
      local: cleanText(extracted?.local),
      pct_local: extracted?.pct_local === null || extracted?.pct_local === undefined ? "" : String(extracted.pct_local),
      pct_empate: extracted?.pct_empate === null || extracted?.pct_empate === undefined ? "" : String(extracted.pct_empate),
      pct_visita: extracted?.pct_visita === null || extracted?.pct_visita === undefined ? "" : String(extracted.pct_visita),
      visitante: cleanText(extracted?.visitante),
    };
    inserted += 1;
    cursor += 1;
  }

  return { nextRows: next, inserted };
}

function PanelList({ title, items, renderItem, emptyText }) {
  return (
    <article className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
      <p className="text-xs uppercase tracking-[0.12em] text-white/55">{title}</p>
      {items.length ? items.map(renderItem) : <p className="mt-2 text-sm text-white/50">{emptyText}</p>}
    </article>
  );
}

export default function ProGoldPage() {
  const [rows, setRows] = useState(() => createEmptyRows());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [jornadaNombre, setJornadaNombre] = useState("Concurso Analitico - Jornada Actual");
  const [viewMode, setViewMode] = useState("boleto");
  const [debugMode, setDebugMode] = useState(false);

  const [historicalEnabled, setHistoricalEnabled] = useState(true);
  const [historicalMinRequired, setHistoricalMinRequired] = useState(8);
  const [historicalMaxDistance, setHistoricalMaxDistance] = useState(0.33);
  const [historicalTopK, setHistoricalTopK] = useState(40);

  const [importError, setImportError] = useState("");
  const [ocrStatus, setOcrStatus] = useState({ available: false, message: "Consultando OCR..." });
  const [ocrFiles, setOcrFiles] = useState([]);
  const [ocrSection, setOcrSection] = useState("progol");
  const [ocrStartPartido, setOcrStartPartido] = useState(1);
  const [ocrOnlyEmptyRows, setOcrOnlyEmptyRows] = useState(false);
  const [ocrBusy, setOcrBusy] = useState(false);
  const [ocrError, setOcrError] = useState("");
  const [ocrNotes, setOcrNotes] = useState([]);
  const [selectedPartido, setSelectedPartido] = useState(null);

  const [filterTipo, setFilterTipo] = useState("__all");
  const [filterConfianza, setFilterConfianza] = useState("__all");
  const [filterSemaforo, setFilterSemaforo] = useState("__all");
  const [searchTeam, setSearchTeam] = useState("");

  const reports = useMemo(
    () => (Array.isArray(result?.reports) ? result.reports.filter((row) => row?.estado !== "vacio") : []),
    [result],
  );

  const okReports = useMemo(
    () => reports.filter((row) => row?.estado === "ok"),
    [reports],
  );

  const summary = useMemo(() => {
    if (result?.summary) return result.summary;
    return {
      total_rows: MAX_MATCHES,
      evaluated_rows: okReports.length,
      picks_directos: okReports.filter((row) => row.apto_pick_directo).length,
      dobles: okReports.filter((row) => row.apto_doble_oportunidad).length,
      trampa: okReports.filter((row) => row.semaforo === "rojo").length,
      contrarian: okReports.filter((row) => row.sugerir_sorpresa).length,
    };
  }, [okReports, result]);

  const directosTop = useMemo(() => okReports.filter((row) => row.apto_pick_directo).slice(0, 4), [okReports]);
  const doblesTop = useMemo(() => okReports.filter((row) => row.apto_doble_oportunidad).slice(0, 4), [okReports]);
  const trampaTop = useMemo(() => okReports.filter((row) => row.semaforo === "rojo").slice(0, 4), [okReports]);
  const contrarianTop = useMemo(() => okReports.filter((row) => row.sugerir_sorpresa).slice(0, 4), [okReports]);

  const tipoOptions = useMemo(() => {
    const values = Array.from(new Set(okReports.map((row) => row.tipo_partido).filter(Boolean)));
    return values.sort((a, b) => String(a).localeCompare(String(b)));
  }, [okReports]);

  const confianzaOptions = useMemo(() => {
    const values = Array.from(new Set(okReports.map((row) => row.confianza).filter(Boolean)));
    return values.sort((a, b) => String(a).localeCompare(String(b)));
  }, [okReports]);

  const semaforoOptions = useMemo(() => {
    const values = Array.from(new Set(okReports.map((row) => row.semaforo).filter(Boolean)));
    return values.sort((a, b) => String(a).localeCompare(String(b)));
  }, [okReports]);

  const filteredReports = useMemo(() => {
    const needle = searchTeam.trim().toLowerCase();
    return okReports.filter((row) => {
      if (filterTipo !== "__all" && row.tipo_partido !== filterTipo) return false;
      if (filterConfianza !== "__all" && row.confianza !== filterConfianza) return false;
      if (filterSemaforo !== "__all" && row.semaforo !== filterSemaforo) return false;
      if (!needle) return true;
      return (
        String(row.local || "").toLowerCase().includes(needle) ||
        String(row.visitante || "").toLowerCase().includes(needle)
      );
    });
  }, [filterTipo, filterConfianza, filterSemaforo, searchTeam, okReports]);

  useEffect(() => {
    if (!filteredReports.length) {
      setSelectedPartido(null);
      return;
    }
    const stillExists = filteredReports.some((row) => row.partido === selectedPartido);
    if (!stillExists) {
      setSelectedPartido(filteredReports[0].partido);
    }
  }, [filteredReports, selectedPartido]);

  const selectedRow = useMemo(
    () => filteredReports.find((row) => row.partido === selectedPartido) || null,
    [filteredReports, selectedPartido],
  );

  useEffect(() => {
    let active = true;

    async function loadOcrStatus() {
      try {
        const payload = await fetchProgoldOcrStatus();
        if (!active) return;
        setOcrStatus({
          available: !!payload?.available,
          message: String(payload?.message || ""),
        });
      } catch {
        if (!active) return;
        setOcrStatus({
          available: false,
          message: "No se pudo consultar estado OCR.",
        });
      }
    }

    loadOcrStatus();
    return () => {
      active = false;
    };
  }, []);

  function handleRowChange(index, key, value) {
    setRows((current) => {
      const next = [...current];
      next[index] = { ...next[index], [key]: value };
      return next;
    });
  }

  function loadDummyRows() {
    const next = createEmptyRows();
    DUMMY_ROWS.forEach((row, index) => {
      next[index] = {
        partido: index + 1,
        local: row.local,
        pct_local: row.pct_local,
        pct_empate: row.pct_empate,
        pct_visita: row.pct_visita,
        visitante: row.visitante,
      };
    });
    setRows(next);
    setError("");
    setImportError("");
    setOcrError("");
  }

  function clearRows() {
    setRows(createEmptyRows());
    setResult(null);
    setError("");
    setImportError("");
    setOcrError("");
    setOcrNotes([]);
    setSelectedPartido(null);
  }

  async function runAnalysis() {
    try {
      setLoading(true);
      setError("");
      const payloadRows = rows.map((row) => ({
        partido: row.partido,
        local: cleanText(row.local),
        visitante: cleanText(row.visitante),
        pct_local: toNullableNumber(row.pct_local),
        pct_empate: toNullableNumber(row.pct_empate),
        pct_visita: toNullableNumber(row.pct_visita),
      }));
      const response = await analyzeProgoldTicket({ rows: payloadRows, debugMode });
      setResult(response);
    } catch (err) {
      setError(err?.message || "No se pudo ejecutar el analisis PROGOLD.");
    } finally {
      setLoading(false);
    }
  }

  async function handleImportSession(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      setRows(normalizeImportedRows(payload?.matches));
      setJornadaNombre(String(payload?.jornada || "Concurso Analitico - Jornada Actual"));
      setViewMode(payload?.view_mode === "analisis" ? "analisis" : "boleto");
      setDebugMode(Boolean(payload?.debug_mode));
      setImportError("");
      setError("");
    } catch {
      setImportError("No se pudo leer la sesion JSON.");
    } finally {
      event.target.value = "";
    }
  }

  function exportSessionJson() {
    const json = buildSessionJson({ jornadaNombre, rows, debugMode, viewMode });
    downloadText("progold_sesion.json", json, "application/json;charset=utf-8");
  }

  function exportReportsCsv() {
    if (!reports.length) return;
    const csv = buildReportsCsv(reports);
    downloadText("progold_tabla.csv", csv, "text/csv;charset=utf-8");
  }

  async function importFromCapture() {
    if (!ocrFiles.length) {
      setOcrError("Sube al menos una captura para iniciar el OCR.");
      return;
    }

    try {
      setOcrBusy(true);
      setOcrError("");
      const extractedRows = [];
      const notes = [];

      for (const file of ocrFiles) {
        const imageBase64 = await fileToDataUrl(file);
        const payload = await extractProgoldCapture({
          imageBase64,
          section: ocrSection,
          maxMatches: MAX_MATCHES,
        });

        if (!payload?.ok) {
          const message = String(payload?.error || "OCR no pudo procesar la captura.");
          notes.push(`${file.name}: ${message}`);
          continue;
        }

        const rowsFromFile = Array.isArray(payload?.rows) ? payload.rows : [];
        extractedRows.push(...rowsFromFile);

        const fileNotes = Array.isArray(payload?.notes) ? payload.notes : [];
        if (fileNotes.length) {
          for (const note of fileNotes.slice(0, 5)) {
            notes.push(`${file.name}: ${String(note)}`);
          }
        } else {
          notes.push(`${file.name}: ${rowsFromFile.length} filas detectadas.`);
        }
      }

      if (!extractedRows.length) {
        setOcrError("No se detectaron filas validas desde la captura.");
        setOcrNotes(notes);
        return;
      }

      const { nextRows, inserted } = applyExtractedRows(
        rows,
        extractedRows,
        ocrStartPartido,
        ocrOnlyEmptyRows,
      );

      setRows(nextRows);
      setOcrNotes([
        `Importacion lista: ${inserted} partidos cargados.`,
        ...notes.slice(0, 10),
      ]);
      setOcrFiles([]);
      if (inserted === 0) {
        setOcrError("No hubo espacio disponible para insertar filas con la configuracion actual.");
      } else {
        setOcrError("");
      }
    } catch {
      setOcrError("Error al procesar captura OCR.");
    } finally {
      setOcrBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-[1880px] px-4 py-6 xl:px-6 2xl:px-8">
      <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="lg:sticky lg:top-24 lg:self-start">
          <section className="rounded-2xl border border-white/12 bg-[linear-gradient(180deg,rgba(8,16,31,0.97),rgba(5,11,23,0.98))] p-4 shadow-[0_18px_45px_rgba(2,8,20,0.45)]">
            <div className="rounded-xl border border-cyan-300/30 bg-cyan-400/10 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100">Progol Contrarian Lab</div>
            <div className="mt-4 space-y-2">
              <button type="button" onClick={() => scrollToSection("progold-header")} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-3 py-2 text-left text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]">Header y jornada</button>
              <button type="button" onClick={() => scrollToSection("progold-controls")} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-3 py-2 text-left text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]">Controles de carga</button>
              <button type="button" onClick={() => scrollToSection("progold-memory")} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-3 py-2 text-left text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]">Memoria historica</button>
              <button type="button" onClick={() => scrollToSection("progold-editor")} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-3 py-2 text-left text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]">Editor boleto</button>
              <button type="button" onClick={() => scrollToSection("progold-output")} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-3 py-2 text-left text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]">Salida analitica</button>
            </div>
            <div className="mt-5 rounded-xl border border-white/12 bg-white/[0.03] p-3">
              <label className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Vista</label>
              <select value={viewMode} onChange={(event) => setViewMode(event.target.value)} className="w-full rounded-lg border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50">
                <option value="boleto">Vista boleto</option>
                <option value="analisis">Vista analisis</option>
              </select>
              <label className="mt-3 flex items-center gap-2 text-sm text-white/80">
                <input type="checkbox" checked={debugMode} onChange={(event) => setDebugMode(event.target.checked)} className="h-4 w-4 rounded border-white/25 bg-transparent" />
                Modo debug
              </label>
            </div>
            <div className="mt-4 grid gap-2">
              <button type="button" onClick={runAnalysis} disabled={loading} className="rounded-lg border border-cyan-300/40 bg-cyan-400/12 px-3 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/18 disabled:cursor-not-allowed disabled:opacity-60">{loading ? "Analizando..." : "Analizar boleto"}</button>
              <button type="button" onClick={loadDummyRows} className="rounded-lg border border-white/15 bg-white/[0.05] px-3 py-2 text-sm font-semibold text-white/80 transition hover:bg-white/[0.09]">Cargar dummy</button>
              <button type="button" onClick={clearRows} className="rounded-lg border border-rose-300/40 bg-rose-400/10 px-3 py-2 text-sm font-semibold text-rose-100 transition hover:bg-rose-400/16">Limpiar jornada</button>
            </div>
          </section>
        </aside>

        <div className="space-y-5">
          <section id="progold-header" className="rounded-[26px] border border-white/12 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.15),transparent_33%),radial-gradient(circle_at_top_right,rgba(16,185,129,0.10),transparent_30%),linear-gradient(180deg,rgba(14,24,42,0.97),rgba(7,12,24,0.98))] p-5 shadow-[0_24px_60px_rgba(2,8,20,0.50)]">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-200/80">Integracion PRO-TIP</p>
                <h2 className="mt-1 text-2xl font-semibold text-white md:text-3xl">PROGOLD Contrarian Lab</h2>
                <p className="mt-1 text-sm text-white/70">Layout inspirado en tu app original, corriendo dentro de PickGold en una sola instancia.</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={exportSessionJson} className="rounded-xl border border-white/18 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white/80 transition hover:bg-white/[0.1]">Guardar sesion JSON</button>
                <button type="button" onClick={runAnalysis} disabled={loading} className="rounded-xl border border-cyan-300/40 bg-cyan-400/12 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/18 disabled:cursor-not-allowed disabled:opacity-60">{loading ? "Analizando..." : "Analizar boleto"}</button>
              </div>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <article className="rounded-xl border border-emerald-300/35 bg-emerald-400/10 p-3"><p className="text-xs uppercase tracking-[0.12em] text-emerald-100/80">Picks directos</p><p className="mt-1 text-3xl font-semibold text-emerald-100">{summary.picks_directos || 0}</p><p className="text-xs text-emerald-100/70">Partidos aptos para 1, X o 2</p></article>
              <article className="rounded-xl border border-cyan-300/35 bg-cyan-400/10 p-3"><p className="text-xs uppercase tracking-[0.12em] text-cyan-100/80">Doble oportunidad</p><p className="mt-1 text-3xl font-semibold text-cyan-100">{summary.dobles || 0}</p><p className="text-xs text-cyan-100/70">Cobertura sugerida</p></article>
              <article className="rounded-xl border border-rose-300/35 bg-rose-400/10 p-3"><p className="text-xs uppercase tracking-[0.12em] text-rose-100/85">Partidos trampa</p><p className="mt-1 text-3xl font-semibold text-rose-100">{summary.trampa || 0}</p><p className="text-xs text-rose-100/70">Riesgo alto por sesgo de masa</p></article>
              <article className="rounded-xl border border-amber-300/35 bg-amber-400/10 p-3"><p className="text-xs uppercase tracking-[0.12em] text-amber-100/85">Contrarian</p><p className="mt-1 text-3xl font-semibold text-amber-100">{summary.contrarian || 0}</p><p className="text-xs text-amber-100/70">Posible valor en sorpresa</p></article>
            </div>
            {error ? <p className="mt-4 rounded-lg border border-rose-300/45 bg-rose-400/12 px-3 py-2 text-sm text-rose-100">{error}</p> : null}
          </section>

          <section id="progold-controls" className="rounded-2xl border border-white/12 bg-[linear-gradient(180deg,rgba(13,21,36,0.96),rgba(8,13,25,0.98))] p-4">
            <h3 className="text-lg font-semibold text-white">Controles de jornada y carga</h3>
            <p className="mt-1 text-sm text-white/65">Replica de bloques de configuracion de PRO-TIP.</p>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
                <label className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Nombre jornada</label>
                <input value={jornadaNombre} onChange={(event) => setJornadaNombre(event.target.value)} className="w-full rounded-lg border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50" />
              </div>
              <div className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Carga rapida</p>
                <div className="mt-2 grid gap-2">
                  <button type="button" onClick={loadDummyRows} className="rounded-lg border border-white/15 bg-white/[0.06] px-3 py-1.5 text-sm font-semibold text-white/80 transition hover:bg-white/[0.1]">Cargar datos dummy</button>
                  <button type="button" onClick={clearRows} className="rounded-lg border border-rose-300/35 bg-rose-400/10 px-3 py-1.5 text-sm font-semibold text-rose-100 transition hover:bg-rose-400/16">Limpiar jornada</button>
                </div>
              </div>
              <div className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Sesion</p>
                <label className="mt-2 block rounded-lg border border-white/16 bg-black/20 px-2 py-2 text-xs text-white/70">
                  Cargar sesion JSON
                  <input type="file" accept="application/json" onChange={handleImportSession} className="mt-2 block w-full text-xs" />
                </label>
                {importError ? <p className="mt-2 text-xs text-rose-200">{importError}</p> : null}
              </div>
              <div className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Exportacion</p>
                <div className="mt-2 grid gap-2">
                  <button type="button" onClick={exportSessionJson} className="rounded-lg border border-cyan-300/35 bg-cyan-400/10 px-3 py-1.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/16">Guardar sesion JSON</button>
                  <button type="button" onClick={exportReportsCsv} disabled={!reports.length} className="rounded-lg border border-emerald-300/35 bg-emerald-400/10 px-3 py-1.5 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/16 disabled:cursor-not-allowed disabled:opacity-60">Exportar tabla CSV</button>
                </div>
              </div>
            </div>
            <div className="mt-4 rounded-xl border border-white/12 bg-white/[0.04] p-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/55">Importar desde captura (OCR)</p>
              <p className={`mt-1 text-xs ${ocrStatus.available ? "text-emerald-200/80" : "text-amber-200/80"}`}>{ocrStatus.message}</p>

              <label className="mt-2 block rounded-lg border border-white/16 bg-black/20 px-2 py-2 text-xs text-white/70">
                Captura(s) Progol/Revancha
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp"
                  multiple
                  onChange={(event) => {
                    setOcrFiles(Array.from(event.target.files || []));
                    setOcrError("");
                  }}
                  className="mt-2 block w-full text-xs"
                />
              </label>

              <div className="mt-3 grid gap-2 md:grid-cols-3">
                <label className="rounded-lg border border-white/12 bg-black/20 px-2 py-2 text-xs text-white/70">
                  Bloque a leer
                  <select
                    value={ocrSection}
                    onChange={(event) => setOcrSection(event.target.value)}
                    className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50"
                  >
                    <option value="progol">Progol (tabla superior)</option>
                    <option value="revancha">Revancha (tabla inferior)</option>
                  </select>
                </label>
                <label className="rounded-lg border border-white/12 bg-black/20 px-2 py-2 text-xs text-white/70">
                  Cargar desde partido
                  <input
                    type="number"
                    min="1"
                    max={MAX_MATCHES}
                    value={ocrStartPartido}
                    onChange={(event) => setOcrStartPartido(Number(event.target.value) || 1)}
                    className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50"
                  />
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-white/12 bg-black/20 px-2 py-2 text-sm text-white/80">
                  <input
                    type="checkbox"
                    checked={ocrOnlyEmptyRows}
                    onChange={(event) => setOcrOnlyEmptyRows(event.target.checked)}
                    className="h-4 w-4 rounded border-white/25 bg-transparent"
                  />
                  Solo filas vacias
                </label>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={importFromCapture}
                  disabled={!ocrStatus.available || ocrBusy}
                  className="rounded-lg border border-cyan-300/40 bg-cyan-400/12 px-3 py-1.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/18 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {ocrBusy ? "Leyendo captura..." : "Leer captura(s) y cargar boleto"}
                </button>
                <span className="text-xs text-white/55">{ocrFiles.length ? `${ocrFiles.length} archivo(s) seleccionado(s)` : "Sin archivos seleccionados"}</span>
              </div>

              {ocrError ? (
                <p className="mt-2 rounded-lg border border-rose-300/45 bg-rose-400/12 px-3 py-2 text-xs text-rose-100">{ocrError}</p>
              ) : null}
              {ocrNotes.length ? (
                <div className="mt-2 max-h-32 overflow-auto rounded-lg border border-white/12 bg-black/20 px-3 py-2 text-xs text-white/70">
                  {ocrNotes.slice(0, 10).map((note, idx) => (
                    <p key={`ocr-note-${idx}`}>- {note}</p>
                  ))}
                </div>
              ) : null}
            </div>
          </section>

          <section id="progold-memory" className="rounded-2xl border border-white/12 bg-[linear-gradient(180deg,rgba(13,21,36,0.96),rgba(8,13,25,0.98))] p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-white">Memoria historica (patrones)</h3>
                <p className="mt-1 text-sm text-white/65">Panel visual equivalente a PRO-TIP. Integracion completa de historicos queda lista para siguiente iteracion.</p>
              </div>
              <label className="inline-flex items-center gap-2 rounded-full border border-white/16 bg-white/[0.04] px-3 py-1.5 text-sm text-white/80">
                <input type="checkbox" checked={historicalEnabled} onChange={(event) => setHistoricalEnabled(event.target.checked)} className="h-4 w-4 rounded border-white/25 bg-transparent" />
                Activar capa historica
              </label>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <label className="rounded-xl border border-white/12 bg-white/[0.04] p-3 text-sm text-white/80">Minimos requeridos<input type="number" min="1" value={historicalMinRequired} onChange={(event) => setHistoricalMinRequired(Number(event.target.value) || 0)} className="mt-2 w-full rounded-lg border border-white/16 bg-black/25 px-2 py-1.5 text-white outline-none focus:border-cyan-300/50" /></label>
              <label className="rounded-xl border border-white/12 bg-white/[0.04] p-3 text-sm text-white/80">Distancia maxima<input type="number" step="0.01" value={historicalMaxDistance} onChange={(event) => setHistoricalMaxDistance(Number(event.target.value) || 0)} className="mt-2 w-full rounded-lg border border-white/16 bg-black/25 px-2 py-1.5 text-white outline-none focus:border-cyan-300/50" /></label>
              <label className="rounded-xl border border-white/12 bg-white/[0.04] p-3 text-sm text-white/80">Top K similares<input type="number" min="1" value={historicalTopK} onChange={(event) => setHistoricalTopK(Number(event.target.value) || 0)} className="mt-2 w-full rounded-lg border border-white/16 bg-black/25 px-2 py-1.5 text-white outline-none focus:border-cyan-300/50" /></label>
            </div>
            <div className="mt-4 rounded-xl border border-cyan-300/25 bg-cyan-400/8 px-3 py-2 text-sm text-cyan-100/90">Estado actual: UI replicada. Persistencia historica y autosave quedan para la siguiente fase backend.</div>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {PATTERN_GLOSSARY.map((item) => (
                <article key={item.name} className="rounded-xl border border-white/12 bg-white/[0.04] p-3">
                  <p className="inline-flex rounded-full border border-cyan-300/35 bg-cyan-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.09em] text-cyan-100">{item.name}</p>
                  <p className="mt-2 text-sm text-white/80">{item.desc}</p>
                  <p className="mt-2 text-xs text-white/55">Tip: {item.tip}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="progold-editor" className="rounded-2xl border border-white/12 bg-[linear-gradient(180deg,rgba(13,21,36,0.96),rgba(8,13,25,0.98))] p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-white">Editor de boleto (14 partidos)</h3>
                <p className="mt-1 text-sm text-white/65">Captura directa tipo quiniela con el mismo flujo operativo de PRO-TIP.</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={loadDummyRows} className="rounded-lg border border-white/15 bg-white/[0.05] px-3 py-1.5 text-sm font-semibold text-white/80 transition hover:bg-white/[0.1]">Dummy</button>
                <button type="button" onClick={runAnalysis} disabled={loading} className="rounded-lg border border-cyan-300/40 bg-cyan-400/12 px-3 py-1.5 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-400/18 disabled:cursor-not-allowed disabled:opacity-60">{loading ? "Analizando..." : "Analizar"}</button>
              </div>
            </div>
            <div className="mt-4 overflow-x-auto rounded-xl border border-white/12 bg-black/25">
              <table className="min-w-[1060px] w-full text-sm">
                <thead className="bg-white/[0.06] text-white/75">
                  <tr><th className="px-3 py-2 text-left">#</th><th className="px-3 py-2 text-left">Local</th><th className="px-3 py-2 text-left">% Local</th><th className="px-3 py-2 text-left">% Empate</th><th className="px-3 py-2 text-left">% Visita</th><th className="px-3 py-2 text-left">Visitante</th></tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr key={`progold-row-${row.partido}`} className="border-t border-white/10">
                      <td className="px-3 py-2 text-white/70">{row.partido}</td>
                      <td className="px-3 py-2"><input value={row.local} onChange={(event) => handleRowChange(index, "local", event.target.value)} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-2 py-1.5 text-white outline-none focus:border-cyan-300/60" placeholder="Equipo local" /></td>
                      <td className="px-3 py-2"><input type="number" value={row.pct_local} onChange={(event) => handleRowChange(index, "pct_local", event.target.value)} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-2 py-1.5 text-white outline-none focus:border-cyan-300/60" placeholder="0-100" /></td>
                      <td className="px-3 py-2"><input type="number" value={row.pct_empate} onChange={(event) => handleRowChange(index, "pct_empate", event.target.value)} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-2 py-1.5 text-white outline-none focus:border-cyan-300/60" placeholder="0-100" /></td>
                      <td className="px-3 py-2"><input type="number" value={row.pct_visita} onChange={(event) => handleRowChange(index, "pct_visita", event.target.value)} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-2 py-1.5 text-white outline-none focus:border-cyan-300/60" placeholder="0-100" /></td>
                      <td className="px-3 py-2"><input value={row.visitante} onChange={(event) => handleRowChange(index, "visitante", event.target.value)} className="w-full rounded-lg border border-white/14 bg-white/[0.04] px-2 py-1.5 text-white outline-none focus:border-cyan-300/60" placeholder="Equipo visitante" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section id="progold-output" className="rounded-2xl border border-white/12 bg-[linear-gradient(180deg,rgba(13,21,36,0.96),rgba(8,13,25,0.98))] p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-white">Salida analitica</h3>
                <p className="mt-1 text-sm text-white/65">Vista boleto y vista analisis, en el mismo estilo operativo de PRO-TIP.</p>
              </div>
              <div className="text-sm text-white/70">Evaluados: <span className="font-semibold text-white">{summary.evaluated_rows || 0}</span> / {MAX_MATCHES}</div>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <PanelList title="Picks directos" items={directosTop} emptyText="Sin picks directos fuertes." renderItem={(row) => <p key={`directo-${row.partido}`} className="mt-1 text-sm text-white/85">#{row.partido} {row.local} vs {row.visitante}</p>} />
              <PanelList title="Doble oportunidad" items={doblesTop} emptyText="No se detectaron coberturas." renderItem={(row) => <p key={`doble-${row.partido}`} className="mt-1 text-sm text-white/85">#{row.partido} {row.doble_oportunidad}</p>} />
              <PanelList title="Partidos trampa" items={trampaTop} emptyText="Sin riesgo extremo." renderItem={(row) => <p key={`trampa-${row.partido}`} className="mt-1 text-sm text-white/85">#{row.partido} {row.local} vs {row.visitante}</p>} />
              <PanelList title="Partidos contrarian" items={contrarianTop} emptyText="Sin lectura contrarian marcada." renderItem={(row) => <p key={`contrarian-${row.partido}`} className="mt-1 text-sm text-white/85">#{row.partido} {row.local} vs {row.visitante}</p>} />
            </div>

            {viewMode === "boleto" ? (
              <div className="mt-5 space-y-3">
                <h4 className="text-base font-semibold text-white">Vista boleto: {jornadaNombre}</h4>
                {reports.length ? (
                  reports.map((row) => {
                    const confidenceText = String(row.confianza || "-").toLowerCase();
                    return (
                      <article key={`ticket-row-${row.partido}`} className={`rounded-xl border bg-[linear-gradient(180deg,rgba(16,26,44,0.98),rgba(10,16,30,0.98))] p-3 shadow-[0_12px_30px_rgba(2,8,20,0.35)] ${cardClassBySemaforo(row.semaforo)}`}>
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2">
                          <p className="text-sm font-semibold text-white">Partido {String(row.partido).padStart(2, "0")}</p>
                          <span className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] ${pickToneClass(confidenceText)}`}>{String(row.confianza || "sin confianza")}</span>
                        </div>
                        <div className="mt-3 grid gap-3 lg:grid-cols-[1.3fr_1fr_1.1fr]">
                          <div className="space-y-1 text-sm text-white/85">
                            <div className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5"><span>{row.local || "Local"}</span><span className="text-cyan-200">{row.pct_local ?? "--"}%</span></div>
                            <div className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5"><span>Empate</span><span className="text-cyan-200">{row.pct_empate ?? "--"}%</span></div>
                            <div className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5"><span>{row.visitante || "Visitante"}</span><span className="text-cyan-200">{row.pct_visita ?? "--"}%</span></div>
                          </div>
                          <div className="flex flex-col items-center justify-center rounded-lg border border-cyan-300/25 bg-cyan-400/8 p-3 text-center">
                            <p className="text-xs uppercase tracking-[0.08em] text-cyan-100/80">Pick principal</p>
                            <p className="mt-1 text-xl font-semibold text-cyan-100">{row.pick_symbol || "-"}</p>
                            <p className="text-sm text-white/80">{row.recomendacion || "-"}</p>
                          </div>
                          <div className="space-y-2">
                            <p className="rounded-full border border-white/15 bg-white/[0.05] px-2.5 py-1 text-xs text-white/80">Doble: <span className="font-semibold text-cyan-100">{row.doble_oportunidad || "-"}</span></p>
                            <p className={`rounded-full border px-2.5 py-1 text-xs ${semaforoClass(row.semaforo)}`}>Semaforo: <span className="font-semibold">{row.semaforo || "neutro"}</span></p>
                            <p className="rounded-lg border border-white/12 bg-white/[0.03] px-2.5 py-2 text-xs text-white/75">{row.alerta || "Sin alerta"}</p>
                          </div>
                        </div>
                      </article>
                    );
                  })
                ) : (
                  <p className="rounded-lg border border-white/12 bg-white/[0.03] px-3 py-2 text-sm text-white/60">Ejecuta "Analizar boleto" para generar la vista completa.</p>
                )}
              </div>
            ) : (
              <div className="mt-5 space-y-4">
                <h4 className="text-base font-semibold text-white">Vista analisis</h4>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
                  <label className="rounded-lg border border-white/12 bg-white/[0.04] p-2 text-xs text-white/70">Tipo<select value={filterTipo} onChange={(event) => setFilterTipo(event.target.value)} className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50"><option value="__all">Todos</option>{tipoOptions.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                  <label className="rounded-lg border border-white/12 bg-white/[0.04] p-2 text-xs text-white/70">Confianza<select value={filterConfianza} onChange={(event) => setFilterConfianza(event.target.value)} className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50"><option value="__all">Todas</option>{confianzaOptions.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                  <label className="rounded-lg border border-white/12 bg-white/[0.04] p-2 text-xs text-white/70">Semaforo<select value={filterSemaforo} onChange={(event) => setFilterSemaforo(event.target.value)} className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50"><option value="__all">Todos</option>{semaforoOptions.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                  <label className="rounded-lg border border-white/12 bg-white/[0.04] p-2 text-xs text-white/70 xl:col-span-2">Buscar equipo<input value={searchTeam} onChange={(event) => setSearchTeam(event.target.value)} className="mt-1 w-full rounded-md border border-white/16 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-cyan-300/50" placeholder="Ej. America, Tigres..." /></label>
                </div>
                <div className="overflow-x-auto rounded-xl border border-white/12 bg-black/20">
                  <table className="min-w-[1080px] w-full text-sm">
                    <thead className="bg-white/[0.06] text-white/75"><tr><th className="px-3 py-2 text-left">#</th><th className="px-3 py-2 text-left">Partido</th><th className="px-3 py-2 text-left">Pick</th><th className="px-3 py-2 text-left">Doble</th><th className="px-3 py-2 text-left">Confianza</th><th className="px-3 py-2 text-left">Semaforo</th><th className="px-3 py-2 text-left">Tipo</th></tr></thead>
                    <tbody>
                      {filteredReports.map((row) => {
                        const isSelected = selectedPartido === row.partido;
                        return (
                          <tr key={`analysis-row-${row.partido}`} onClick={() => setSelectedPartido(row.partido)} className={`cursor-pointer border-t border-white/10 transition ${isSelected ? "bg-cyan-400/10" : "hover:bg-white/[0.04]"}`}>
                            <td className="px-3 py-2 text-white/70">{row.partido}</td><td className="px-3 py-2 text-white">{row.local} vs {row.visitante}</td><td className="px-3 py-2 text-white/90">{row.pick_symbol || "-"} - {row.recomendacion || "-"}</td><td className="px-3 py-2 text-cyan-200">{row.doble_oportunidad || "-"}</td><td className="px-3 py-2 text-white/85">{row.confianza || "-"}</td><td className="px-3 py-2"><span className={`rounded-full border px-2 py-0.5 text-xs ${semaforoClass(row.semaforo)}`}>{row.semaforo || "neutro"}</span></td><td className="px-3 py-2 text-white/75">{row.tipo_partido || "-"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {selectedRow ? (
                  <article className="rounded-xl border border-white/12 bg-[linear-gradient(180deg,rgba(16,26,44,0.97),rgba(10,16,30,0.98))] p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2"><h5 className="text-base font-semibold text-white">Partido {selectedRow.partido}: {selectedRow.local} vs {selectedRow.visitante}</h5><span className={`rounded-full border px-2.5 py-1 text-xs ${semaforoClass(selectedRow.semaforo)}`}>{selectedRow.semaforo || "neutro"}</span></div>
                    <div className="mt-3 grid gap-3 md:grid-cols-3 xl:grid-cols-6">
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score local</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_local)}</p></div>
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score empate</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_empate)}</p></div>
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score visita</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_visita)}</p></div>
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score riesgo</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_riesgo)}</p></div>
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score contrarian</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_contrarian)}</p></div>
                      <div className="rounded-lg border border-white/12 bg-white/[0.04] p-2"><p className="text-[11px] uppercase tracking-[0.08em] text-white/55">Score estabilidad</p><p className="mt-1 text-lg font-semibold text-white">{safeFixed(selectedRow.analisis?.score_estabilidad)}</p></div>
                    </div>
                    <div className="mt-3 grid gap-2 text-sm text-white/80 md:grid-cols-2">
                      <p className="rounded-lg border border-white/12 bg-white/[0.03] px-3 py-2">Pick final: <span className="font-semibold text-white">{selectedRow.pick_symbol || "-"} - {selectedRow.recomendacion || "-"}</span></p>
                      <p className="rounded-lg border border-white/12 bg-white/[0.03] px-3 py-2">Doble sugerida: <span className="font-semibold text-cyan-100">{selectedRow.doble_oportunidad || "-"}</span></p>
                      <p className="rounded-lg border border-white/12 bg-white/[0.03] px-3 py-2 md:col-span-2">{selectedRow.analisis?.explicacion || selectedRow.explicacion || "Sin explicacion disponible."}</p>
                    </div>
                  </article>
                ) : (
                  <p className="rounded-lg border border-white/12 bg-white/[0.03] px-3 py-2 text-sm text-white/60">No hay filas analizadas con los filtros actuales.</p>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
