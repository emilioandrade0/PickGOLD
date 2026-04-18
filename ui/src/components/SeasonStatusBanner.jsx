import { useEffect, useMemo, useState } from "react";
import { getCountdownParts, resolveSeasonStatus } from "../utils/seasonStatus.js";

function CountdownCell({ value, label }) {
  return (
    <div className="rounded-2xl border border-white/12 bg-black/20 px-3 py-3 text-center">
      <p className="text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-[11px] uppercase tracking-[0.16em] text-white/50">{label}</p>
    </div>
  );
}

export default function SeasonStatusBanner({ sportKey, sportLabel }) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const intervalId = setInterval(() => {
      setNow(new Date());
    }, 1000);
    return () => clearInterval(intervalId);
  }, []);

  const seasonStatus = useMemo(
    () => resolveSeasonStatus(sportKey, sportLabel, now),
    [sportKey, sportLabel, now],
  );
  const countdown = useMemo(
    () => getCountdownParts(seasonStatus.countdownTarget, now),
    [seasonStatus.countdownTarget, now],
  );

  return (
    <section className="mb-6 overflow-hidden rounded-[28px] border border-white/12 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.16),transparent_42%),radial-gradient(circle_at_bottom_right,rgba(52,211,153,0.12),transparent_38%),linear-gradient(180deg,rgba(22,26,37,0.96),rgba(14,17,24,0.98))] p-5 shadow-[0_22px_56px_rgba(0,0,0,0.24)]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
            seasonStatus.inSeason
              ? "border-emerald-300/35 bg-emerald-300/12 text-emerald-200"
              : "border-amber-300/35 bg-amber-300/12 text-amber-200"
          }`}
          >
            {seasonStatus.statusLabel}
          </span>
          <span className="rounded-full border border-cyan-300/25 bg-cyan-300/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100">
            Temporada {seasonStatus.seasonLabel}
          </span>
        </div>
        <p className="text-xs text-white/55">
          Objetivo: {seasonStatus.countdownTargetText}
        </p>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div>
          <h3 className="text-xl font-semibold text-white">
            {seasonStatus.inSeason
              ? `${sportLabel}: la temporada esta en curso`
              : `${sportLabel}: cuenta regresiva al nuevo arranque`}
          </h3>
          <p className="mt-2 text-sm text-white/70">
            Campeon vigente ({seasonStatus.championSeasonLabel}):{" "}
            <span className="font-semibold text-white">{seasonStatus.championName}</span>
          </p>
          <div className="mt-4 grid gap-2 text-sm text-white/65 sm:grid-cols-2">
            <p>Inicio temporada: <span className="text-white/85">{seasonStatus.seasonStartText}</span></p>
            <p>Fin temporada: <span className="text-white/85">{seasonStatus.seasonEndText}</span></p>
            <p className="sm:col-span-2">
              Proximo inicio estimado: <span className="text-white/85">{seasonStatus.nextSeasonStartText}</span>
            </p>
          </div>
        </div>

        <div className="rounded-[24px] border border-white/12 bg-white/[0.035] p-4">
          <p className="text-[11px] uppercase tracking-[0.16em] text-white/50">
            {seasonStatus.countdownLabel}
          </p>
          <div className="mt-3 grid grid-cols-4 gap-2">
            <CountdownCell value={countdown.days} label="dias" />
            <CountdownCell value={countdown.hours} label="horas" />
            <CountdownCell value={countdown.minutes} label="min" />
            <CountdownCell value={countdown.seconds} label="seg" />
          </div>
          {countdown.expired && (
            <p className="mt-3 text-xs text-white/60">
              El contador llego a cero. Actualiza fechas en `seasonStatus.js` para el nuevo ciclo.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
