import { useCallback, useEffect, useMemo, useState } from "react";
import { startSportUpdateAll } from "../services/api.js";
import { adminDeleteUser, adminResetUserPassword, approveUser, getAdminAllSportsUpdateStatus, getAdminSportUpdates, getAdminUsers, getPendingUsers, startAdminAllSportsUpdate } from "../services/auth.js";

const ROLE_OPTIONS = [
  { value: "member", label: "Starter" },
  { value: "vip", label: "VIP" },
  { value: "capper", label: "Pro" },
  { value: "admin", label: "Admin" },
];

const PLAN_LABELS = {
  member: "Starter",
  vip: "VIP",
  capper: "Pro",
  admin: "Admin",
};

function defaultSettingsMap(rows) {
  const map = {};
  for (const row of rows) {
    map[row.id] = { role: "member", accessDays: 30 };
  }
  return map;
}

function formatDate(value) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "N/A";
  return date.toLocaleString();
}

function AccessBadge({ user }) {
  const isActive = Boolean(user?.is_active);
  const isExpired = Boolean(user?.is_expired);
  const label = isExpired ? "Expirado" : isActive ? "Activo" : user?.status || "N/A";
  const classes = isExpired
    ? "border-rose-400/30 bg-rose-400/10 text-rose-200"
    : isActive
      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200"
      : "border-white/12 bg-white/[0.05] text-white/70";
  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${classes}`}>{label}</span>;
}

export default function AdminUserApproval() {
  const [pending, setPending] = useState([]);
  const [activeUsers, setActiveUsers] = useState([]);
  const [sportUpdates, setSportUpdates] = useState([]);
  const [allSportsUpdate, setAllSportsUpdate] = useState(null);
  const [stats, setStats] = useState({ active_count: 0, approved_count: 0 });
  const [settingsByUser, setSettingsByUser] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState("");
  const [runningSport, setRunningSport] = useState("");
  const [passwordDrafts, setPasswordDrafts] = useState({});
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchAdminData = useCallback(async () => {
    setLoading(true);
    setError("");

    const [pendingRes, usersRes, updatesRes, allRes] = await Promise.all([
      getPendingUsers(),
      getAdminUsers(),
      getAdminSportUpdates(),
      getAdminAllSportsUpdateStatus(),
    ]);

    if (!pendingRes.ok) {
      setPending([]);
      setActiveUsers([]);
      setSportUpdates([]);
      setAllSportsUpdate(null);
      setAllSportsUpdate(null);
      setStats({ active_count: 0, approved_count: 0 });
      setError(pendingRes.error || usersRes.error || updatesRes.error || "Error al cargar usuarios.");
      setLoading(false);
      return;
    }

    const nextPending = pendingRes.pending || [];
    setPending(nextPending);
    setSettingsByUser((current) => ({
      ...defaultSettingsMap(nextPending),
      ...current,
    }));

    if (usersRes.ok) {
      setActiveUsers(usersRes.users || []);
      setStats({
        active_count: usersRes.active_count || 0,
        approved_count: usersRes.approved_count || 0,
      });
    } else {
      setActiveUsers([]);
      setStats({ active_count: 0, approved_count: 0 });
      setError(usersRes.error || "No se pudo cargar la lista de usuarios activos.");
    }

    setSportUpdates(updatesRes?.ok ? (updatesRes.sports || []) : []);
    setAllSportsUpdate(allRes?.ok ? allRes : null);
    setLoading(false);
  }, []);

  useEffect(() => {
    const timerId = setTimeout(() => {
      fetchAdminData();
    }, 0);
    return () => clearTimeout(timerId);
  }, [fetchAdminData]);

  const anyUpdateRunning = useMemo(
    () => (allSportsUpdate?.status === "running") || sportUpdates.some((sport) => sport?.update_status?.status === "running"),
    [allSportsUpdate?.status, sportUpdates],
  );

  useEffect(() => {
    if (!anyUpdateRunning) return undefined;
    const intervalId = setInterval(() => {
      fetchAdminData();
    }, 2500);
    return () => clearInterval(intervalId);
  }, [anyUpdateRunning, fetchAdminData]);

  function updateUserSetting(userId, key, value) {
    setSettingsByUser((current) => ({
      ...current,
      [userId]: {
        role: current[userId]?.role || "member",
        accessDays: current[userId]?.accessDays || 30,
        [key]: value,
      },
    }));
  }

  function updatePasswordDraft(userId, value) {
    setPasswordDrafts((current) => ({ ...current, [userId]: value }));
  }

  async function handleResetPassword(userId) {
    const newPassword = String(passwordDrafts[userId] || "").trim();
    if (!newPassword) {
      setError("Escribe una nueva contrasena para ese usuario.");
      return;
    }
    setSavingId(`reset:${userId}`);
    setError("");
    setSuccess("");
    const res = await adminResetUserPassword({ userId, newPassword });
    if (res?.ok) {
      setSuccess("Contrasena actualizada correctamente.");
      setPasswordDrafts((current) => ({ ...current, [userId]: "" }));
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo actualizar la contrasena.");
    }
    setSavingId("");
  }

  async function handleDeleteUser(userId, userName) {
    const confirmed = window.confirm(`Vas a eliminar a ${userName}. Esta accion no se puede deshacer. Deseas continuar?`);
    if (!confirmed) return;
    setSavingId(`delete:${userId}`);
    setError("");
    setSuccess("");
    const res = await adminDeleteUser({ userId });
    if (res?.ok) {
      setSuccess("Usuario eliminado correctamente.");
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo eliminar el usuario.");
    }
    setSavingId("");
  }

  async function handleApprove(userId) {
    const settings = settingsByUser[userId] || { role: "member", accessDays: 30 };
    setSavingId(userId);
    setError("");
    setSuccess("");

    const res = await approveUser({
      userId,
      role: settings.role,
      accessDays: Number(settings.accessDays) || 30,
    });

    if (res.ok) {
      setSuccess("Usuario aprobado correctamente.");
      await fetchAdminData();
    } else {
      setError(res.error || "No se pudo aprobar usuario.");
    }

    setSavingId("");
  }

  function formatEta(seconds) {
    const total = Number(seconds || 0);
    if (!Number.isFinite(total) || total <= 0) return "Calculando...";
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins <= 0) return `${secs}s`;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    return `${hours}h ${mins % 60}m`;
  }

  async function handleRunAllSportsUpdate() {
    setError("");
    setSuccess("");
    const res = await startAdminAllSportsUpdate();
    if (res?.ok) {
      setSuccess("Actualizacion global iniciada.");
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo iniciar la actualizacion global.");
    }
  }

  async function handleRunSportUpdate(sportKey) {
    setRunningSport(sportKey);
    setError("");
    setSuccess("");
    try {
      await startSportUpdateAll(sportKey);
      setSuccess(`Actualizacion iniciada para ${sportKey.toUpperCase()}.`);
      await fetchAdminData();
    } catch (err) {
      setError(err.message || "No se pudo iniciar la actualizacion del deporte.");
    } finally {
      setRunningSport("");
    }
  }

  return (
    <div className="mx-auto mt-8 max-w-6xl rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(28,32,42,0.96),rgba(18,21,29,0.98))] p-6 text-white shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/75">Admin control</p>
          <h2 className="mt-2 text-2xl font-semibold">Aprobacion de usuarios</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-white/68">
            Aqui definimos rol, acceso y validamos que datos usa cada board antes de publicar cambios en produccion.
          </p>
        </div>
        <button
          type="button"
          onClick={fetchAdminData}
          className="rounded-2xl border border-white/12 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]"
        >
          Recargar
        </button>
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Pendientes</p>
          <p className="mt-2 text-3xl font-semibold">{pending.length}</p>
        </div>
        <div className="rounded-[24px] border border-emerald-400/15 bg-emerald-400/[0.05] p-4">
          <p className="text-[11px] uppercase tracking-[0.16em] text-emerald-200/70">Usuarios activos</p>
          <p className="mt-2 text-3xl font-semibold text-emerald-200">{stats.active_count}</p>
        </div>
        <div className="rounded-[24px] border border-cyan-300/15 bg-cyan-300/[0.05] p-4">
          <p className="text-[11px] uppercase tracking-[0.16em] text-cyan-100/70">Aprobados totales</p>
          <p className="mt-2 text-3xl font-semibold text-cyan-100">{stats.approved_count}</p>
        </div>
      </div>

      <div className="mt-6 rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Actualizar todos los deportes</p>
            <p className="mt-2 text-sm text-white/68">Lanza el pipeline completo de todas las ligas y monitorea el avance desde este panel.</p>
          </div>
          <button
            type="button"
            onClick={handleRunAllSportsUpdate}
            disabled={allSportsUpdate?.status === "running"}
            className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-5 py-3 text-sm font-bold text-[#131821] shadow-[0_12px_28px_rgba(246,196,83,0.2)] transition hover:-translate-y-0.5 hover:brightness-105 disabled:opacity-60"
          >
            {allSportsUpdate?.status === "running" ? "Actualizando todos..." : "Actualizar todos los deportes"}
          </button>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-4">
          <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Estado</p>
            <p className={`mt-2 text-base font-semibold ${allSportsUpdate?.status === "running" ? "text-cyan-100" : allSportsUpdate?.status === "completed" ? "text-emerald-200" : "text-white"}`}>{allSportsUpdate?.status === "running" ? "En progreso" : allSportsUpdate?.status === "completed" ? "Completado" : "Disponible"}</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Avance</p>
            <p className="mt-2 text-base font-semibold text-white">{Number(allSportsUpdate?.percent || 0)}%</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Deportes completados</p>
            <p className="mt-2 text-base font-semibold text-white">{allSportsUpdate?.completed_sports || 0} / {allSportsUpdate?.total_sports || sportUpdates.length}</p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">ETA</p>
            <p className="mt-2 text-base font-semibold text-white">{formatEta(allSportsUpdate?.eta_seconds)}</p>
          </div>
        </div>
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/8">
          <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-400 to-amber-300 transition-all duration-500" style={{ width: `${Math.max(0, Math.min(100, Number(allSportsUpdate?.percent || 0)))}%` }} />
        </div>
        <p className="mt-3 text-sm text-white/70">{allSportsUpdate?.current_sport_label ? `Trabajando en ${allSportsUpdate.current_sport_label}.` : (allSportsUpdate?.message || "Listo para iniciar la actualizacion global.")}</p>
      </div>

      <div className="mt-6 space-y-3">
        {loading && <p className="text-white/70">Cargando usuarios...</p>}
        {error && <p className="rounded-2xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        {success && <p className="rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">{success}</p>}
      </div>

      <section className="mt-10">
        <div className="mb-4">
          <h3 className="text-xl font-semibold">Actualizacion y fuentes por deporte</h3>
          <p className="mt-1 text-sm text-white/60">
            Aqui validas que usa cada pestana, lanzas la actualizacion completa y revisas que archivos estan alimentando el board.
          </p>
        </div>

        <div className="space-y-4">
          {sportUpdates.map((sport) => {
            const updateState = sport.update_status || {};
            const isRunning = updateState.status === "running";
            const steps = Array.isArray(sport.steps) ? sport.steps : [];
            return (
              <div key={sport.sport} className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-3">
                      <h4 className="text-lg font-semibold">{sport.label}</h4>
                      <span className="rounded-full border border-white/12 bg-white/[0.05] px-3 py-1 text-xs text-white/70">{sport.board_route}</span>
                      <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${isRunning ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : sport.board_status?.freshness === "ok" ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200" : sport.board_status?.freshness === "stale" ? "border-amber-300/30 bg-amber-300/10 text-amber-200" : "border-rose-400/30 bg-rose-400/10 text-rose-200"}`}>
                        {isRunning ? `En progreso ${Number(updateState.percent || 0)}%` : sport.board_status?.freshness === "ok" ? "LIVE" : sport.board_status?.freshness === "stale" ? "Desactualizado" : "Fuentes incompletas"}
                      </span>
                    </div>

                    {sport.board_status?.freshness !== "ok" && !isRunning && (
                      <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${sport.board_status?.freshness === "stale" ? "border-amber-300/25 bg-amber-300/10 text-amber-100" : "border-rose-400/25 bg-rose-400/10 text-rose-200"}`}>
                        <p className="font-semibold">{sport.board_status?.title || "Atencion requerida"}</p>
                        <p className="mt-1 text-current/80">{sport.board_status?.message}</p>
                      </div>
                    )}
                    {sport.board_status?.freshness === "ok" && !isRunning && (
                      <div className="mt-4 rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">
                        <p className="flex items-center gap-2 font-semibold"><span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.9)]" /> LIVE</p>
                        <p className="mt-1 text-current/80">{sport.board_status?.message || "Este deporte esta al corriente y listo para usarse."}</p>
                      </div>
                    )}

                    <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Raw history</p>
                        <p className="mt-2 break-all text-xs text-white/72">{sport.raw_history_file || "N/A"}</p>
                        <p className="mt-2 text-[11px] text-white/45">{sport.raw_history_updated_at || "Sin fecha"}</p>
                      </div>
                      <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Upcoming schedule</p>
                        <p className="mt-2 break-all text-xs text-white/72">{sport.upcoming_schedule_file || "N/A"}</p>
                        <p className="mt-2 text-[11px] text-white/45">{sport.upcoming_schedule_updated_at || "Sin fecha"}</p>
                      </div>
                      <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Predicciones activas</p>
                        <p className="mt-2 text-sm font-semibold text-white">{sport.latest_prediction_file || "Sin snapshot"}</p>
                        <p className="mt-2 text-[11px] text-white/45">{sport.prediction_files_count} archivos ? {sport.latest_prediction_updated_at || "Sin fecha"}</p>
                      </div>
                      <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Historico del board</p>
                        <p className="mt-2 text-sm font-semibold text-white">{sport.latest_historical_file || "Sin historial"}</p>
                        <p className="mt-2 text-[11px] text-white/45">{sport.historical_files_count} archivos ? {sport.latest_historical_updated_at || "Sin fecha"}</p>
                      </div>
                    </div>

                    <div className="mt-4 rounded-2xl border border-white/8 bg-black/18 p-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Pipeline de actualizacion</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {steps.map((step) => (
                          <span key={`${sport.sport}-${step.key}`} className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-white/72">
                            {step.label}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="w-full max-w-[260px] rounded-[24px] border border-white/10 bg-black/18 p-4">
                    <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Actualizar {sport.label}</p>
                    <p className="mt-2 text-sm text-white/68">{updateState.current_step_label || updateState.message || "Listo para correr el pipeline completo."}</p>
                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/8">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${isRunning ? "bg-gradient-to-r from-cyan-400 via-sky-400 to-amber-300" : "bg-gradient-to-r from-emerald-500 to-lime-300"}`}
                        style={{ width: `${Math.max(0, Math.min(100, Number(updateState.percent || 0)))}%` }}
                      />
                    </div>
                    <p className="mt-2 text-xs text-white/50">Paso {updateState.completed_steps || 0} de {updateState.total_steps || steps.length} ? ETA {formatEta(updateState.eta_seconds)}</p>
                    <button
                      type="button"
                      onClick={() => handleRunSportUpdate(sport.sport)}
                      disabled={isRunning || runningSport === sport.sport}
                      className="mt-4 w-full rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-4 py-3 text-sm font-bold text-[#131821] shadow-[0_12px_28px_rgba(246,196,83,0.2)] transition hover:-translate-y-0.5 hover:brightness-105 disabled:opacity-60"
                    >
                      {isRunning || runningSport === sport.sport ? "Actualizando..." : `Actualizar ${sport.label}`}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="mt-8">
        <div className="mb-4">
          <h3 className="text-xl font-semibold">Usuarios pendientes</h3>
          <p className="mt-1 text-sm text-white/60">Aqui veras los registros nuevos que todavia no han sido aprobados.</p>
        </div>

        <div className="space-y-3">
          {!loading && !error && pending.length === 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-5 text-sm text-white/68">
              No hay usuarios pendientes por aprobar.
            </div>
          )}

          {pending.map((user) => {
            const settings = settingsByUser[user.id] || { role: "member", accessDays: 30 };
            return (
              <div
                key={user.id}
                className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]"
              >
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_160px_140px_160px] lg:items-end">
                  <div>
                    <p className="text-lg font-semibold">{user.name}</p>
                    <p className="mt-1 text-sm text-white/68">{user.email}</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.16em] text-white/40">
                      Registro: {formatDate(user.created_at)}
                    </p>
                  </div>

                  <label className="block text-sm">
                    <span className="mb-2 block text-white/70">Plan</span>
                    <select
                      value={settings.role}
                      onChange={(e) => updateUserSetting(user.id, "role", e.target.value)}
                      className="w-full rounded-2xl border border-white/12 bg-black/16 px-3 py-3 text-white outline-none transition focus:border-cyan-300/55"
                    >
                      {ROLE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value} className="bg-[#151922]">
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="block text-sm">
                    <span className="mb-2 block text-white/70">Dias de acceso</span>
                    <input
                      type="number"
                      min="1"
                      value={settings.accessDays}
                      onChange={(e) => updateUserSetting(user.id, "accessDays", e.target.value)}
                      className="w-full rounded-2xl border border-white/12 bg-black/16 px-3 py-3 text-white outline-none transition focus:border-cyan-300/55"
                    />
                  </label>

                  <button
                    type="button"
                    onClick={() => handleApprove(user.id)}
                    disabled={savingId === user.id}
                    className="rounded-2xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-4 py-3 text-sm font-bold text-[#131821] shadow-[0_12px_28px_rgba(246,196,83,0.2)] transition hover:-translate-y-0.5 hover:brightness-105 disabled:opacity-60"
                  >
                    {savingId === user.id ? "Guardando..." : "Aprobar acceso"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="mt-10">
        <div className="mb-4">
          <h3 className="text-xl font-semibold">Usuarios activos y aprobados</h3>
          <p className="mt-1 text-sm text-white/60">
            Aqui puedes ver quien ya tiene acceso, que plan tiene asignado y cuando vence.
          </p>
        </div>

        <div className="space-y-3">
          {!loading && !error && activeUsers.length === 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-5 text-sm text-white/68">
              Todavia no hay usuarios aprobados.
            </div>
          )}

          {activeUsers.map((user) => (
            <div
              key={user.id}
              className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.12)]"
            >
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_120px_130px_160px_160px]">
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <p className="text-lg font-semibold">{user.name}</p>
                    <AccessBadge user={user} />
                  </div>
                  <p className="mt-1 text-sm text-white/68">{user.email}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Plan</p>
                  <p className="mt-2 text-sm font-semibold text-amber-200">{PLAN_LABELS[user.plan] || user.plan || "N/A"}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Vence</p>
                  <p className="mt-2 text-sm text-white/80">{user.access_expires_at ? formatDate(user.access_expires_at) : "Sin vencimiento"}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Ultimo login</p>
                  <p className="mt-2 text-sm text-white/80">{formatDate(user.last_login_at)}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Aprobado</p>
                  <p className="mt-2 text-sm text-white/80">{formatDate(user.approved_at)}</p>
                </div>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_170px_170px]">
                <label className="block text-sm">
                  <span className="mb-2 block text-white/70">Nueva contrasena</span>
                  <input
                    type="text"
                    value={passwordDrafts[user.id] || ""}
                    onChange={(e) => updatePasswordDraft(user.id, e.target.value)}
                    placeholder="Escribe una contrasena temporal"
                    className="w-full rounded-2xl border border-white/12 bg-black/16 px-3 py-3 text-white outline-none transition placeholder:text-white/30 focus:border-cyan-300/55"
                  />
                </label>

                <button
                  type="button"
                  onClick={() => handleResetPassword(user.id)}
                  disabled={savingId === `reset:${user.id}`}
                  className="rounded-2xl border border-cyan-300/30 bg-cyan-300/12 px-4 py-3 text-sm font-semibold text-cyan-100 transition hover:border-cyan-200/60 hover:bg-cyan-300/18 disabled:opacity-60"
                >
                  {savingId === `reset:${user.id}` ? "Guardando..." : "Resetear contrasena"}
                </button>

                <button
                  type="button"
                  onClick={() => handleDeleteUser(user.id, user.name)}
                  disabled={savingId === `delete:${user.id}`}
                  className="rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-3 text-sm font-semibold text-rose-200 transition hover:bg-rose-400/16 disabled:opacity-60"
                >
                  {savingId === `delete:${user.id}` ? "Eliminando..." : "Eliminar usuario"}
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
