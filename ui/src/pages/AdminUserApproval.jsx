import { useCallback, useEffect, useState } from "react";
import { approveUser, getAdminUsers, getPendingUsers } from "../services/auth.js";

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
  const [stats, setStats] = useState({ active_count: 0, approved_count: 0 });
  const [settingsByUser, setSettingsByUser] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchAdminData = useCallback(async () => {
    setLoading(true);
    setError("");

    const [pendingRes, usersRes] = await Promise.all([getPendingUsers(), getAdminUsers()]);

    if (!pendingRes.ok) {
      setPending([]);
      setActiveUsers([]);
      setStats({ active_count: 0, approved_count: 0 });
      setError(pendingRes.error || usersRes.error || "Error al cargar usuarios.");
      setLoading(false);
      return;
    }

    if (!usersRes.ok) {
      setPending(pendingRes.pending || []);
      setActiveUsers([]);
      setStats({ active_count: 0, approved_count: 0 });
      setSettingsByUser((current) => ({
        ...defaultSettingsMap(pendingRes.pending || []),
        ...current,
      }));
      setError(usersRes.error || "No se pudo cargar la lista de usuarios activos.");
      setLoading(false);
      return;
    }

    const nextPending = pendingRes.pending || [];
    const nextUsers = usersRes.users || [];

    setPending(nextPending);
    setActiveUsers(nextUsers);
    setStats({
      active_count: usersRes.active_count || 0,
      approved_count: usersRes.approved_count || 0,
    });
    setSettingsByUser((current) => ({
      ...defaultSettingsMap(nextPending),
      ...current,
    }));
    setLoading(false);
  }, []);

  useEffect(() => {
    const timerId = setTimeout(() => {
      fetchAdminData();
    }, 0);
    return () => clearTimeout(timerId);
  }, [fetchAdminData]);

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

  return (
    <div className="mx-auto mt-8 max-w-5xl rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(28,32,42,0.96),rgba(18,21,29,0.98))] p-6 text-white shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/75">Admin control</p>
          <h2 className="mt-2 text-2xl font-semibold">Aprobación de usuarios</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-white/68">
            Aquí definimos rol y duración del acceso antes de habilitar el dashboard privado.
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

      <div className="mt-6 space-y-3">
        {loading && <p className="text-white/70">Cargando usuarios...</p>}
        {error && <p className="rounded-2xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        {success && <p className="rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">{success}</p>}
      </div>

      <section className="mt-8">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-xl font-semibold">Usuarios pendientes</h3>
            <p className="mt-1 text-sm text-white/60">Aquí verás los registros nuevos que todavía no han sido aprobados.</p>
          </div>
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
                    <span className="mb-2 block text-white/70">Días de acceso</span>
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
            Aquí puedes ver quién ya tiene acceso, qué plan tiene asignado y cuándo vence.
          </p>
        </div>

        <div className="space-y-3">
          {!loading && !error && activeUsers.length === 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-5 text-sm text-white/68">
              Todavía no hay usuarios aprobados.
            </div>
          )}

          {activeUsers.map((user) => (
            <div
              key={user.id}
              className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.12)]"
            >
              <div className="grid gap-4 md:grid-cols-[minmax(0,1.4fr)_140px_160px_160px] lg:grid-cols-[minmax(0,1.6fr)_120px_130px_160px_160px]">
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
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Último login</p>
                  <p className="mt-2 text-sm text-white/80">{formatDate(user.last_login_at)}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Aprobado</p>
                  <p className="mt-2 text-sm text-white/80">{formatDate(user.approved_at)}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
