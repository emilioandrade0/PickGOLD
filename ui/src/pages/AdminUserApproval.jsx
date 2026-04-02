import { useCallback, useEffect, useState } from "react";
import { approveUser, getPendingUsers } from "../services/auth.js";

const ROLE_OPTIONS = [
  { value: "member", label: "Member" },
  { value: "vip", label: "VIP" },
  { value: "capper", label: "Capper" },
  { value: "admin", label: "Admin" },
];

function defaultSettingsMap(rows) {
  const map = {};
  for (const row of rows) {
    map[row.id] = { role: "member", accessDays: 30 };
  }
  return map;
}

export default function AdminUserApproval() {
  const [pending, setPending] = useState([]);
  const [settingsByUser, setSettingsByUser] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchPending = useCallback(async () => {
    setLoading(true);
    setError("");
    const res = await getPendingUsers();
    if (res.ok) {
      setPending(res.pending || []);
      setSettingsByUser((current) => ({
        ...defaultSettingsMap(res.pending || []),
        ...current,
      }));
    } else {
      setError(res.error || "Error al cargar usuarios pendientes.");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const timerId = setTimeout(() => {
      fetchPending();
    }, 0);
    return () => clearTimeout(timerId);
  }, [fetchPending]);

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
      setPending((prev) => prev.filter((u) => u.id !== userId));
      setSuccess("Usuario aprobado correctamente.");
    } else {
      setError(res.error || "No se pudo aprobar usuario.");
    }

    setSavingId("");
  }

  return (
    <div className="mx-auto mt-8 max-w-4xl rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(28,32,42,0.96),rgba(18,21,29,0.98))] p-6 text-white shadow-[0_24px_60px_rgba(0,0,0,0.22)]">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-amber-200/75">Admin control</p>
          <h2 className="mt-2 text-2xl font-semibold">Aprobación de usuarios</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-white/68">
            Aquí definimos rol y duración del acceso antes de habilitar el dashboard privado.
          </p>
        </div>
        <button
          type="button"
          onClick={fetchPending}
          className="rounded-2xl border border-white/12 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]"
        >
          Recargar
        </button>
      </div>

      <div className="mt-5 space-y-3">
        {loading && <p className="text-white/70">Cargando usuarios pendientes...</p>}
        {error && <p className="rounded-2xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        {success && <p className="rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">{success}</p>}
        {!loading && pending.length === 0 && (
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
                    Registro: {user.created_at ? new Date(user.created_at).toLocaleString() : "N/A"}
                  </p>
                </div>

                <label className="block text-sm">
                  <span className="mb-2 block text-white/70">Rol</span>
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
                  <span className="mb-2 block text-white/70">Días acceso</span>
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
    </div>
  );
}
