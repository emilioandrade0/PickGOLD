import { useCallback, useEffect, useMemo, useState } from "react";
import { startSportUpdateAll } from "../services/api.js";
import { useAppSettings } from "../context/AppSettingsContext.jsx";
import { getTeamDisplayName } from "../utils/teamNames.js";
import {
  adminCaptureTeamSnapshot,
  adminDeleteUser,
  adminResetUserPassword,
  approveUser,
  getAdminPurchaseOrders,
  getAdminAppSettings,
  getAdminMarketAccuracyReport,
  getAdminAllSportsUpdateStatus,
  getAdminTeamSnapshotCompare,
  getAdminSportUpdates,
  getAdminTeamSnapshotMonthly,
  getAdminTeamSnapshotStatus,
  getAdminUsers,
  getPendingUsers,
  startAdminAllSportsUpdate,
  updateAdminPurchaseOrderStatus,
  updateAdminAppSettings,
} from "../services/auth.js";

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

function formatPct(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(2)}%`;
}

function currentMonthYmd() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  return `${y}-${m}`;
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
  const {
    socialMode,
    setSocialMode,
    uiTheme,
    setUiTheme,
    classicLightNightMode,
    setClassicLightNightMode,
  } = useAppSettings();
  const isClassicLight = uiTheme === "classic_light";
  const [pending, setPending] = useState([]);
  const [activeUsers, setActiveUsers] = useState([]);
  const [sportUpdates, setSportUpdates] = useState([]);
  const [marketAccuracy, setMarketAccuracy] = useState([]);
  const [allSportsUpdate, setAllSportsUpdate] = useState(null);
  const [stats, setStats] = useState({ active_count: 0, approved_count: 0 });
  const [purchaseOrders, setPurchaseOrders] = useState([]);
  const [settingsByUser, setSettingsByUser] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingId, setSavingId] = useState("");
  const [runningSport, setRunningSport] = useState("");
  const [expandedSports, setExpandedSports] = useState({});
  const [passwordDrafts, setPasswordDrafts] = useState({});
  const [activeSection, setActiveSection] = useState("overview");
  const [snapshotMonth, setSnapshotMonth] = useState(() => currentMonthYmd());
  const [snapshotStatus, setSnapshotStatus] = useState(null);
  const [snapshotMonthly, setSnapshotMonthly] = useState(null);
  const [snapshotCompare, setSnapshotCompare] = useState(null);
  const [snapshotCompareBusy, setSnapshotCompareBusy] = useState(false);
  const [snapshotBusy, setSnapshotBusy] = useState(false);
  const [appSettingsBusy, setAppSettingsBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fetchAdminData = useCallback(async () => {
    setLoading(true);
    setError("");

    const [pendingRes, usersRes, updatesRes, allRes, marketAccuracyRes, snapshotStatusRes, snapshotMonthlyRes, appSettingsRes, purchaseOrdersRes] = await Promise.all([
      getPendingUsers(),
      getAdminUsers(),
      getAdminSportUpdates(),
      getAdminAllSportsUpdateStatus(),
      getAdminMarketAccuracyReport(),
      getAdminTeamSnapshotStatus(snapshotMonth),
      getAdminTeamSnapshotMonthly(snapshotMonth),
      getAdminAppSettings(),
      getAdminPurchaseOrders({ limit: 300 }),
    ]);

    if (!pendingRes.ok) {
      setPending([]);
      setActiveUsers([]);
      setSportUpdates([]);
      setMarketAccuracy([]);
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
    setMarketAccuracy(marketAccuracyRes?.ok ? (marketAccuracyRes.sports || []) : []);
    setSnapshotStatus(snapshotStatusRes?.ok ? snapshotStatusRes : null);
    setSnapshotMonthly(snapshotMonthlyRes?.ok ? snapshotMonthlyRes : null);
    setPurchaseOrders(purchaseOrdersRes?.ok ? (purchaseOrdersRes.orders || []) : []);
    if (appSettingsRes?.ok) {
      setSocialMode(Boolean(appSettingsRes.social_mode));
      if (appSettingsRes.ui_theme) {
        setUiTheme(String(appSettingsRes.ui_theme));
      }
      setClassicLightNightMode(Boolean(appSettingsRes.classic_light_night_mode));
    }
    setLoading(false);
  }, [setClassicLightNightMode, setSocialMode, setUiTheme, snapshotMonth]);

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

  const marketAccuracyBySport = useMemo(() => {
    const map = {};
    for (const item of marketAccuracy || []) {
      if (!item?.sport) continue;
      map[item.sport] = item;
    }
    return map;
  }, [marketAccuracy]);

  useEffect(() => {
    setExpandedSports((prev) => {
      const next = {};
      for (const sport of sportUpdates || []) {
        const key = sport?.sport;
        if (!key) continue;
        next[key] = prev[key] ?? false;
      }
      return next;
    });
  }, [sportUpdates]);

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

  function toggleSportExpanded(sportKey) {
    setExpandedSports((prev) => ({
      ...prev,
      [sportKey]: !prev[sportKey],
    }));
  }

  async function handleToggleSocialMode() {
    setAppSettingsBusy(true);
    setError("");
    setSuccess("");
    const res = await updateAdminAppSettings({
      socialMode: !socialMode,
      uiTheme,
      classicLightNightMode,
    });
    if (res?.ok) {
      setSocialMode(Boolean(res.social_mode));
      if (res.ui_theme) setUiTheme(String(res.ui_theme));
      setClassicLightNightMode(Boolean(res.classic_light_night_mode));
      setSuccess(res.social_mode ? "Modo redes activado." : "Modo redes desactivado.");
    } else {
      setError(res?.error || "No se pudo actualizar el modo redes.");
    }
    setAppSettingsBusy(false);
  }

  function normalizeThemeValue(nextTheme) {
    const txt = String(nextTheme || "").toLowerCase().trim();
    if (txt === "dashboard_pro") return "dashboard_pro";
    if (txt === "classic_light") return "classic_light";
    return "original";
  }

  function uiThemeLabel(themeValue) {
    if (themeValue === "dashboard_pro") return "Dashboard Pro";
    if (themeValue === "classic_light") return "Classic Light";
    return "Original";
  }

  async function handleUiThemeChange(nextTheme) {
    const normalized = normalizeThemeValue(nextTheme);
    setAppSettingsBusy(true);
    setError("");
    setSuccess("");
    const res = await updateAdminAppSettings({
      socialMode,
      uiTheme: normalized,
      classicLightNightMode,
    });
    if (res?.ok) {
      setUiTheme(res.ui_theme ? String(res.ui_theme) : normalized);
      setClassicLightNightMode(Boolean(res.classic_light_night_mode));
      setSuccess(`Tema UI actualizado a: ${uiThemeLabel(normalized)}.`);
    } else {
      setUiTheme(normalized);
      setSuccess(`Tema UI local actualizado a: ${uiThemeLabel(normalized)}.`);
    }
    setAppSettingsBusy(false);
  }

  async function handleToggleClassicLightNightMode() {
    const nextValue = !classicLightNightMode;
    setAppSettingsBusy(true);
    setError("");
    setSuccess("");
    const res = await updateAdminAppSettings({
      socialMode,
      uiTheme,
      classicLightNightMode: nextValue,
    });
    if (res?.ok) {
      setClassicLightNightMode(Boolean(res.classic_light_night_mode));
      setSuccess(Boolean(res.classic_light_night_mode) ? "Modo noche de Classic Light activado." : "Modo noche de Classic Light desactivado.");
    } else {
      setClassicLightNightMode(nextValue);
      setSuccess(nextValue ? "Modo noche local activado." : "Modo noche local desactivado.");
    }
    setAppSettingsBusy(false);
  }

  async function handleCaptureSnapshot() {
    setSnapshotBusy(true);
    setError("");
    setSuccess("");
    const today = new Date();
    const y = today.getFullYear();
    const m = String(today.getMonth() + 1).padStart(2, "0");
    const d = String(today.getDate()).padStart(2, "0");
    const dateStr = `${y}-${m}-${d}`;

    const res = await adminCaptureTeamSnapshot({ date: dateStr, windowGames: 8 });
    if (res?.ok) {
      setSuccess(res?.message || `Snapshot guardado para ${dateStr}.`);
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo guardar el snapshot.");
    }
    setSnapshotBusy(false);
  }

  async function handleCaptureSnapshotYesterday() {
    setSnapshotBusy(true);
    setError("");
    setSuccess("");
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const y = yesterday.getFullYear();
    const m = String(yesterday.getMonth() + 1).padStart(2, "0");
    const d = String(yesterday.getDate()).padStart(2, "0");
    const dateStr = `${y}-${m}-${d}`;

    const res = await adminCaptureTeamSnapshot({ date: dateStr, windowGames: 8 });
    if (res?.ok) {
      setSuccess(res?.message || `Snapshot guardado para ${dateStr}.`);
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo guardar el snapshot de ayer.");
    }
    setSnapshotBusy(false);
  }

  async function handleUpdateOrderStatus(orderId, nextStatus) {
    setSavingId(`order:${orderId}:${nextStatus}`);
    setError("");
    setSuccess("");
    const res = await updateAdminPurchaseOrderStatus({
      orderId,
      status: nextStatus,
      accessDays: 30,
    });
    if (res?.ok) {
      const code = res?.order?.order_code ? ` (${res.order.order_code})` : "";
      const activated = res?.activated_user ? " y acceso activado" : "";
      setSuccess(`Orden actualizada${code}: ${nextStatus}${activated}.`);
      await fetchAdminData();
    } else {
      setError(res?.error || "No se pudo actualizar la orden.");
    }
    setSavingId("");
  }

  useEffect(() => {
    async function loadCompare() {
      const dates = snapshotStatus?.snapshots_dates || [];
      if (!Array.isArray(dates) || dates.length < 2) {
        setSnapshotCompare(null);
        return;
      }
      const targetDate = dates[dates.length - 1];
      const baseDate = dates[dates.length - 2];
      setSnapshotCompareBusy(true);
      const res = await getAdminTeamSnapshotCompare({ baseDate, targetDate });
      if (res?.ok) setSnapshotCompare(res);
      else setSnapshotCompare(null);
      setSnapshotCompareBusy(false);
    }
    loadCompare();
  }, [snapshotStatus?.snapshots_dates]);

  const shellClass = isClassicLight
    ? "admin-approval-shell admin-approval-shell--classic mx-auto mt-8 max-w-6xl rounded-[28px] border border-[#b7bac2] bg-[linear-gradient(180deg,#d4d5da,#c7c8ce)] p-6 text-[#1f2430] shadow-[0_20px_46px_rgba(0,0,0,0.14)]"
    : "admin-approval-shell mx-auto mt-8 max-w-6xl rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(28,32,42,0.96),rgba(18,21,29,0.98))] p-6 text-white shadow-[0_24px_60px_rgba(0,0,0,0.22)]";
  const headerKickerClass = isClassicLight
    ? "text-[11px] uppercase tracking-[0.18em] text-[#5f6372]"
    : "text-[11px] uppercase tracking-[0.18em] text-amber-200/75";
  const headerSubtitleClass = isClassicLight
    ? "mt-2 max-w-2xl text-sm leading-6 text-[#4e5465]"
    : "mt-2 max-w-2xl text-sm leading-6 text-white/68";
  const reloadButtonClass = isClassicLight
    ? "rounded-2xl border border-[#a7abb6] bg-[#ececf0] px-4 py-2 text-sm font-semibold text-[#2c303a] transition hover:bg-[#f4f4f6]"
    : "rounded-2xl border border-white/12 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]";
  const tabsWrapClass = isClassicLight
    ? "mt-6 flex flex-wrap gap-2 rounded-2xl border border-[#aeb2bc] bg-[#d9dae0] p-2"
    : "mt-6 flex flex-wrap gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-2";

  function sectionTabClass(tabKey) {
    if (isClassicLight) {
      return `rounded-xl border px-3 py-2 text-sm font-semibold transition ${
        activeSection === tabKey
          ? "border-[#8f94a1] bg-[#eef0f3] text-[#20242d]"
          : "border-[#aeb2bc] bg-[#d2d4db] text-[#545b6d] hover:bg-[#e8e9ee] hover:text-[#222730]"
      }`;
    }
    return `rounded-xl border px-3 py-2 text-sm font-semibold transition ${
      activeSection === tabKey
        ? "border-amber-300/60 bg-amber-300/12 text-amber-100"
        : "border-white/10 bg-white/[0.02] text-white/70 hover:border-white/20 hover:text-white"
    }`;
  }

  return (
    <div className={shellClass}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className={headerKickerClass}>Admin control</p>
          <h2 className="mt-2 text-2xl font-semibold">Configuración</h2>
          <p className={headerSubtitleClass}>
            Board de estado de la app.
          </p>
        </div>
        <button
          type="button"
          onClick={fetchAdminData}
          className={reloadButtonClass}
        >
          Recargar
        </button>
      </div>

      <div className={tabsWrapClass}>
        {[
          { key: "overview", label: "Resumen" },
          { key: "sports", label: "Deportes" },
          { key: "snapshots", label: "Snapshots" },
          { key: "orders", label: "Ordenes" },
          { key: "pending", label: "Pendientes" },
          { key: "users", label: "Usuarios" },
        ].map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setActiveSection(tab.key)}
            className={sectionTabClass(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeSection === "overview" && (
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
      )}

      {activeSection === "overview" && (
      <div className="mt-6 rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-black/18 p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Modo redes</p>
            <p className="mt-2 text-sm text-white/68">
              Activa una version mas safe del producto para posts, screenshots y demos.
            </p>
            <button
              type="button"
              onClick={handleToggleSocialMode}
              disabled={appSettingsBusy}
              className={`mt-3 w-full rounded-2xl px-5 py-3 text-sm font-bold transition ${
                socialMode
                  ? "border border-emerald-400/30 bg-emerald-400/12 text-emerald-100 shadow-[0_0_24px_rgba(52,211,153,0.18)]"
                  : "bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] text-[#131821] shadow-[0_12px_28px_rgba(246,196,83,0.2)]"
              } disabled:opacity-60`}
            >
              {appSettingsBusy ? "Guardando..." : socialMode ? "LIVE · Modo redes activado" : "Activar modo redes"}
            </button>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/18 p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Estilo de UI</p>
            <p className="mt-2 text-sm text-white/68">Selecciona el diseño visual de toda la plataforma.</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => handleUiThemeChange("original")}
                disabled={appSettingsBusy}
                className={`rounded-xl border px-4 py-2 text-sm font-semibold transition ${
                  uiTheme === "original"
                    ? "border-amber-300/60 bg-amber-300/12 text-amber-100"
                    : "border-white/12 bg-white/[0.04] text-white/75 hover:border-white/22 hover:text-white"
                }`}
              >
                Original
              </button>
              <button
                type="button"
                onClick={() => handleUiThemeChange("dashboard_pro")}
                disabled={appSettingsBusy}
                className={`rounded-xl border px-4 py-2 text-sm font-semibold transition ${
                  uiTheme === "dashboard_pro"
                    ? "border-cyan-300/60 bg-cyan-300/15 text-cyan-100"
                    : "border-white/12 bg-white/[0.04] text-white/75 hover:border-white/22 hover:text-white"
                }`}
              >
                Dashboard Pro
              </button>
              <button
                type="button"
                onClick={() => handleUiThemeChange("classic_light")}
                disabled={appSettingsBusy}
                className={`rounded-xl border px-4 py-2 text-sm font-semibold transition ${
                  uiTheme === "classic_light"
                    ? "border-slate-300/60 bg-slate-200/15 text-slate-100"
                    : "border-white/12 bg-white/[0.04] text-white/75 hover:border-white/22 hover:text-white"
                }`}
              >
                Classic Light
              </button>
            </div>

            {uiTheme === "classic_light" && (
              <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-white/75">Modo noche</p>
                    <p className="mt-1 text-xs text-white/60">Activa una variante nocturna sin salir de Classic Light.</p>
                  </div>
                  <button
                    type="button"
                    onClick={handleToggleClassicLightNightMode}
                    disabled={appSettingsBusy}
                    className={`relative inline-flex h-8 w-16 items-center rounded-full border transition ${
                      classicLightNightMode
                        ? "border-emerald-300/50 bg-emerald-300/25"
                        : "border-white/20 bg-white/[0.08]"
                    } disabled:opacity-60`}
                    aria-pressed={classicLightNightMode}
                    title={classicLightNightMode ? "Desactivar modo noche" : "Activar modo noche"}
                  >
                    <span
                      className={`inline-block h-6 w-6 transform rounded-full bg-white shadow transition ${
                        classicLightNightMode ? "translate-x-8" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      )}

      {activeSection === "overview" && (
      <div className="mt-6 space-y-4">
        <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]">
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

      </div>
      )}

      <div className="mt-6 space-y-3">
        {loading && <p className="text-white/70">Cargando usuarios...</p>}
        {error && <p className="rounded-2xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        {success && <p className="rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">{success}</p>}
      </div>

      {activeSection === "sports" && (
      <section className="mt-10">
        <div className="mb-4">
          <h3 className="text-xl font-semibold">Actualizacion y fuentes por deporte</h3>
          <p className="mt-1 text-sm text-white/60">
            Aqui validas que usa cada pestana, lanzas la actualizacion completa y revisas que archivos estan alimentando el board.
          </p>
        </div>

        <div className="space-y-4">
          <div className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Accuracy por mercado (validacion vs historico)</p>
            <p className="mt-2 text-sm text-white/65">
              Validacion = accuracy de entrenamiento (ensemble). Historico = desempeno sobre JSON historico (base/adjusted).
            </p>
          </div>
          {sportUpdates.map((sport) => {
            const updateState = sport.update_status || {};
            const isRunning = updateState.status === "running";
            const steps = Array.isArray(sport.steps) ? sport.steps : [];
            const accuracyRow = marketAccuracyBySport[sport.sport] || null;
            const markets = Array.isArray(accuracyRow?.markets) ? accuracyRow.markets : [];
            const isExpanded = expandedSports[sport.sport] === true;
            return (
              <div key={sport.sport} className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]">
                <button
                  type="button"
                  onClick={() => toggleSportExpanded(sport.sport)}
                  className="flex w-full items-center gap-3 text-left"
                >
                  <span className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-white/15 bg-white/[0.05] text-xs text-white/80">
                    {isExpanded ? "-" : "+"}
                  </span>
                  <h4 className="text-lg font-semibold">{sport.label}</h4>
                  <span className="rounded-full border border-white/12 bg-white/[0.05] px-3 py-1 text-xs text-white/70">{sport.board_route}</span>
                  <span className={`ml-auto rounded-full border px-3 py-1 text-xs font-semibold ${isRunning ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : sport.board_status?.freshness === "ok" ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200" : sport.board_status?.freshness === "stale" ? "border-amber-300/30 bg-amber-300/10 text-amber-200" : "border-rose-400/30 bg-rose-400/10 text-rose-200"}`}>
                    {isRunning ? `En progreso ${Number(updateState.percent || 0)}%` : sport.board_status?.freshness === "ok" ? "LIVE" : sport.board_status?.freshness === "stale" ? "Desactualizado" : "Fuentes incompletas"}
                  </span>
                </button>

                {isExpanded && (
                <div className="mt-4">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">

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

                    <div className="mt-4 rounded-2xl border border-white/8 bg-black/18 p-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Accuracy por mercado</p>
                      {markets.length === 0 ? (
                        <p className="mt-2 text-sm text-white/60">Sin reportes aun para este deporte.</p>
                      ) : (
                        <div className="mt-3 overflow-x-auto">
                          <table className="w-full text-left text-xs">
                            <thead className="text-white/45">
                              <tr>
                                <th className="pb-2 pr-3">Mercado</th>
                                <th className="pb-2 pr-3">Validacion</th>
                                <th className="pb-2 pr-3">Historico base</th>
                                <th className="pb-2 pr-3">Historico adj.</th>
                              </tr>
                            </thead>
                            <tbody>
                              {markets.map((mkt) => (
                                <tr key={`${sport.sport}-${mkt.market_key}`} className="border-t border-white/10 text-white/80">
                                  <td className="py-2 pr-3 font-semibold text-white/90">{mkt.market_label || mkt.market_key}</td>
                                  <td className="py-2 pr-3">{formatPct(mkt.validation_accuracy)}</td>
                                  <td className="py-2 pr-3">{formatPct(mkt.historical_base_accuracy)}</td>
                                  <td className="py-2 pr-3">{formatPct(mkt.historical_adjusted_accuracy)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
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
              )}
              </div>
            );
          })}
        </div>
      </section>
      )}

      {activeSection === "snapshots" && (
      <section className="mt-10">
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-xl font-semibold">Snapshots de rendimiento por equipo</h3>
            <p className="mt-1 text-sm text-white/60">
              Guarda fotos diarias y analiza al cierre de mes cuÃ¡ntas predicciones acertÃ³ cada equipo.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="month"
              value={snapshotMonth}
              onChange={(e) => setSnapshotMonth(e.target.value)}
              className="rounded-xl border border-white/12 bg-black/18 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300/50"
            />
            <button
              type="button"
              onClick={handleCaptureSnapshot}
              disabled={snapshotBusy}
              className="rounded-xl border border-emerald-300/35 bg-emerald-300/12 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-300/20 disabled:opacity-60"
            >
              {snapshotBusy ? "Guardando..." : "Capturar snapshot hoy"}
            </button>
            <button
              type="button"
              onClick={handleCaptureSnapshotYesterday}
              disabled={snapshotBusy}
              className="rounded-xl border border-cyan-300/35 bg-cyan-300/12 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/20 disabled:opacity-60"
            >
              {snapshotBusy ? "Guardando..." : "Capturar snapshot ayer"}
            </button>
            <button
              type="button"
              onClick={fetchAdminData}
              className="rounded-xl border border-white/12 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white/85 transition hover:bg-white/[0.08]"
            >
              Refrescar
            </button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Mes</p>
            <p className="mt-2 text-2xl font-semibold text-white">{snapshotMonth || "-"}</p>
          </div>
          <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Snapshots</p>
            <p className="mt-2 text-2xl font-semibold text-cyan-100">{snapshotStatus?.snapshots_count ?? 0}</p>
          </div>
          <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Ultimo snapshot</p>
            <p className="mt-2 text-lg font-semibold text-emerald-200">{snapshotStatus?.latest_snapshot_date || "N/A"}</p>
          </div>
          <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
            <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Equipos en reporte</p>
            <p className="mt-2 text-2xl font-semibold text-amber-200">{snapshotMonthly?.teams?.length ?? 0}</p>
          </div>
        </div>

        <div className="mt-4 rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Comparativo ultimo snapshot vs anterior</p>
          {snapshotCompareBusy ? (
            <p className="mt-3 text-sm text-white/65">Cargando comparativo...</p>
          ) : !snapshotCompare ? (
            <p className="mt-3 text-sm text-white/65">Necesitas al menos 2 snapshots en el mes para comparar.</p>
          ) : (
            <>
              <div className="mt-3 grid gap-3 md:grid-cols-5">
                <div className="rounded-xl border border-white/10 bg-black/18 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Base</p>
                  <p className="mt-1 text-sm font-semibold text-white">{snapshotCompare.base_date}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-black/18 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Objetivo</p>
                  <p className="mt-1 text-sm font-semibold text-white">{snapshotCompare.target_date}</p>
                </div>
                <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-emerald-200/75">Mejoraron</p>
                  <p className="mt-1 text-sm font-semibold text-emerald-200">{snapshotCompare.improved_count}</p>
                </div>
                <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-rose-200/75">Empeoraron</p>
                  <p className="mt-1 text-sm font-semibold text-rose-200">{snapshotCompare.declined_count}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-black/18 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Equipos comparables</p>
                  <p className="mt-1 text-sm font-semibold text-white">{snapshotCompare.teams_comparable ?? snapshotCompare.teams_compared}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-black/18 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Equipos totales</p>
                  <p className="mt-1 text-sm font-semibold text-white">{snapshotCompare.teams_compared}</p>
                </div>
              </div>

              <div className="mt-4 grid gap-4 xl:grid-cols-2">
                <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/[0.06] p-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-200/80">Top mejora (hit rate o forma)</p>
                  <div className="mt-2 space-y-2">
                    {(snapshotCompare.top_improved || []).slice(0, 8).map((row) => (
                      <div key={`imp-${row.sport}-${row.team}`} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/18 px-2 py-1.5 text-sm">
                        <span className="text-white/85">{getTeamDisplayName(row.sport, row.team)} <span className="text-white/50">({row.sport_label})</span></span>
                        <span className="font-semibold text-emerald-200">
                          {row.delta_hit_rate !== null && row.delta_hit_rate !== undefined
                            ? `+${((Number(row.delta_hit_rate) || 0) * 100).toFixed(1)}% HR`
                            : `+${Number(row.delta_form_score_pct || 0).toFixed(1)} pts forma`}
                        </span>
                      </div>
                    ))}
                    {(!snapshotCompare.top_improved || snapshotCompare.top_improved.length === 0) && (
                      <p className="text-sm text-emerald-100/70">Sin mejoras registradas.</p>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-rose-400/20 bg-rose-400/[0.06] p-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-rose-200/80">Top caida (hit rate o forma)</p>
                  <div className="mt-2 space-y-2">
                    {(snapshotCompare.top_declined || []).slice(0, 8).map((row) => (
                      <div key={`dec-${row.sport}-${row.team}`} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/18 px-2 py-1.5 text-sm">
                        <span className="text-white/85">{getTeamDisplayName(row.sport, row.team)} <span className="text-white/50">({row.sport_label})</span></span>
                        <span className="font-semibold text-rose-200">
                          {row.delta_hit_rate !== null && row.delta_hit_rate !== undefined
                            ? `${((Number(row.delta_hit_rate) || 0) * 100).toFixed(1)}% HR`
                            : `${Number(row.delta_form_score_pct || 0).toFixed(1)} pts forma`}
                        </span>
                      </div>
                    ))}
                    {(!snapshotCompare.top_declined || snapshotCompare.top_declined.length === 0) && (
                      <p className="text-sm text-rose-100/70">Sin caidas registradas.</p>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="mt-4 rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
          <p className="mb-3 rounded-xl border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-xs text-amber-100/90">
            Nota: si capturas el snapshot antes de terminar los juegos, veras muchos picks en pendiente y el hit rate aparecera como "-".
          </p>
          <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Top rendimiento mensual por equipo (FG)</p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-white/55">
                <tr>
                  <th className="pb-2">Equipo</th>
                  <th className="pb-2">Deporte</th>
                  <th className="pb-2">Dias</th>
                  <th className="pb-2">Picks</th>
                  <th className="pb-2">Aciertos</th>
                  <th className="pb-2">Pendientes</th>
                  <th className="pb-2">Hit rate</th>
                  <th className="pb-2">Forma prom.</th>
                </tr>
              </thead>
              <tbody>
                {(snapshotMonthly?.top_teams || []).slice(0, 30).map((row) => (
                  <tr key={`${row.sport}-${row.team}`} className="border-t border-white/10">
                    <td className="py-2 font-semibold text-white">{getTeamDisplayName(row.sport, row.team)}</td>
                    <td className="py-2 text-white/75">{row.sport_label || row.sport}</td>
                    <td className="py-2 text-white/75">{row.days}</td>
                    <td className="py-2 text-white/75">{row.picks}</td>
                    <td className="py-2 text-emerald-200">{row.hits}</td>
                    <td className="py-2 text-white/70">{row.pending}</td>
                    <td className="py-2 text-cyan-100">{Number(row.settled || 0) > 0 ? `${((Number(row.hit_rate) || 0) * 100).toFixed(1)}%` : "-"}</td>
                    <td className="py-2 text-amber-200">{row.avg_form_score_pct ?? "-"}</td>
                  </tr>
                ))}
                {(!snapshotMonthly?.top_teams || snapshotMonthly.top_teams.length === 0) && (
                  <tr>
                    <td colSpan={8} className="py-4 text-center text-white/60">
                      Aun no hay snapshots para este mes.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
      )}

      {activeSection === "pending" && (
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
      )}

      {activeSection === "orders" && (
      <section className="mt-8">
        <div className="mb-4">
          <h3 className="text-xl font-semibold">Ordenes de compra por Telegram</h3>
          <p className="mt-1 text-sm text-white/60">
            Gestiona el flujo manual: contacto, pago y activacion de acceso por orden.
          </p>
        </div>

        <div className="space-y-3">
          {!loading && !error && purchaseOrders.length === 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-5 text-sm text-white/68">
              Aun no hay ordenes registradas.
            </div>
          )}

          {purchaseOrders.map((order) => (
            <div
              key={order.id}
              className="rounded-[24px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.14)]"
            >
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_150px_130px_190px] xl:items-center">
                <div>
                  <p className="text-lg font-semibold">{order.order_code}</p>
                  <p className="mt-1 text-sm text-white/68">{order.name} Â· {order.email}</p>
                  <p className="mt-2 text-xs text-white/45">
                    Plan: <span className="font-semibold text-white/80">{String(order.plan_key || "").toUpperCase()}</span> Â· ${order.plan_price_mxn} MXN
                  </p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Estado</p>
                  <p className="mt-2 text-sm font-semibold text-amber-100">{order.status || "N/A"}</p>
                </div>

                <div>
                  <p className="text-[11px] uppercase tracking-[0.16em] text-white/45">Creada</p>
                  <p className="mt-2 text-sm text-white/80">{formatDate(order.created_at)}</p>
                </div>

                <div className="grid gap-2 sm:grid-cols-2">
                  <button
                    type="button"
                    onClick={() => handleUpdateOrderStatus(order.id, "contacted")}
                    disabled={savingId === `order:${order.id}:contacted`}
                    className="rounded-xl border border-cyan-300/30 bg-cyan-300/12 px-3 py-2 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-300/20 disabled:opacity-60"
                  >
                    Contactado
                  </button>
                  <button
                    type="button"
                    onClick={() => handleUpdateOrderStatus(order.id, "paid")}
                    disabled={savingId === `order:${order.id}:paid`}
                    className="rounded-xl border border-emerald-300/30 bg-emerald-300/12 px-3 py-2 text-xs font-semibold text-emerald-100 transition hover:bg-emerald-300/20 disabled:opacity-60"
                  >
                    Pago recibido
                  </button>
                  <button
                    type="button"
                    onClick={() => handleUpdateOrderStatus(order.id, "approved")}
                    disabled={savingId === `order:${order.id}:approved`}
                    className="rounded-xl bg-[linear-gradient(180deg,#ffd95c,#ffbf1f)] px-3 py-2 text-xs font-bold text-[#131821] transition hover:brightness-105 disabled:opacity-60"
                  >
                    Aprobar y activar
                  </button>
                  <button
                    type="button"
                    onClick={() => handleUpdateOrderStatus(order.id, "cancelled")}
                    disabled={savingId === `order:${order.id}:cancelled`}
                    className="rounded-xl border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-xs font-semibold text-rose-200 transition hover:bg-rose-400/16 disabled:opacity-60"
                  >
                    Cancelar
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
      )}

      {activeSection === "users" && (
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
      )}
    </div>
  );
}
