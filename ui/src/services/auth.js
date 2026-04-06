const AUTH_BASE = (
  import.meta.env.VITE_AUTH_API_BASE ||
  import.meta.env.VITE_API_BASE ||
  "/api"
).trim().replace(/\/$/, "");
const AUTH_FALLBACK_BASE = "http://127.0.0.1:8010/api";
const SESSION_STORAGE_KEY = "nba_gold_session";

function shouldUseLocalFallback() {
  if (!AUTH_BASE) return true;
  return AUTH_BASE.startsWith("/") || AUTH_BASE.includes("127.0.0.1") || AUTH_BASE.includes("localhost");
}

function parseJsonSafe(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function parseResponse(res) {
  const text = await res.text();
  const contentType = String(res.headers.get("content-type") || "").toLowerCase();
  if (!contentType.includes("application/json")) {
    if (res.ok) {
      return { ok: false, error: "La respuesta del servidor no es JSON. Revisa la configuracion de auth en produccion." };
    }
    return { ok: false, error: `Error ${res.status}` };
  }
  const payload = parseJsonSafe(text);
  if (res.ok) {
    return payload || { ok: true };
  }
  return payload || { ok: false, error: `Error ${res.status}` };
}

async function rawFetchAuth(path, options = undefined) {
  try {
    const res = await fetch(`${AUTH_BASE}${path}`, options);
    return await parseResponse(res);
  } catch (error) {
    if (!shouldUseLocalFallback()) {
      return { ok: false, error: error.message || "Error de red." };
    }
    try {
      const res = await fetch(`${AUTH_FALLBACK_BASE}${path}`, options);
      return await parseResponse(res);
    } catch {
      return { ok: false, error: "Error de red." };
    }
  }
}

function withAuthHeaders(token, headers = {}) {
  const nextHeaders = { ...headers };
  if (token) {
    nextHeaders.Authorization = `Bearer ${token}`;
  }
  return nextHeaders;
}

function isSessionExpired(session) {
  if (!session?.session_expires_at) return false;
  const expiresAt = new Date(session.session_expires_at);
  return Number.isFinite(expiresAt.getTime()) && expiresAt.getTime() <= Date.now();
}

export function setActiveSession(payload) {
  if (typeof window === "undefined" || !payload) return;
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(payload));
}

export function clearActiveSession() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

export function getActiveSession() {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
  const session = parseJsonSafe(raw);
  if (!session || isSessionExpired(session)) {
    clearActiveSession();
    return null;
  }
  return session;
}

function normalizeAuthPayload(payload) {
  if (!payload?.ok) return payload;
  if (!payload.user) return payload;
  return {
    ...payload,
    ...payload.user,
  };
}

export async function registerUser({ name, email, password }) {
  const payload = await rawFetchAuth("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  return normalizeAuthPayload(payload);
}

export async function loginUser({ email, password }) {
  const payload = await rawFetchAuth("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return normalizeAuthPayload(payload);
}

export async function refreshSession() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "No hay sesion activa." };
  }
  const payload = await rawFetchAuth("/session", {
    headers: withAuthHeaders(session.token),
  });
  if (!payload?.ok) {
    clearActiveSession();
    return payload;
  }
  const normalized = normalizeAuthPayload(payload);
  setActiveSession(normalized);
  return normalized;
}

export async function logoutUser() {
  const session = getActiveSession();
  if (session?.token) {
    await rawFetchAuth("/logout", {
      method: "POST",
      headers: withAuthHeaders(session.token, { "Content-Type": "application/json" }),
      body: JSON.stringify({ token: session.token }),
    });
  }
  clearActiveSession();
}

export async function getPendingUsers() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/pending-users", {
    headers: withAuthHeaders(session.token),
  });
}

export async function approveUser({ userId, role, accessDays }) {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/approve-user", {
    method: "POST",
    headers: withAuthHeaders(session.token, { "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId, role, access_days: accessDays }),
  });
}

export async function getAdminSportUpdates() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/sport-updates", {
    headers: withAuthHeaders(session.token),
  });
}

export async function getAdminAllSportsUpdateStatus() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/all-sports-update-status", {
    headers: withAuthHeaders(session.token),
  });
}

export async function startAdminAllSportsUpdate() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/update-all-sports", {
    method: "POST",
    headers: withAuthHeaders(session.token, { "Content-Type": "application/json" }),
    body: JSON.stringify({}),
  });
}

export async function adminResetUserPassword({ userId, newPassword }) {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/reset-user-password", {
    method: "POST",
    headers: withAuthHeaders(session.token, { "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId, new_password: newPassword }),
  });
}

export async function adminDeleteUser({ userId }) {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/delete-user", {
    method: "POST",
    headers: withAuthHeaders(session.token, { "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId }),
  });
}

export async function getAdminUsers() {
  const session = getActiveSession();
  if (!session?.token) {
    return { ok: false, error: "Sesion invalida." };
  }
  return rawFetchAuth("/admin/users", {
    headers: withAuthHeaders(session.token),
  });
}
