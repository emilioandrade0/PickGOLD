const AUTH_BASE = (import.meta.env.VITE_AUTH_API_BASE || "/api").trim().replace(/\/$/, "");
const AUTH_FALLBACK_BASE = "http://127.0.0.1:8010/api";

async function parseResponse(res) {
  let payload = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }
  if (res.ok) {
    return payload || { ok: true };
  }
  return payload || { ok: false, error: `Error ${res.status}` };
}

async function fetchAuth(path, options = undefined) {
  try {
    const res = await fetch(`${AUTH_BASE}${path}`, options);
    return await parseResponse(res);
  } catch {
    try {
      const res = await fetch(`${AUTH_FALLBACK_BASE}${path}`, options);
      return await parseResponse(res);
    } catch {
      return { ok: false, error: "Error de red." };
    }
  }
}

export async function registerUser({ name, email, password }) {
  return fetchAuth("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
}

export async function loginUser({ email, password }) {
  return fetchAuth("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function logoutUser() {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem("nba_gold_session");
  }
}

export function setActiveSession(user) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem("nba_gold_session", JSON.stringify(user));
  }
}

export function getActiveSession() {
  if (typeof window !== "undefined") {
    const raw = window.localStorage.getItem("nba_gold_session");
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
  return null;
}

export async function getPendingUsers(adminEmail) {
  return fetchAuth(`/pending-users?admin_email=${encodeURIComponent(adminEmail)}`);
}

export async function approveUser(adminEmail, userId) {
  return fetchAuth("/approve-user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ admin_email: adminEmail, user_id: userId }),
  });
}
