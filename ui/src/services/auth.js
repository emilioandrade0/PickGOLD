import { API_BASE } from "./api.js";

export async function registerUser({ name, email, password }) {
  try {
    const res = await fetch(`${API_BASE}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    return await res.json();
  } catch (e) {
    return { ok: false, error: "Error de red." };
  }
}

export async function loginUser({ email, password }) {
  try {
    const res = await fetch(`${API_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    return await res.json();
  } catch (e) {
    return { ok: false, error: "Error de red." };
  }
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
  try {
    const res = await fetch(`${API_BASE}/pending-users?admin_email=${encodeURIComponent(adminEmail)}`);
    return await res.json();
  } catch (e) {
    return { ok: false, error: "Error de red." };
  }
}

export async function approveUser(adminEmail, userId) {
  try {
    const res = await fetch(`${API_BASE}/approve-user`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ admin_email: adminEmail, user_id: userId }),
    });
    return await res.json();
  } catch (e) {
    return { ok: false, error: "Error de red." };
  }
}
