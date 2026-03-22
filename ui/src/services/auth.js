const REGISTERED_USER_KEY = "nba_gold_registered_user";
const ACTIVE_SESSION_KEY = "nba_gold_active_session";

function safeParse(raw) {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function getRegisteredUser() {
  if (!canUseStorage()) return null;
  return safeParse(window.localStorage.getItem(REGISTERED_USER_KEY));
}

export function registerUser({ name, email, password }) {
  if (!canUseStorage()) {
    return { ok: false, error: "Storage no disponible." };
  }

  const cleanName = String(name || "").trim();
  const cleanEmail = String(email || "").trim().toLowerCase();
  const cleanPassword = String(password || "");

  if (cleanName.length < 2) return { ok: false, error: "Nombre demasiado corto." };
  if (!cleanEmail.includes("@")) return { ok: false, error: "Email inválido." };
  if (cleanPassword.length < 4) return { ok: false, error: "Password mínimo 4 caracteres." };

  const payload = {
    name: cleanName,
    email: cleanEmail,
    password: cleanPassword,
    createdAt: new Date().toISOString(),
  };

  window.localStorage.setItem(REGISTERED_USER_KEY, JSON.stringify(payload));
  window.localStorage.setItem(
    ACTIVE_SESSION_KEY,
    JSON.stringify({ name: payload.name, email: payload.email, loggedAt: new Date().toISOString() })
  );

  return { ok: true, user: { name: payload.name, email: payload.email } };
}

export function loginUser({ email, password }) {
  if (!canUseStorage()) {
    return { ok: false, error: "Storage no disponible." };
  }

  const registered = getRegisteredUser();
  if (!registered) {
    return { ok: false, error: "Primero debes registrarte." };
  }

  const cleanEmail = String(email || "").trim().toLowerCase();
  const cleanPassword = String(password || "");

  if (cleanEmail !== String(registered.email || "").toLowerCase() || cleanPassword !== String(registered.password || "")) {
    return { ok: false, error: "Credenciales incorrectas." };
  }

  const session = {
    name: String(registered.name || "Usuario"),
    email: String(registered.email || cleanEmail),
    loggedAt: new Date().toISOString(),
  };

  window.localStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify(session));
  return { ok: true, user: session };
}

export function getActiveSession() {
  if (!canUseStorage()) return null;
  return safeParse(window.localStorage.getItem(ACTIVE_SESSION_KEY));
}

export function logoutUser() {
  if (!canUseStorage()) return;
  window.localStorage.removeItem(ACTIVE_SESSION_KEY);
}
