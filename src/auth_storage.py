from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
AUTH_DATA_DIR = BASE_DIR / "data" / "auth"
AUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
AUTH_DB_PATH = AUTH_DATA_DIR / "auth.db"

DEFAULT_SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "30"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                access_expires_at TEXT NULL,
                created_at TEXT NOT NULL,
                approved_at TEXT NULL,
                approved_by TEXT NULL,
                last_login_at TEXT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")


def ensure_admin_user(admin_email: str, admin_password: str) -> None:
    email = admin_email.strip().lower()
    now = iso_utc(utc_now())
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        password_hash = hash_password(admin_password)
        if existing:
            conn.execute(
                """
                UPDATE users
                SET name = ?, password_hash = ?, role = 'admin', status = 'approved', access_expires_at = NULL
                WHERE email = ?
                """,
                ("Admin", password_hash, email),
            )
        else:
            conn.execute(
                """
                INSERT INTO users (
                    id, name, email, password_hash, role, status, access_expires_at,
                    created_at, approved_at, approved_by, last_login_at
                ) VALUES (?, ?, ?, ?, 'admin', 'approved', NULL, ?, ?, ?, NULL)
                """,
                (str(uuid.uuid4()), "Admin", email, password_hash, now, now, email),
            )


def row_to_user_payload(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    access_expires_at = row["access_expires_at"]
    expires_at_dt = parse_utc(access_expires_at)
    is_expired = expires_at_dt is not None and utc_now() > expires_at_dt
    is_active = row["status"] == "approved" and not is_expired
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
        "plan": row["role"],
        "status": row["status"],
        "access_expires_at": row["access_expires_at"],
        "created_at": row["created_at"],
        "approved_at": row["approved_at"],
        "approved_by": row["approved_by"],
        "last_login_at": row["last_login_at"],
        "is_active": is_active,
        "is_expired": is_expired,
        "access_status": "expired" if is_expired else ("active" if is_active else row["status"]),
    }


def find_user_by_email(email: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()


def find_user_by_id(user_id: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def register_user(name: str, email: str, password: str) -> dict:
    now = iso_utc(utc_now())
    user_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, name, email, password_hash, role, status, access_expires_at,
                created_at, approved_at, approved_by, last_login_at
            ) VALUES (?, ?, ?, ?, 'member', 'pending', NULL, ?, NULL, NULL, NULL)
            """,
            (user_id, name.strip(), email.strip().lower(), hash_password(password), now),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row_to_user_payload(row)


def set_user_approval(user_id: str, *, role: str, access_expires_at: str | None, approved_by: str) -> dict | None:
    approved_at = iso_utc(utc_now())
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET role = ?, status = 'approved', access_expires_at = ?, approved_at = ?, approved_by = ?
            WHERE id = ?
            """,
            (role, access_expires_at, approved_at, approved_by, user_id),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row_to_user_payload(row)


def list_pending_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM users
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """
        ).fetchall()
    return [row_to_user_payload(row) for row in rows]


def reset_user_password(user_id: str, new_password: str) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (hash_password(new_password), user_id),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row_to_user_payload(row)


def delete_user_account(user_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_non_pending_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM users
            WHERE status != 'pending'
            ORDER BY
                CASE WHEN status = 'approved' THEN 0 ELSE 1 END,
                COALESCE(approved_at, created_at) DESC
            """
        ).fetchall()
    return [row_to_user_payload(row) for row in rows]


def is_access_expired(user_row: sqlite3.Row | dict | None) -> bool:
    if not user_row:
        return False
    value = user_row["access_expires_at"] if isinstance(user_row, sqlite3.Row) else user_row.get("access_expires_at")
    dt = parse_utc(value)
    if dt is None:
        return False
    return utc_now() > dt


def create_session(user_id: str, ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> dict:
    token = uuid.uuid4().hex
    now = utc_now()
    expires_at = now + timedelta(days=max(1, int(ttl_days)))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, expires_at, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, user_id, iso_utc(expires_at), iso_utc(now), iso_utc(now)),
        )
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (iso_utc(now), user_id),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return {
        "token": token,
        "session_expires_at": iso_utc(expires_at),
        "user": row_to_user_payload(row),
    }


def delete_session(token: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def get_session(token: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT s.token, s.expires_at AS session_expires_at, s.created_at AS session_created_at,
                   s.last_seen_at, u.*
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            return None
        session_expires_at = parse_utc(row["session_expires_at"])
        if session_expires_at is None or utc_now() > session_expires_at:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None

        conn.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE token = ?",
            (iso_utc(utc_now()), token),
        )

    user = row_to_user_payload(row)
    return {
        "token": row["token"],
        "session_expires_at": row["session_expires_at"],
        "user": user,
    }
