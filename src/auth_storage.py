from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


BASE_DIR = Path(__file__).resolve().parent
AUTH_DATA_DIR = BASE_DIR / "data" / "auth"
AUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_PATH = AUTH_DATA_DIR / "auth.db"

AUTH_DATABASE_URL = (
    os.getenv("AUTH_DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or ""
).strip()
AUTH_DB_PATH = Path(os.getenv("AUTH_DB_PATH", str(DEFAULT_SQLITE_PATH))).expanduser()
DEFAULT_SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "30"))
DB_IS_POSTGRES = AUTH_DATABASE_URL.startswith(("postgres://", "postgresql://"))

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency in local dev
    psycopg = None
    dict_row = None


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


def _normalize_row(row: Any) -> dict | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if isinstance(row, sqlite3.Row):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return None


def _normalize_rows(rows: list[Any]) -> list[dict]:
    return [row for row in (_normalize_row(item) for item in rows) if row is not None]


def _adapt_query(query: str) -> str:
    if DB_IS_POSTGRES:
        return query.replace("?", "%s")
    return query


@contextmanager
def get_connection() -> Iterator[Any]:
    if DB_IS_POSTGRES:
        if psycopg is None:
            raise RuntimeError(
                "AUTH_DATABASE_URL esta configurado como Postgres, pero psycopg no esta instalado."
            )
        conn = psycopg.connect(AUTH_DATABASE_URL, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(AUTH_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _execute(conn: Any, query: str, params: tuple = ()) -> Any:
    return conn.execute(_adapt_query(query), params)


def _fetchone(conn: Any, query: str, params: tuple = ()) -> dict | None:
    row = _execute(conn, query, params).fetchone()
    return _normalize_row(row)


def _fetchall(conn: Any, query: str, params: tuple = ()) -> list[dict]:
    rows = _execute(conn, query, params).fetchall()
    return _normalize_rows(rows)


def init_auth_db() -> None:
    with get_connection() as conn:
        if DB_IS_POSTGRES:
            _execute(
                conn,
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
                """,
            )
            _execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """,
            )
            _execute(conn, "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            _execute(conn, "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        else:
            _execute(
                conn,
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
                """,
            )
            _execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """,
            )
            _execute(conn, "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            _execute(conn, "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")


def ensure_admin_user(admin_email: str, admin_password: str) -> None:
    email = admin_email.strip().lower()
    now = iso_utc(utc_now())
    password_hash = hash_password(admin_password)
    with get_connection() as conn:
        existing = _fetchone(conn, "SELECT id FROM users WHERE email = ?", (email,))
        if existing:
            _execute(
                conn,
                """
                UPDATE users
                SET name = ?, password_hash = ?, role = 'admin', status = 'approved', access_expires_at = NULL
                WHERE email = ?
                """,
                ("Admin", password_hash, email),
            )
        else:
            _execute(
                conn,
                """
                INSERT INTO users (
                    id, name, email, password_hash, role, status, access_expires_at,
                    created_at, approved_at, approved_by, last_login_at
                ) VALUES (?, ?, ?, ?, 'admin', 'approved', NULL, ?, ?, ?, NULL)
                """,
                (str(uuid.uuid4()), "Admin", email, password_hash, now, now, email),
            )


def row_to_user_payload(row: dict | sqlite3.Row | None) -> dict | None:
    normalized = _normalize_row(row)
    if normalized is None:
        return None
    access_expires_at = normalized.get("access_expires_at")
    expires_at_dt = parse_utc(access_expires_at)
    is_expired = expires_at_dt is not None and utc_now() > expires_at_dt
    is_active = normalized.get("status") == "approved" and not is_expired
    return {
        "id": normalized.get("id"),
        "name": normalized.get("name"),
        "email": normalized.get("email"),
        "role": normalized.get("role"),
        "plan": normalized.get("role"),
        "status": normalized.get("status"),
        "access_expires_at": normalized.get("access_expires_at"),
        "created_at": normalized.get("created_at"),
        "approved_at": normalized.get("approved_at"),
        "approved_by": normalized.get("approved_by"),
        "last_login_at": normalized.get("last_login_at"),
        "is_active": is_active,
        "is_expired": is_expired,
        "access_status": "expired" if is_expired else ("active" if is_active else normalized.get("status")),
    }


def find_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE email = ?", (email.strip().lower(),))


def find_user_by_id(user_id: str) -> dict | None:
    with get_connection() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))


def register_user(name: str, email: str, password: str) -> dict:
    now = iso_utc(utc_now())
    user_id = str(uuid.uuid4())
    with get_connection() as conn:
        _execute(
            conn,
            """
            INSERT INTO users (
                id, name, email, password_hash, role, status, access_expires_at,
                created_at, approved_at, approved_by, last_login_at
            ) VALUES (?, ?, ?, ?, 'member', 'pending', NULL, ?, NULL, NULL, NULL)
            """,
            (user_id, name.strip(), email.strip().lower(), hash_password(password), now),
        )
        row = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    return row_to_user_payload(row)


def set_user_approval(user_id: str, *, role: str, access_expires_at: str | None, approved_by: str) -> dict | None:
    approved_at = iso_utc(utc_now())
    with get_connection() as conn:
        _execute(
            conn,
            """
            UPDATE users
            SET role = ?, status = 'approved', access_expires_at = ?, approved_at = ?, approved_by = ?
            WHERE id = ?
            """,
            (role, access_expires_at, approved_at, approved_by, user_id),
        )
        row = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    return row_to_user_payload(row)


def list_pending_users() -> list[dict]:
    with get_connection() as conn:
        rows = _fetchall(
            conn,
            """
            SELECT * FROM users
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """,
        )
    return [row_to_user_payload(row) for row in rows]


def reset_user_password(user_id: str, new_password: str) -> dict | None:
    with get_connection() as conn:
        _execute(
            conn,
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (hash_password(new_password), user_id),
        )
        _execute(conn, "DELETE FROM sessions WHERE user_id = ?", (user_id,))
        row = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    return row_to_user_payload(row)


def delete_user_account(user_id: str) -> None:
    with get_connection() as conn:
        _execute(conn, "DELETE FROM sessions WHERE user_id = ?", (user_id,))
        _execute(conn, "DELETE FROM users WHERE id = ?", (user_id,))


def list_non_pending_users() -> list[dict]:
    with get_connection() as conn:
        rows = _fetchall(
            conn,
            """
            SELECT * FROM users
            WHERE status != 'pending'
            ORDER BY
                CASE WHEN status = 'approved' THEN 0 ELSE 1 END,
                COALESCE(approved_at, created_at) DESC
            """,
        )
    return [row_to_user_payload(row) for row in rows]


def is_access_expired(user_row: dict | sqlite3.Row | None) -> bool:
    normalized = _normalize_row(user_row)
    if not normalized:
        return False
    dt = parse_utc(normalized.get("access_expires_at"))
    if dt is None:
        return False
    return utc_now() > dt


def create_session(user_id: str, ttl_days: int = DEFAULT_SESSION_TTL_DAYS) -> dict:
    token = uuid.uuid4().hex
    now = utc_now()
    expires_at = now + timedelta(days=max(1, int(ttl_days)))
    with get_connection() as conn:
        _execute(
            conn,
            """
            INSERT INTO sessions (token, user_id, expires_at, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, user_id, iso_utc(expires_at), iso_utc(now), iso_utc(now)),
        )
        _execute(
            conn,
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (iso_utc(now), user_id),
        )
        row = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    return {
        "token": token,
        "session_expires_at": iso_utc(expires_at),
        "user": row_to_user_payload(row),
    }


def delete_session(token: str) -> None:
    with get_connection() as conn:
        _execute(conn, "DELETE FROM sessions WHERE token = ?", (token,))


def get_session(token: str) -> dict | None:
    with get_connection() as conn:
        row = _fetchone(
            conn,
            """
            SELECT s.token, s.expires_at AS session_expires_at, s.created_at AS session_created_at,
                   s.last_seen_at, u.*
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        )
        if row is None:
            return None
        session_expires_at = parse_utc(row.get("session_expires_at"))
        if session_expires_at is None or utc_now() > session_expires_at:
            _execute(conn, "DELETE FROM sessions WHERE token = ?", (token,))
            return None

        _execute(
            conn,
            "UPDATE sessions SET last_seen_at = ? WHERE token = ?",
            (iso_utc(utc_now()), token),
        )

    user = row_to_user_payload(row)
    return {
        "token": row.get("token"),
        "session_expires_at": row.get("session_expires_at"),
        "user": user,
    }
