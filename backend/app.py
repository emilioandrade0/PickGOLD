from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import json
import os
import uuid
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "users.json"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "emilio.andra.na@gmail.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")


def _hash_password(password: str) -> str:
    return hashlib.sha256((password or "").encode()).hexdigest()


def _sanitize_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "status": user["status"],
    }


def _bootstrap_admin() -> list:
    users = [
        {
            "id": str(uuid.uuid4()),
            "name": "Admin",
            "email": ADMIN_EMAIL,
            "password": _hash_password(ADMIN_PASSWORD),
            "role": "admin",
            "status": "approved",
        }
    ]
    save_users(users)
    return users


def load_users() -> list:
    if not DATA_FILE.exists():
        return _bootstrap_admin()

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    except Exception:
        users = []

    if not isinstance(users, list):
        users = []

    admin = next((u for u in users if (u.get("email") or "").strip().lower() == ADMIN_EMAIL), None)
    if admin is None:
        users.append(
            {
                "id": str(uuid.uuid4()),
                "name": "Admin",
                "email": ADMIN_EMAIL,
                "password": _hash_password(ADMIN_PASSWORD),
                "role": "admin",
                "status": "approved",
            }
        )
        save_users(users)
    else:
        changed = False
        if admin.get("role") != "admin":
            admin["role"] = "admin"
            changed = True
        if admin.get("status") != "approved":
            admin["status"] = "approved"
            changed = True
        if changed:
            save_users(users)

    return users


def save_users(users: list) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


@app.get("/api/health")
def health():
    users = load_users()
    pending_count = sum(1 for u in users if u.get("status") == "pending")
    return jsonify(
        {
            "ok": True,
            "service": "auth",
            "admin_email": ADMIN_EMAIL,
            "users": len(users),
            "pending_users": pending_count,
        }
    )


@app.post("/api/register")
def register():
    users = load_users()
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"ok": False, "error": "Todos los campos son requeridos."}), 400

    if any((u.get("email") or "").strip().lower() == email for u in users):
        return jsonify({"ok": False, "error": "El email ya está registrado."}), 400

    is_admin = email == ADMIN_EMAIL
    user = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "password": _hash_password(password),
        "role": "admin" if is_admin else "user",
        "status": "approved" if is_admin else "pending",
    }
    users.append(user)
    save_users(users)

    return jsonify({"ok": True, "user": _sanitize_user(user)})


@app.post("/api/login")
def login():
    users = load_users()
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password_hash = _hash_password(data.get("password") or "")

    user = next(
        (
            u
            for u in users
            if (u.get("email") or "").strip().lower() == email and u.get("password") == password_hash
        ),
        None,
    )
    if not user:
        return jsonify({"ok": False, "error": "Credenciales incorrectas."}), 401

    if user.get("status") != "approved":
        return jsonify({"ok": False, "error": "Usuario pendiente de aprobación."}), 403

    return jsonify({"ok": True, "user": _sanitize_user(user)})


@app.get("/api/pending-users")
def pending_users():
    users = load_users()
    admin_email = (request.args.get("admin_email") or "").strip().lower()
    admin = next(
        (
            u
            for u in users
            if (u.get("email") or "").strip().lower() == admin_email and u.get("role") == "admin"
        ),
        None,
    )
    if not admin:
        return jsonify({"ok": False, "error": "No autorizado."}), 403

    pending = [
        {"id": u["id"], "name": u["name"], "email": u["email"]}
        for u in users
        if u.get("status") == "pending"
    ]
    return jsonify({"ok": True, "pending": pending})


@app.post("/api/approve-user")
def approve_user():
    users = load_users()
    data = request.get_json(silent=True) or {}
    admin_email = (data.get("admin_email") or "").strip().lower()
    user_id = (data.get("user_id") or "").strip()

    admin = next(
        (
            u
            for u in users
            if (u.get("email") or "").strip().lower() == admin_email and u.get("role") == "admin"
        ),
        None,
    )
    if not admin:
        return jsonify({"ok": False, "error": "No autorizado."}), 403

    user = next((u for u in users if u.get("id") == user_id), None)
    if not user:
        return jsonify({"ok": False, "error": "Usuario no encontrado."}), 404

    user["status"] = "approved"
    save_users(users)
    return jsonify({"ok": True, "user": _sanitize_user(user)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    app.run(host="0.0.0.0", port=port, debug=True)
