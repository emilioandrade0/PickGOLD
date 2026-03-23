from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import uuid

app = Flask(__name__)
CORS(app)

# Simulated in-memory user database
users = [
    {
        "id": str(uuid.uuid4()),
        "name": "Admin",
        "email": "emilio.andra.na@gmail.com",
        "password": hashlib.sha256("adminpassword".encode()).hexdigest(),
        "role": "admin",
        "status": "approved"
    }
]

@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"ok": False, "error": "Todos los campos son requeridos."}), 400
    if any(u["email"] == email for u in users):
        return jsonify({"ok": False, "error": "El email ya está registrado."}), 400
    user = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "role": "user",
        "status": "pending"
    }
    users.append(user)
    return jsonify({"ok": True, "user": {"name": user["name"], "email": user["email"], "role": user["role"], "status": user["status"]}})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email", "").strip().lower()
    password = hashlib.sha256(data.get("password", "").encode()).hexdigest()
    user = next((u for u in users if u["email"] == email and u["password"] == password), None)
    if not user:
        return jsonify({"ok": False, "error": "Credenciales incorrectas."}), 401
    if user["status"] != "approved":
        return jsonify({"ok": False, "error": "Usuario pendiente de aprobación."}), 403
    return jsonify({"ok": True, "user": {"name": user["name"], "email": user["email"], "role": user["role"]}})

@app.route("/api/pending-users", methods=["GET"])
def pending_users():
    admin_email = request.args.get("admin_email", "")
    admin = next((u for u in users if u["email"] == admin_email and u["role"] == "admin"), None)
    if not admin:
        return jsonify({"ok": False, "error": "No autorizado."}), 403
    pending = [
        {"id": u["id"], "name": u["name"], "email": u["email"]}
        for u in users if u["status"] == "pending"
    ]
    return jsonify({"ok": True, "pending": pending})

@app.route("/api/approve-user", methods=["POST"])
def approve_user():
    data = request.json
    admin_email = data.get("admin_email", "")
    user_id = data.get("user_id", "")
    admin = next((u for u in users if u["email"] == admin_email and u["role"] == "admin"), None)
    if not admin:
        return jsonify({"ok": False, "error": "No autorizado."}), 403
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return jsonify({"ok": False, "error": "Usuario no encontrado."}), 404
    user["status"] = "approved"
    return jsonify({"ok": True, "user": {"name": user["name"], "email": user["email"], "role": user["role"], "status": user["status"]}})

if __name__ == "__main__":
    app.run(port=8010, debug=True)
