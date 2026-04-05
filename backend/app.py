import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
import stripe
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "users.json"
PAYMENTS_FILE = BASE_DIR / "payments.json"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "emilio.andra.na@gmail.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173").strip().rstrip("/")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "").strip()
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox").strip().lower()

PLAN_CATALOG = {
    "starter": {"name": "Starter", "price_mxn": 99, "role": "member"},
    "pro": {"name": "Pro", "price_mxn": 249, "role": "vip"},
    "vip": {"name": "VIP", "price_mxn": 499, "role": "capper"},
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def save_json_list(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_payments() -> list:
    return load_json_list(PAYMENTS_FILE)


def save_payments(payments: list) -> None:
    save_json_list(PAYMENTS_FILE, payments)


def record_payment(payload: dict) -> None:
    payments = load_payments()
    payments.append(payload)
    save_payments(payments)


def update_payment_by_reference(provider: str, reference: str, updates: dict) -> None:
    payments = load_payments()
    changed = False
    for payment in payments:
        if payment.get("provider") == provider and payment.get("provider_reference") == reference:
            payment.update(updates)
            payment["updated_at"] = utc_now_iso()
            changed = True
            break
    if changed:
        save_payments(payments)


def get_payment_by_reference(provider: str, reference: str) -> dict | None:
    payments = load_payments()
    return next(
        (
            payment
            for payment in payments
            if payment.get("provider") == provider and payment.get("provider_reference") == reference
        ),
        None,
    )


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

    users = load_json_list(DATA_FILE)

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
    save_json_list(DATA_FILE, users)


def get_plan_or_400(plan_key: str) -> tuple[dict | None, tuple | None]:
    plan = PLAN_CATALOG.get((plan_key or "").strip().lower())
    if not plan:
        return None, (jsonify({"ok": False, "error": "Plan invalido."}), 400)
    return plan, None


def build_frontend_auth_url(plan_key: str, provider: str, payment: str, **extra_params) -> str:
    query = {"plan": plan_key, "provider": provider, "payment": payment}
    for key, value in extra_params.items():
        if value is not None and value != "":
            query[key] = value
    return f"{FRONTEND_ORIGIN}/auth?{urlencode(query)}"


def get_paypal_api_base() -> str:
    return "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"


def get_paypal_access_token() -> str:
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise ValueError("PayPal no esta configurado en el backend.")

    auth_value = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode()).decode()
    response = requests.post(
        f"{get_paypal_api_base()}/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {auth_value}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("PayPal no devolvio access token.")
    return token


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
            "payments_enabled": {
                "stripe": bool(STRIPE_SECRET_KEY),
                "paypal": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET),
            },
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
        return jsonify({"ok": False, "error": "El email ya esta registrado."}), 400

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
        return jsonify({"ok": False, "error": "Usuario pendiente de aprobacion."}), 403

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


@app.post("/api/payments/stripe/checkout-session")
def create_stripe_checkout_session():
    if not STRIPE_SECRET_KEY:
        return jsonify({"ok": False, "error": "Stripe no esta configurado en el backend."}), 400

    data = request.get_json(silent=True) or {}
    plan_key = (data.get("plan_key") or "").strip().lower()
    plan, error_response = get_plan_or_400(plan_key)
    if error_response:
        return error_response

    stripe.api_key = STRIPE_SECRET_KEY

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[
            {
                "price_data": {
                    "currency": "mxn",
                    "product_data": {
                        "name": f"NBA GOLD {plan['name']}",
                        "description": "Acceso mensual con activacion manual.",
                    },
                    "unit_amount": int(plan["price_mxn"] * 100),
                },
                "quantity": 1,
            }
        ],
        metadata={"plan_key": plan_key, "plan_name": plan["name"], "role": plan["role"]},
        success_url=build_frontend_auth_url(
            plan_key,
            "stripe",
            "success",
            session_id="{CHECKOUT_SESSION_ID}",
        ),
        cancel_url=build_frontend_auth_url(plan_key, "stripe", "cancel"),
    )

    record_payment(
        {
            "id": str(uuid.uuid4()),
            "provider": "stripe",
            "provider_reference": session.get("id"),
            "plan_key": plan_key,
            "plan_name": plan["name"],
            "amount_mxn": plan["price_mxn"],
            "status": "created",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
    )

    return jsonify({"ok": True, "url": session.get("url"), "session_id": session.get("id")})


@app.get("/api/payments/stripe/session-status")
def get_stripe_session_status():
    if not STRIPE_SECRET_KEY:
        return jsonify({"ok": False, "error": "Stripe no esta configurado en el backend."}), 400

    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"ok": False, "error": "Falta session_id."}), 400

    stripe.api_key = STRIPE_SECRET_KEY
    session = stripe.checkout.Session.retrieve(session_id)
    paid = session.get("payment_status") == "paid"
    metadata = session.get("metadata") or {}

    update_payment_by_reference(
        "stripe",
        session_id,
        {
            "status": "paid" if paid else session.get("payment_status") or session.get("status") or "unknown",
            "raw_status": session.get("status"),
            "payment_status": session.get("payment_status"),
        },
    )

    return jsonify(
        {
            "ok": True,
            "paid": paid,
            "plan_key": metadata.get("plan_key"),
            "plan_name": metadata.get("plan_name"),
            "payment_status": session.get("payment_status"),
            "status": session.get("status"),
        }
    )


@app.post("/api/payments/paypal/order")
def create_paypal_order():
    data = request.get_json(silent=True) or {}
    plan_key = (data.get("plan_key") or "").strip().lower()
    plan, error_response = get_plan_or_400(plan_key)
    if error_response:
        return error_response

    try:
        access_token = get_paypal_access_token()
    except requests.RequestException:
        return jsonify({"ok": False, "error": "PayPal rechazo la autenticacion del backend."}), 502
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    response = requests.post(
        f"{get_paypal_api_base()}/v2/checkout/orders",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": plan_key,
                    "description": f"NBA GOLD {plan['name']} mensual",
                    "amount": {
                        "currency_code": "MXN",
                        "value": f"{plan['price_mxn']:.2f}",
                    },
                }
            ],
            "application_context": {
                "brand_name": "NBA GOLD",
                "landing_page": "LOGIN",
                "user_action": "PAY_NOW",
                "return_url": build_frontend_auth_url(plan_key, "paypal", "success"),
                "cancel_url": build_frontend_auth_url(plan_key, "paypal", "cancel"),
            },
        },
        timeout=30,
    )

    if response.status_code >= 400:
        return jsonify({"ok": False, "error": "PayPal no pudo crear la orden."}), 502

    payload = response.json()
    order_id = payload.get("id")
    approve_url = next(
        (
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") in {"approve", "payer-action"}
        ),
        "",
    )
    if not order_id or not approve_url:
        return jsonify({"ok": False, "error": "PayPal no devolvio un link de aprobacion."}), 502

    record_payment(
        {
            "id": str(uuid.uuid4()),
            "provider": "paypal",
            "provider_reference": order_id,
            "plan_key": plan_key,
            "plan_name": plan["name"],
            "amount_mxn": plan["price_mxn"],
            "status": payload.get("status") or "created",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
    )

    return jsonify({"ok": True, "url": approve_url, "order_id": order_id})


@app.post("/api/payments/paypal/capture-order")
def capture_paypal_order():
    order_id = ((request.get_json(silent=True) or {}).get("order_id") or "").strip()
    if not order_id:
        return jsonify({"ok": False, "error": "Falta order_id."}), 400

    existing_payment = get_payment_by_reference("paypal", order_id)
    if existing_payment and existing_payment.get("status") == "paid":
        return jsonify(
            {
                "ok": True,
                "paid": True,
                "plan_key": existing_payment.get("plan_key"),
                "plan_name": existing_payment.get("plan_name"),
                "status": "COMPLETED",
            }
        )

    try:
        access_token = get_paypal_access_token()
    except requests.RequestException:
        return jsonify({"ok": False, "error": "PayPal rechazo la autenticacion del backend."}), 502
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    response = requests.post(
        f"{get_paypal_api_base()}/v2/checkout/orders/{order_id}/capture",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    if response.status_code >= 400:
        return jsonify({"ok": False, "error": "PayPal no pudo capturar la orden."}), 502

    payload = response.json()
    paid = payload.get("status") == "COMPLETED"
    purchase_units = payload.get("purchase_units") or []
    plan_key = purchase_units[0].get("reference_id") if purchase_units else None
    plan = PLAN_CATALOG.get(plan_key or "", {})

    update_payment_by_reference(
        "paypal",
        order_id,
        {
            "status": "paid" if paid else payload.get("status") or "captured",
            "raw_status": payload.get("status"),
        },
    )

    return jsonify(
        {
            "ok": True,
            "paid": paid,
            "plan_key": plan_key,
            "plan_name": plan.get("name"),
            "status": payload.get("status"),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    app.run(host="0.0.0.0", port=port, debug=True)
