import { logoutUser } from "../services/auth.js";

export default function PendingApprovalPage({ user, onLogout }) {
  function handleLogout() {
    logoutUser();
    onLogout?.();
  }

  return (
    <div className="min-h-screen bg-[#232725] text-white">
      <div className="mx-auto flex min-h-screen w-full max-w-5xl items-center justify-center px-4 py-8">
        <div className="w-full max-w-xl rounded-2xl border border-white/15 bg-[#2d3330] p-6 shadow-2xl">
          <h1 className="text-3xl font-semibold tracking-tight">Cuenta pendiente de aprobación</h1>
          <p className="mt-3 text-sm leading-6 text-white/75">
            Tu cuenta fue creada correctamente, pero todavía no puede ver la interfaz de predicciones.
            Primero debe ser aprobada por el administrador.
          </p>

          <div className="mt-5 rounded-xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
            Correo registrado: <span className="font-semibold">{user?.email || "Sin correo"}</span>
          </div>

          <div className="mt-4 rounded-xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-white/75">
            En cuanto el administrador te apruebe, podrás iniciar sesión normalmente y acceder a NBA, MLB, NHL y el resto de módulos.
          </div>

          <div className="mt-6 flex gap-3">
            <button
              type="button"
              onClick={handleLogout}
              className="rounded-lg border border-white/15 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15"
            >
              Cerrar sesión
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
