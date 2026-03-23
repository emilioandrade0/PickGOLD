export default function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isLogin = mode === "login";

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    let action;
    if (isLogin) {
      action = await loginUser({ email, password });
    } else {
      action = await registerUser({ name, email, password });
    }
    setLoading(false);
    if (!action.ok) {
      setError(action.error || "No se pudo completar la acción.");
      return;
    }
    setActiveSession(action.user);
    onAuthenticated?.(action.user);
  }

  return (
    <div className="min-h-screen bg-[#232725] text-white">
      <div className="mx-auto flex min-h-screen w-full max-w-5xl items-center justify-center px-4 py-8">
        <div className="w-full max-w-md rounded-2xl border border-white/15 bg-[#2d3330] p-6 shadow-2xl">
          <h1 className="text-3xl font-semibold tracking-tight">NBA GOLD Access</h1>
          <p className="mt-2 text-sm text-white/70">
            Regístrate una vez para habilitar acceso y luego inicia sesión para entrar a la app.
          </p>

          <div className="mt-5 grid grid-cols-2 gap-2 rounded-xl bg-black/25 p-1">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                isLogin ? "bg-[#007b55] text-white" : "text-white/70 hover:text-white"
              }`}
            >
              Login
            </button>
            <button
              type="button"
              onClick={() => setMode("register")}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                !isLogin ? "bg-[#007b55] text-white" : "text-white/70 hover:text-white"
              }`}
            >
              Registro
            </button>
          </div>

          <form onSubmit={handleSubmit} className="mt-5 space-y-4">
            {!isLogin && (
              <label className="block text-sm">
                <span className="mb-1 block text-white/80">Nombre</span>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-lg border border-white/20 bg-black/20 px-3 py-2 outline-none ring-0 focus:border-[#16a085]"
                  placeholder="Tu nombre"
                  required
                />
              </label>
            )}

            <label className="block text-sm">
              <span className="mb-1 block text-white/80">Email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/20 px-3 py-2 outline-none ring-0 focus:border-[#16a085]"
                placeholder="correo@ejemplo.com"
                required
              />
            </label>

            <label className="block text-sm">
              <span className="mb-1 block text-white/80">Password</span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-white/20 bg-black/20 px-3 py-2 outline-none ring-0 focus:border-[#16a085]"
                placeholder="••••••••"
                required
              />
            </label>

            {error && <p className="text-sm text-red-300">{error}</p>}

            <button
              type="submit"
              className="w-full rounded-lg bg-[#007b55] px-4 py-2 font-semibold text-white transition hover:bg-[#0a8d63] disabled:opacity-60"
              disabled={loading}
            >
              {loading ? "Procesando..." : isLogin ? "Entrar" : "Crear cuenta"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
