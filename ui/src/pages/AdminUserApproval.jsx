import { useEffect, useState } from "react";
import { getPendingUsers, approveUser } from "../services/auth.js";

export default function AdminUserApproval({ adminEmail }) {
  const [pending, setPending] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function fetchPending() {
    setLoading(true);
    setError("");
    const res = await getPendingUsers(adminEmail);
    if (res.ok) {
      setPending(res.pending);
    } else {
      setError(res.error || "Error al cargar usuarios pendientes.");
    }
    setLoading(false);
  }

  async function handleApprove(userId) {
    setError("");
    const res = await approveUser(adminEmail, userId);
    if (res.ok) {
      setPending((prev) => prev.filter((u) => u.id !== userId));
    } else {
      setError(res.error || "No se pudo aprobar usuario.");
    }
  }

  useEffect(() => {
    fetchPending();
    // eslint-disable-next-line
  }, []);

  return (
    <div className="p-4 bg-[#232725] text-white rounded-xl border border-white/10 max-w-xl mx-auto mt-8">
      <h2 className="text-xl font-bold mb-4">Usuarios pendientes de aprobación</h2>
      {loading && <p>Cargando...</p>}
      {error && <p className="text-red-400">{error}</p>}
      {!loading && pending.length === 0 && <p>No hay usuarios pendientes.</p>}
      <ul className="space-y-2">
        {pending.map((user) => (
          <li key={user.id} className="flex items-center justify-between bg-[#2d3330] p-3 rounded-lg">
            <span>{user.name} ({user.email})</span>
            <button
              className="bg-[#007b55] px-3 py-1 rounded text-white hover:bg-[#0a8d63]"
              onClick={() => handleApprove(user.id)}
            >
              Aprobar
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
