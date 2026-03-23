ARCHIVOS INCLUIDOS
- backend/app.py
- backend/users.json
- ui/src/services/auth.js
- ui/.env.production.example

CAMBIOS
- auth del frontend ahora usa VITE_AUTH_API_BASE
- el backend auth ya persiste usuarios en backend/users.json
- el correo emilio.andra.na@gmail.com siempre queda como admin aprobado
- si te registras con ese correo, queda como admin
- si ya existe en users.json, el backend lo fuerza a role=admin y status=approved

CREDENCIALES INICIALES DEL ADMIN EN ESTE PATCH
- email: emilio.andra.na@gmail.com
- password: adminpassword

SI QUIERES CAMBIAR EL PASSWORD DEL ADMIN
Opcion 1: editar backend/users.json y reemplazar el hash.
Opcion 2: usar variable de entorno ADMIN_PASSWORD antes de crear users.json en un deploy limpio.

VARIABLES DE ENTORNO RECOMENDADAS
Frontend (Vercel)
- VITE_API_BASE=https://TU-BACKEND-PREDICCIONES.onrender.com/api
- VITE_AUTH_API_BASE=https://TU-BACKEND-AUTH.onrender.com/api

Backend auth (Render)
- ADMIN_EMAIL=emilio.andra.na@gmail.com
- ADMIN_PASSWORD=adminpassword

NOTA
Si en Render el disco es efimero, backend/users.json puede reiniciarse en redeploy. Para persistencia real, usa base de datos o un disco persistente.
