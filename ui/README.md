# NBA GOLD UI

Frontend built with React + Vite.

## Local development

1. Install dependencies:

```bash
npm install
```

2. Run dev server:

```bash
npm run dev
```

By default, the app calls `/api` and falls back to `http://127.0.0.1:8010/api` if needed.

## Production configuration

Set `VITE_API_BASE` in your hosting provider.

Example:

```env
VITE_API_BASE=https://pickgold-api.onrender.com/api
```

You can use `ui/.env.example` as reference.

## Vercel deployment notes

- Build command: `npm run build`
- Output directory: `dist`
- Node version: 20+
- Add env var: `VITE_API_BASE` pointing to your backend API.

PROGOLD is integrated natively in the same frontend/backend deployment (`/progold` + `/api/progold/*`), so no extra app URL is needed.

`ui/vercel.json` includes SPA rewrites so direct routes open correctly.
