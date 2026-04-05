# NBA GOLD Deployment Guide

## 1) Backend (Render)

This repository includes `render.yaml` for one-click deployment.

1. In Render, create a new Web Service from this GitHub repository.
2. Render should detect `render.yaml` automatically.
3. Set the following environment variables in Render:
	- `THEODDSAPI_KEY` (required)
	- `FRONTEND_ORIGIN` (required in production, example: `https://pickgold.vercel.app`)
	- `STRIPE_SECRET_KEY` (optional, required para checkout con Stripe)
	- `PAYPAL_CLIENT_ID` (optional, required para checkout con PayPal)
	- `PAYPAL_CLIENT_SECRET` (optional, required para checkout con PayPal)
	- `PAYPAL_ENV` (optional, `sandbox` o `live`)
	- `FLASHSCORE_ENABLED` (optional, default `1`)
	- `THEODDSAPI_DAYS_AHEAD` (optional, default `2`)
4. Deploy and verify:
	- `https://YOUR-RENDER-SERVICE.onrender.com/health`
	- Expected response: `{"status":"ok","service":"nba-gold-api"}`

## 2) Frontend (Vercel)

1. Import the `ui` folder as a Vercel project.
2. Configure build settings:
	- Framework Preset: Vite
	- Build Command: `npm run build`
	- Output Directory: `dist`
3. Add env var in Vercel:
	- `VITE_API_BASE=https://YOUR-RENDER-SERVICE.onrender.com/api`
4. Redeploy frontend.

`ui/vercel.json` is already configured for SPA route rewrites.

## 3) GitHub hourly refresh automation

The workflow file `.github/workflows/hourly-refresh.yml` runs every hour.

Set repository secrets:
- `THEODDSAPI_KEY` (required)
- `FLASHSCORE_ENABLED` (optional)
- `THEODDSAPI_DAYS_AHEAD` (optional)

Then run once manually from GitHub Actions:
1. Actions -> `Hourly Data Refresh`
2. Click `Run workflow`
3. Confirm it completes successfully

## 4) Important note on Render free plan

Render free web services can sleep after inactivity and may take time to wake up.

For strict 24/7 availability:
- Use Render paid plan, or
- Move backend to an always-on host (for example, Oracle Cloud Always Free VM).
