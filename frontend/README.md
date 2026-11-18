# PathSnap frontend

The PathSnap frontend is a React + Vite + TypeScript control room for inspecting GPS trajectories and requesting server-side route reconstruction. It intentionally contains no ML, HMM, Viterbi, preprocessing, or ground-truth logic.

## Run locally

```bash
npm install
npm run dev
```

Open the URL printed by Vite. The default UI opens in clearly labelled Demo mode with illustrative geometry. Switch to Live mode to use a FastAPI backend.

## API configuration

Copy `.env.example` to `.env.local` and set:

```env
VITE_API_BASE_URL=http://localhost:8000
```

The API URL can also be changed at runtime from **API settings**; the value is stored only in local browser storage. For a two-laptop demo, use the backend laptop's reachable LAN address, not `localhost` on the frontend laptop.

The live contract is implemented by `api/main.py` and centralized in `src/services/api.ts`:

- `GET /health`
- `GET /trips` → an array of trip IDs or `{ "trips": [...] }`
- `GET /trips/{trip_id}` → permitted observations and nearby candidate-road context
- `POST /reconstruct` with `{ "trip_id": "..." }` → permitted prediction geometry and optional `metrics`

The API returns `[longitude, latitude]`; the map converts it to Leaflet's `[latitude, longitude]` order. Standard GeoJSON `coordinates` wrappers are also accepted. Reference routes remain on the backend and are used only to calculate development/evaluation metrics after prediction.

## Production build

```bash
npm run build
npm run preview
```

Deploy the `dist/` output to Vercel, Netlify, or another static host and set `VITE_API_BASE_URL` to a reachable backend URL at build time. FastAPI must listen on an appropriate interface and allow the deployed frontend origin through CORS.

Demo mode is deliberately separate and never claims to be model output. For a two-laptop run, configure the backend laptop's LAN address from API settings on the UI laptop.
