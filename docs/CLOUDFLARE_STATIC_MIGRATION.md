# Cloudflare static deployment

The React dashboard runs as a static Cloudflare Pages site. GitHub Actions
precomputes the dashboard JSON snapshots from the repository's data artifacts,
commits them under `frontend/public/data/`, and Cloudflare Pages builds and
deploys the frontend after those commits reach `main`.

## One-time Cloudflare setup

1. Connect the `gmalbert/college-football-predictions` GitHub repository in
   Cloudflare Pages and name the project `tailgate-edge`.
2. Set the production branch to `main`.
3. Set the root directory to `frontend`, build command to
   `npm ci && test -s public/data/snapshots/manifest.json && npm run build`,
   and build output directory to `dist`.
4. Add the Pages environment variable `VITE_STATIC_DATA_ONLY=true` for
   production and preview builds.
5. Add the production custom domain after the first successful deployment.

The manifest check prevents Cloudflare from publishing a frontend before the
first complete catalog is committed. The weekly pipeline creates the full
catalog; the market snapshot workflow refreshes the live market pages while
retaining the historical catalogs. Cloudflare's Git integration deploys each
snapshot commit without exposing any source API key to the browser.

`CFBD_API_KEY` remains a GitHub Actions secret because the Win Probability
snapshots are compiled from CFBD responses. The existing Render blueprint and
FastAPI application remain available for local development during the
migration. Keep the Render service until the Cloudflare site has been checked
across the dashboard routes and interactive selectors.

