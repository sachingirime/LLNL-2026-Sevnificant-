# Upload proxy (Cloudflare Worker)

This is the one piece of "server" in the whole public-dashboard setup. GitHub Pages
(where `platform/frontend/static_index.html` is published) can only serve static
files -- it can't accept an upload or hold a credential. This Worker does both, and
nothing else: it takes the uploaded scan, pushes it to a new branch via GitHub's API,
and fires the event that makes `.github/workflows/analyze-upload.yml` run Codex against
it. All the actual analysis happens in that GitHub Actions workflow, not here.

I can't create your Cloudflare account, GitHub token, or OpenAI key for you, so this
part needs about five minutes of setup on your end. Everything else (the workflow, the
dashboard) is already committed and doesn't need touching.

## 1. Create a GitHub token for the Worker

Settings -> Developer settings -> Personal access tokens -> **Fine-grained token** (or
classic, simpler):

- **Classic token**: scope `repo` (needed to push branches and trigger
  `repository_dispatch`). Simplest option.
- **Fine-grained token**: repository access limited to
  `sachingirime/LLNL-2026-Sevnificant-`, permissions **Contents: Read and write** and
  **Metadata: Read-only** (repository_dispatch needs Contents write; if the dispatch
  call gets a 403, add **Actions: Read and write** too).

Copy the token -- you won't see it again.

## 2. Deploy the Worker

```bash
cd platform/cloudflare-worker
npm install -g wrangler      # if you don't already have it
wrangler login               # opens a browser to your Cloudflare account (free tier is fine)
wrangler secret put GITHUB_TOKEN   # paste the token from step 1 when prompted
wrangler deploy
```

Wrangler prints the deployed URL, something like
`https://lattice-nde-upload-proxy.<your-subdomain>.workers.dev`. That's your proxy URL.

If you forked the repo or use a different owner/repo, edit `wrangler.toml`'s `[vars]`
block first (`GITHUB_OWNER`, `GITHUB_REPO`, `BASE_BRANCH`).

## 3. Point the dashboard at it

Open `platform/frontend/static_index.html`, find the line near the top of the
`<script>` block:

```js
const UPLOAD_PROXY_URL = ""; // <-- set to your deployed Worker URL, e.g.
                              //     "https://lattice-nde-upload-proxy.you.workers.dev/submit"
```

Set it to `<your Worker URL>/submit`, commit, push to `main` -- the Pages workflow
republishes it automatically.

## 4. Add the OpenAI key the workflow needs

Repo Settings -> Secrets and variables -> Actions -> New repository secret:
`OPENAI_API_KEY` = a key from platform.openai.com/api-keys. This is what
`.github/workflows/analyze-upload.yml` uses to run Codex headlessly via
`openai/codex-action`.

## Limits and what's deliberately NOT built here

- **45 MB upload cap** (`MAX_TOTAL_BYTES` in `worker.js`), to stay inside the free
  Workers plan's request/memory limits. A full-resolution CT stack can easily exceed
  this -- if that's your common case, either raise the cap on a paid Workers plan, or
  skip the dashboard upload for those and push the file to the repo yourself (`git lfs`)
  and trigger the workflow manually (`gh workflow run` / the Actions tab accepts
  `workflow_dispatch` too if you add that trigger).
- **No rate limiting is built into the Worker code.** The dashboard is open to anyone
  with the link by design (per the project's own decision), and every submission costs
  real OpenAI API usage plus GitHub Actions minutes. The cheap fix that needs no code:
  Cloudflare dashboard -> your zone/Worker -> **Security -> WAF -> Rate limiting rules**
  -- a few clicks, no redeploy. Worth doing before sharing the link widely.
- **No virus/content scanning** beyond checking the TIFF magic bytes and that the design
  file parses as JSON. Don't treat this as a general-purpose public file drop.
