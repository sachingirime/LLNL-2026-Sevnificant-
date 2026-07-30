/**
 * Upload proxy for the Lattice NDE Platform's public dashboard.
 *
 * GitHub Pages (where the dashboard lives) can't hold secrets or run code, so this
 * small Worker is the only piece of "server" in the whole system. All it does:
 *
 *   1. Accept a multipart upload from the dashboard (a CT scan, optionally a design
 *      graph, optionally a threshold hint).
 *   2. Push those files to a new branch in the repo via GitHub's Git Data API, using
 *      a token stored as a Worker secret (never exposed to the browser).
 *   3. Fire a `repository_dispatch` event so `.github/workflows/analyze-upload.yml`
 *      picks it up and runs Codex against that branch.
 *   4. Hand back a branch name / request id the dashboard can poll GitHub's public,
 *      unauthenticated REST API with (no secret needed to just read run/PR status).
 *
 * It does not run any analysis itself and holds no scientific dependencies -- Codex,
 * via the GitHub Action, does the actual work.
 */

const MAX_TOTAL_BYTES = 45 * 1024 * 1024; // stay well under free-tier Worker memory;
                                           // raise if you're on a paid plan and have
                                           // increased the account's request-body limit.

function cors(origin) {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(data, status, origin) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...cors(origin) },
  });
}

function looksLikeTiff(bytes) {
  if (bytes.length < 4) return false;
  const b = new Uint8Array(bytes.buffer || bytes, bytes.byteOffset ?? 0, 4);
  const little = b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a && b[3] === 0x00; // "II*\0"
  const big = b[0] === 0x4d && b[1] === 0x4d && b[2] === 0x00 && b[3] === 0x2a; // "MM\0*"
  return little || big;
}

// btoa() on a huge string blows the stack / is slow; encode in chunks instead.
function base64FromArrayBuffer(buf) {
  const bytes = new Uint8Array(buf);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function gh(env, path, init = {}) {
  const res = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "lattice-nde-platform-worker",
      ...(init.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`GitHub API ${path} -> ${res.status}: ${text.slice(0, 500)}`);
  }
  return res.status === 204 ? null : res.json();
}

async function handleSubmit(request, env, origin) {
  const owner = env.GITHUB_OWNER;
  const repo = env.GITHUB_REPO;
  const baseBranch = env.BASE_BRANCH || "main";

  const contentLength = Number(request.headers.get("content-length") || 0);
  if (contentLength && contentLength > MAX_TOTAL_BYTES) {
    return json({
      error: `Upload too large (${(contentLength / 1e6).toFixed(1)} MB). This dashboard `
        + `accepts up to ${(MAX_TOTAL_BYTES / 1e6).toFixed(0)} MB. For a larger CT stack, `
        + `push it to the repo yourself (git lfs) and open an issue referencing its path `
        + `-- see the repo README.`,
    }, 413, origin);
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return json({ error: "Could not parse the upload as multipart form data." }, 400, origin);
  }

  const tiff = form.get("tiff");
  const design = form.get("design");
  const threshold = form.get("threshold");

  if (!tiff || typeof tiff === "string") {
    return json({ error: "Missing required 'tiff' file." }, 400, origin);
  }
  const tiffName = (tiff.name || "scan.tif").toLowerCase();
  if (!tiffName.endsWith(".tif") && !tiffName.endsWith(".tiff")) {
    return json({ error: "The scan must be a .tif or .tiff file." }, 400, origin);
  }
  const tiffBuf = await tiff.arrayBuffer();
  if (tiffBuf.byteLength > MAX_TOTAL_BYTES) {
    return json({ error: `Scan is ${(tiffBuf.byteLength / 1e6).toFixed(1)} MB, over the ${(MAX_TOTAL_BYTES / 1e6).toFixed(0)} MB limit.` }, 413, origin);
  }
  if (!looksLikeTiff(new Uint8Array(tiffBuf))) {
    return json({ error: "That file doesn't look like a TIFF (bad magic bytes)." }, 400, origin);
  }

  let designBuf = null;
  if (design && typeof design !== "string") {
    const designName = (design.name || "design.json").toLowerCase();
    if (!designName.endsWith(".json")) {
      return json({ error: "The design graph must be a .json file." }, 400, origin);
    }
    designBuf = await design.arrayBuffer();
    try {
      JSON.parse(new TextDecoder().decode(designBuf));
    } catch {
      return json({ error: "The design graph is not valid JSON." }, 400, origin);
    }
  }

  const requestId = crypto.randomUUID().slice(0, 8);
  const branch = `upload/${requestId}`;
  const tiffPath = `uploads/incoming/${requestId}/scan.tif`;
  const designPath = designBuf ? `uploads/incoming/${requestId}/design.json` : null;

  // 1. Base branch tip.
  const baseRef = await gh(env, `/repos/${owner}/${repo}/git/ref/heads/${baseBranch}`);
  const baseSha = baseRef.object.sha;
  const baseCommit = await gh(env, `/repos/${owner}/${repo}/git/commits/${baseSha}`);
  const baseTreeSha = baseCommit.tree.sha;

  // 2. New branch pointing at the same commit for now.
  await gh(env, `/repos/${owner}/${repo}/git/refs`, {
    method: "POST",
    body: JSON.stringify({ ref: `refs/heads/${branch}`, sha: baseSha }),
  });

  // 3. Blobs for the uploaded file(s).
  const tiffBlob = await gh(env, `/repos/${owner}/${repo}/git/blobs`, {
    method: "POST",
    body: JSON.stringify({ content: base64FromArrayBuffer(tiffBuf), encoding: "base64" }),
  });
  const treeEntries = [{ path: tiffPath, mode: "100644", type: "blob", sha: tiffBlob.sha }];
  if (designBuf) {
    const designBlob = await gh(env, `/repos/${owner}/${repo}/git/blobs`, {
      method: "POST",
      body: JSON.stringify({ content: base64FromArrayBuffer(designBuf), encoding: "base64" }),
    });
    treeEntries.push({ path: designPath, mode: "100644", type: "blob", sha: designBlob.sha });
  }

  // 4. Tree + commit + move the branch ref to it.
  const tree = await gh(env, `/repos/${owner}/${repo}/git/trees`, {
    method: "POST",
    body: JSON.stringify({ base_tree: baseTreeSha, tree: treeEntries }),
  });
  const commit = await gh(env, `/repos/${owner}/${repo}/git/commits`, {
    method: "POST",
    body: JSON.stringify({
      message: `Upload: ${tiffName}${designBuf ? " + design graph" : ""} (request ${requestId})`,
      tree: tree.sha,
      parents: [baseSha],
    }),
  });
  await gh(env, `/repos/${owner}/${repo}/git/refs/heads/${branch}`, {
    method: "PATCH",
    body: JSON.stringify({ sha: commit.sha }),
  });

  // 5. Tell the analyze-upload workflow to go.
  await gh(env, `/repos/${owner}/${repo}/dispatches`, {
    method: "POST",
    body: JSON.stringify({
      event_type: "analyze-upload",
      client_payload: {
        request_id: requestId,
        branch,
        tiff_path: tiffPath,
        design_path: designPath,
        threshold: threshold && typeof threshold === "string" ? threshold : null,
        submitted_at: new Date().toISOString(),
      },
    }),
  });

  return json({ request_id: requestId, branch }, 200, origin);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "*";
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(origin) });
    }
    const url = new URL(request.url);
    if (request.method === "POST" && url.pathname === "/submit") {
      try {
        return await handleSubmit(request, env, origin);
      } catch (err) {
        return json({ error: `Server error: ${err.message}` }, 502, origin);
      }
    }
    return json({ error: "Not found. POST a multipart upload to /submit." }, 404, origin);
  },
};
