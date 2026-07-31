/* Codex dashboard — chat pane + live execution trace.
 *
 * The server streams `codex exec --json` events straight through over SSE.
 * Observed event types: thread.started, turn.started, item.started,
 * item.completed, turn.completed. Observed item types: agent_message,
 * command_execution. Everything else is rendered by the generic branch rather
 * than dropped — the CLI emits more item types than were seen while building
 * this, and a trace that silently omits a step is worse than an ugly one.
 */

const $ = (id) => document.getElementById(id);

const state = {
  threadId: null,
  runId: null,
  turn: 0,
  step: 0,        // trace steps shown in the timeline
  packetStep: 0,  // packet sequence for the exported trace
  running: false,
  startedAt: null,
  timer: null,
  cards: new Map(), // `${turn}:${item.id}` -> element   (item ids restart each turn)
  usage: [],        // one entry per completed turn
  trace: [],        // MEP-shaped packets, for export
  tools: 0,
  outputTokens: 0,
  diagnostics: [],
};

const SUGGESTIONS = [
  "Which MCP tools in src/mcp_server.py are not on the canonical path documented in AGENTS.md? List them with line numbers.",
  "Read outputs/lattice_iou/strut_classes.csv and report the count per defect class, plus each as a fraction of the measurable struts.",
  "Which directories under outputs/ hold superseded methods? Quote the file that says so — don't guess from filenames.",
  "Walk the canonical analysis path in AGENTS.md and tell me, for each step, what failure it guards against.",
];

/* ---------- small helpers ---------- */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function mdLite(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.replace(/^\n/, "")}</code></pre>`);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  return html;
}

const compact = (n) => {
  if (n < 1000) return String(n);
  if (n < 1e6) return (n / 1000).toFixed(n < 10000 ? 1 : 0) + "K";
  return (n / 1e6).toFixed(1) + "M";
};

const commas = (n) => n.toLocaleString("en-US");

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

const hexId = () => Math.random().toString(16).slice(2, 14).padEnd(12, "0");

/* ---------- MEP-shaped trace packets ---------- */

/* The MCP server writes its own packets to outputs/mep/ with session_id and
 * client_id null — it cannot see which agent called it. These are recorded on
 * the agent side, so they can carry that attribution. Same envelope either way,
 * so both halves can be read by the same tooling. */
function packet({ tool, kind, args, why, status, error, artifact }) {
  state.packetStep += 1;
  return {
    run_id: state.runId || "dashboard",
    step: state.packetStep,
    packet_id: hexId(),
    ts: Date.now() / 1000,
    duration_s: 0.0,
    actor: "codex-agent",
    session: {
      session_id: state.threadId,
      client_id: "codex-dashboard",
      request_id: state.runId,
    },
    tool,
    kind,
    args: args || {},
    why: why || "",
    status: status || "ok",
    error: error ?? null,
    artifact: artifact ?? "",
    artifact_files: [],
    inputs: [],
    verification: [],
  };
}

/* ---------- chat pane ---------- */

function addMessage(role, text) {
  $("chat-empty")?.remove();
  const wrap = el("div", `msg msg-${role}`);
  wrap.append(el("span", "msg-role", role === "user" ? "you" : "agent"));
  const bubble = el("div", "bubble");
  bubble.innerHTML = mdLite(text);
  wrap.append(bubble);
  $("transcript").append(wrap);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return wrap;
}

function setThinking(on) {
  $("thinking")?.remove();
  if (!on) return;
  const node = el("div", "thinking", "working…");
  node.id = "thinking";
  $("transcript").append(node);
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

/* ---------- trace pane ---------- */

function turnRule(n) {
  $("timeline").querySelector(".empty")?.remove();
  $("timeline").append(el("div", "turn-rule", `turn ${n}`));
}

function fold(summaryText, bodyText, open) {
  const d = el("details", "fold");
  if (open) d.open = true;
  d.append(el("summary", null, summaryText));
  const body = el("div", "card-body");
  body.append(el("pre", null, bodyText));
  d.append(body);
  return d;
}

/* Build (or rebuild) the card for one trace item. item.started and
 * item.completed carry the same id, so the completed event replaces the
 * in-progress card in place. */
function renderItem(item) {
  const key = `${state.turn}:${item.id}`;
  const existing = state.cards.get(key);
  const card = el("div", "card");
  const head = el("div", "card-head");
  card.append(head);

  const stepNo = existing?.dataset.step || String(++state.step);
  card.dataset.step = stepNo;
  head.append(el("span", "card-step", `${stepNo}`));

  const type = item.type || "unknown";

  if (type === "command_execution") {
    card.classList.add("card-cmd");
    head.append(el("span", "card-kind", "shell"));
    head.append(el("span", "card-title", stripBashWrapper(item.command || "")));
    head.append(statusChip(item));
    const out = (item.aggregated_output || "").trimEnd();
    if (out) {
      const failed = item.exit_code != null && item.exit_code !== 0;
      const lines = out.split("\n").length;
      card.append(fold(`${lines} line${lines === 1 ? "" : "s"} of output`, out, failed));
    }
  } else if (type === "mcp_tool_call") {
    card.classList.add("card-mcp");
    head.append(el("span", "card-kind", "mcp"));
    const name = [item.server, item.tool].filter(Boolean).join(".") || item.name || "tool";
    head.append(el("span", "card-title", name));
    head.append(statusChip(item));
    const args = item.arguments ?? item.args;
    if (args) card.append(fold("arguments", pretty(args), false));
    const result = item.result ?? item.output ?? item.error;
    if (result) card.append(fold("result", pretty(result), item.status === "failed"));
  } else if (type === "agent_message") {
    card.classList.add("card-msg");
    head.append(el("span", "card-kind", "message"));
    const text = item.text || "";
    head.append(el("span", "card-title card-title-plain", text.split("\n")[0].slice(0, 90)));
  } else if (type === "reasoning") {
    card.classList.add("card-reason");
    head.append(el("span", "card-kind", "reasoning"));
    const text = item.text || item.summary || "";
    head.append(el("span", "card-title card-title-plain", firstLine(text, 90)));
    if (text.length > 90) card.append(fold("full", text, false));
  } else if (type === "file_change" || type === "patch_apply") {
    card.classList.add("card-file");
    head.append(el("span", "card-kind", "files"));
    const changes = item.changes || item.files || [];
    const names = changes.map((c) => c.path || c.file || String(c));
    head.append(el("span", "card-title", names.join(", ") || "file change"));
    head.append(statusChip(item));
  } else if (type === "error") {
    card.classList.add("card-error");
    head.append(el("span", "card-kind", "error"));
    head.append(el("span", "card-title card-title-plain", item.message || "error"));
  } else {
    // Unrecognised item type — show it whole rather than guess at its fields.
    head.append(el("span", "card-kind", type));
    head.append(el("span", "card-title card-title-plain", describe(item)));
    card.append(fold("raw event", pretty(item), false));
  }

  if (existing) existing.replaceWith(card);
  else $("timeline").append(card);
  state.cards.set(key, card);
  $("timeline").scrollTop = $("timeline").scrollHeight;
}

function statusChip(item) {
  const running =
    item.status === "in_progress" || (item.exit_code == null && item.status !== "completed");
  if (running && item.status !== "failed") {
    return el("span", "status status-running", "running");
  }
  if (item.exit_code != null) {
    return item.exit_code === 0
      ? el("span", "status status-ok", "exit 0")
      : el("span", "status status-fail", `exit ${item.exit_code}`);
  }
  return item.status === "failed"
    ? el("span", "status status-fail", "failed")
    : el("span", "status status-ok", "ok");
}

const firstLine = (s, n) => String(s).split("\n")[0].slice(0, n);
const pretty = (v) => (typeof v === "string" ? v : JSON.stringify(v, null, 2));

/* Headline for an item type this dashboard has no specific card for: prefer a
 * human-meaningful scalar field over dumping the JSON's opening brace. */
function describe(item) {
  for (const key of ["query", "name", "title", "path", "text", "summary", "message", "command"]) {
    if (typeof item[key] === "string" && item[key].trim()) return firstLine(item[key], 90);
  }
  const scalars = Object.entries(item)
    .filter(([k, v]) => k !== "id" && k !== "type" && (typeof v === "string" || typeof v === "number"))
    .map(([k, v]) => `${k}=${v}`);
  return scalars.length ? firstLine(scalars.join(" "), 90) : "(no detail)";
}

/* codex runs commands through `/bin/bash -lc '…'`; the wrapper is noise on
 * screen. It picks the quote style per command, so match either. */
function stripBashWrapper(cmd) {
  const m = /^\S*(?:bash|sh|zsh)\s+-l?c\s+(['"])([\s\S]*)\1$/.exec(cmd);
  return m ? m[2] : cmd;
}

/* Record the item as a trace packet once it is final. */
function recordItem(item) {
  const type = item.type || "unknown";
  if (type === "command_execution") {
    state.tools += 1;
    state.trace.push(packet({
      tool: "shell",
      kind: "action",
      args: { command: item.command },
      why: "agent-issued shell command",
      status: item.exit_code === 0 ? "ok" : "error",
      error: item.exit_code === 0 ? null : `exit ${item.exit_code}`,
      artifact: item.aggregated_output || "",
    }));
  } else if (type === "mcp_tool_call") {
    state.tools += 1;
    state.trace.push(packet({
      tool: [item.server, item.tool].filter(Boolean).join(".") || "mcp_tool",
      kind: "tool_call",
      args: item.arguments ?? item.args ?? {},
      why: "agent-issued MCP tool call",
      status: item.status === "failed" ? "error" : "ok",
      error: item.error ?? null,
      artifact: pretty(item.result ?? item.output ?? ""),
    }));
  } else if (type === "agent_message") {
    state.trace.push(packet({
      tool: "agent_message", kind: "message", args: {},
      why: "agent reply to the user", artifact: item.text || "",
    }));
  } else if (type === "reasoning") {
    state.trace.push(packet({
      tool: "reasoning", kind: "reasoning", args: {},
      why: "agent reasoning summary", artifact: item.text || item.summary || "",
    }));
  } else {
    state.trace.push(packet({
      tool: type, kind: "event", args: item,
      why: "unclassified codex event", artifact: "",
    }));
  }
}

/* ---------- stats & token chart ---------- */

function refreshStats() {
  $("stat-turns").textContent = commas(state.usage.length);
  $("stat-tools").textContent = commas(state.tools);
  $("stat-output").textContent = compact(state.outputTokens);
  $("timeline-note").textContent = `${state.step} step${state.step === 1 ? "" : "s"}`;
  $("export").disabled = state.trace.length === 0;
}

function tickElapsed() {
  if (!state.startedAt) return;
  const s = Math.round((Date.now() - state.startedAt) / 1000);
  $("stat-elapsed").textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

function renderTokenChart() {
  if (!state.usage.length) return;
  $("token-chart-wrap").hidden = false;
  const bars = $("token-bars");
  bars.replaceChildren();
  const max = Math.max(...state.usage.map((u) => u.input));

  state.usage.forEach((u, i) => {
    const row = el("div", "bar-row");
    row.append(el("span", "bar-turn", String(i + 1)));

    const track = el("div", "bar-track");
    const fresh = Math.max(0, u.input - u.cached);
    const mk = (cls, value) => {
      const seg = el("div", `bar-seg ${cls}`);
      seg.style.width = `${(value / max) * 100}%`;
      return seg;
    };
    if (u.cached > 0) track.append(mk("bar-cached", u.cached));
    if (fresh > 0) track.append(mk("bar-fresh", fresh));

    // Hover gives exact numbers; only the total is labelled directly.
    track.addEventListener("mousemove", (ev) => showTooltip(ev, u, fresh, i + 1));
    track.addEventListener("mouseleave", hideTooltip);

    row.append(track);
    row.append(el("span", "bar-total", compact(u.input)));
    bars.append(row);
  });
}

function showTooltip(ev, u, fresh, turn) {
  const tip = $("tooltip");
  const pct = u.input ? Math.round((u.cached / u.input) * 100) : 0;
  tip.innerHTML = `<strong>Turn ${turn}</strong>
    <dl>
      <dt><i class="swatch swatch-cached"></i>Cached</dt><dd>${commas(u.cached)}</dd>
      <dt><i class="swatch swatch-fresh"></i>Fresh</dt><dd>${commas(fresh)}</dd>
      <dt>Total in</dt><dd>${commas(u.input)}</dd>
      <dt>Output</dt><dd>${commas(u.output)}</dd>
      <dt>Cache hit</dt><dd>${pct}%</dd>
    </dl>`;
  tip.hidden = false;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  let left = ev.clientX + pad;
  if (left + rect.width > window.innerWidth) left = ev.clientX - rect.width - pad;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.min(window.innerHeight - rect.height - 4, ev.clientY + pad)}px`;
}

const hideTooltip = () => { $("tooltip").hidden = true; };

function addDiagnostic(line) {
  state.diagnostics.push(line);
  $("diagnostics").hidden = false;
  $("diag-count").textContent = String(state.diagnostics.length);
  $("diag-body").textContent = state.diagnostics.join("\n");
}

/* ---------- event handling ---------- */

function handleEvent(ev) {
  switch (ev.type) {
    case "run.started":
      state.runId = ev.run_id;
      $("cancel").hidden = false;
      $("composer-hint").textContent = ev.command
        .map((c) => (/[\s"']/.test(c) ? JSON.stringify(c) : c))
        .join(" ")
        .slice(0, 160);
      break;

    case "user_prompt": // replay only — the live path adds this when you hit send
      addMessage("user", ev.text);
      break;

    case "thread.started":
      state.threadId = ev.thread_id;
      $("thread-chip").textContent = ev.thread_id.slice(0, 8);
      $("thread-chip").title = ev.thread_id;
      break;

    case "turn.started":
      state.turn += 1;
      turnRule(state.turn);
      break;

    case "item.started":
    case "item.updated":
      if (ev.item) renderItem(ev.item);
      break;

    case "item.completed":
      if (!ev.item) break;
      renderItem(ev.item);
      recordItem(ev.item);
      if (ev.item.type === "agent_message" && ev.item.text) {
        setThinking(false);
        addMessage("agent", ev.item.text);
        setThinking(state.running);
      }
      refreshStats();
      break;

    case "turn.completed": {
      const u = ev.usage || {};
      state.usage.push({
        input: u.input_tokens || 0,
        cached: u.cached_input_tokens || 0,
        output: u.output_tokens || 0,
      });
      state.outputTokens += u.output_tokens || 0;
      renderTokenChart();
      refreshStats();
      break;
    }

    case "turn.failed":
    case "error":
      addDiagnostic(pretty(ev));
      addMessage("agent", `The turn failed. See diagnostics.\n\n${pretty(ev).slice(0, 400)}`);
      break;

    case "stderr":
      addDiagnostic(ev.line);
      break;

    case "raw":
      addDiagnostic(`unparsed stdout: ${ev.line}`);
      break;

    case "run.failed":
      addMessage("agent", `Could not start codex: ${ev.error}`);
      break;

    case "run.exited":
      if (ev.code !== 0) addDiagnostic(`codex exited with code ${ev.code}`);
      break;

    default:
      addDiagnostic(`unhandled event: ${pretty(ev)}`);
  }
}

/* ---------- SSE plumbing ---------- */

async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (!data) continue; // keep-alive comment or a bare event: line
      try {
        yield JSON.parse(data);
      } catch {
        // A non-JSON data line is worth seeing, not swallowing.
        addDiagnostic(`unparsed SSE data: ${data}`);
      }
    }
  }
}

async function send(prompt) {
  if (state.running) return;
  state.running = true;
  $("send").disabled = true;
  $("prompt").value = "";
  addMessage("user", prompt);
  setThinking(true);

  state.trace.push(packet({
    tool: "user_prompt", kind: "goal", args: { prompt },
    why: "the request, recorded verbatim before the agent acts",
  }));

  state.startedAt = Date.now();
  state.timer = setInterval(tickElapsed, 1000);
  tickElapsed();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        thread_id: state.threadId,
        sandbox: $("sandbox").value,
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      addMessage("agent", `Server refused the turn (${response.status}). ${detail}`);
    } else {
      for await (const ev of sseEvents(response)) handleEvent(ev);
    }
  } catch (err) {
    addMessage("agent", `Stream broke: ${err}`);
  } finally {
    state.running = false;
    state.runId = null;
    setThinking(false);
    clearInterval(state.timer);
    $("send").disabled = false;
    $("cancel").hidden = true;
    $("composer-hint").textContent = "";
    refreshStats();
  }
}

/* =================== visualization tab =================== */

/* What actually decides each class, from scripts/classify_strut_defects.classify. The
 * metric names are columns of strut_classes.csv; a run that lacks them just shows fewer
 * tiles rather than inventing numbers. */
/* Each entry lists candidate column names, most telling first. Names differ between a
 * lattice_iou table (r_eq_med_um, empty_sections) and an MCP run's merged struts.csv +
 * connectivity.npz (radius_med_vox, gap_stations), so both spellings are listed and only
 * the ones this run actually has get a tile. */
const CLASS_EVIDENCE = {
  missing: {
    keys: ["iou", "empty_sections", "gap_stations", "n_matched", "r_eq_med_um", "radius_med_vox"],
    why: "No voxel of the nominal cylinder is material AND every cross-section is empty. " +
         "A count that is zero or not — no threshold is involved.",
  },
  broken: {
    keys: ["detour", "reachable", "geo_len_vox", "span_vox", "tube_material_frac"],
    why: "No geodesic path through material from one node to the other, inside a tube of " +
         "2× the nominal radius. Topological, so it cannot be recovered from cross-sections: " +
         "a crack narrower than the section spacing reads full on every plane.",
  },
  thin: {
    keys: ["r_eq_med_um", "radius_med_vox", "diameter_est_um", "iou"],
    why: "Median section radius below the tolerance band.",
  },
  thick: {
    keys: ["r_eq_med_um", "radius_med_vox", "diameter_est_um", "excess_frac", "iou"],
    why: "Median section radius above the tolerance band.",
  },
  necked: {
    keys: ["r_eq_min_um", "radius_min_vox", "r_eq_med_um", "radius_med_vox", "r_eq_cv"],
    why: "Normal median radius but a local pinch — minimum section radius under the 1st " +
         "percentile. A partial break rather than a thin strut.",
  },
  nominal: {
    keys: ["r_eq_med_um", "radius_med_vox", "iou"],
    why: "Everything the rules above did not claim.",
  },
};
const MAX_TILES = 4;

const DETECT_PROMPT = `Run the canonical strut defect detection through the MCP tools, in the order AGENTS.md gives. Pass actor= and why= on every call.

1. begin_analysis with this request verbatim.
2. refit_lattice_registration to produce correction.json.
3. detect_lattice_defects on mask_filepath="data/9x9x9_octet_lattice/segmentation/mask.tif" and design_filepath="data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json", passing correction_filepath from step 2, writing to a fresh output_directory under outputs/.
4. explain_run(style="story").

Report the per-class counts when done. Do not route around the MCP server.`;

const vizState = { runs: [], run: null, data: null, selected: "missing", rendering: false };

async function loadDefectRuns() {
  const runs = await (await fetch("/api/defect-runs")).json();
  vizState.runs = runs;
  const select = $("run-select");
  select.replaceChildren();
  if (!runs.length) {
    select.append(el("option", null, "no runs found"));
    return;
  }
  runs.forEach((r) => {
    const option = el("option", null,
      `${r.name}  ·  ${new Date(r.mtime * 1000).toLocaleString()}`);
    option.value = r.run;
    select.append(option);
  });
  // Newest first. Every run is equally legible now that struts.csv and connectivity.npz
  // are merged in, so there is no reason to prefer the one with the fat CSV.
  select.value = runs[0].run;
  await loadDefects(runs[0].run);
}

async function loadDefects(run) {
  const response = await fetch(`/api/defects?run=${encodeURIComponent(run)}`);
  if (!response.ok) {
    $("class-note").textContent = "could not read this run";
    return;
  }
  vizState.run = run;
  vizState.data = await response.json();
  renderClasses();
  renderRunFigures();
}

function renderClasses() {
  const d = vizState.data;
  const defects = d.order.filter((c) => c !== "nominal");
  const nominal = d.counts.nominal || 0;
  const flagged = d.total - nominal;

  $("class-note").textContent = `${commas(d.total)} struts`;
  $("class-hero").innerHTML =
    `${commas(flagged)} flagged` +
    `<small>of ${commas(d.total)} struts · ${(100 * flagged / d.total).toFixed(1)}%</small>`;

  // Scale to the largest DEFECT class, not the total: nominal is 80% of the lattice and
  // would flatten every bar that matters into a sliver.
  const max = Math.max(...defects.map((c) => d.counts[c] || 0), 1);
  const rows = $("class-rows");
  rows.replaceChildren();

  defects.forEach((name) => {
    const count = d.counts[name] || 0;
    const row = el("div", "class-row");
    row.dataset.class = name;
    if (name === vizState.selected) row.classList.add("is-selected");

    const swatch = el("span", "class-swatch");
    swatch.style.background = d.colors[name] || "#888";
    row.append(swatch, el("span", "class-name", name));

    const track = el("span", "class-track");
    const fill = document.createElement("i");
    fill.style.width = `${(count / max) * 100}%`;
    fill.style.background = d.colors[name] || "#888";
    track.append(fill);
    row.append(track);

    row.append(el("span", "class-count", commas(count)));
    row.append(el("span", "class-pct", `${(100 * count / d.total).toFixed(2)}%`));

    row.addEventListener("click", () => {
      vizState.selected = name;
      renderClasses();
    });
    rows.append(row);
  });

  $("class-context").innerHTML =
    `<code>nominal</code> ${commas(nominal)} (${(100 * nominal / d.total).toFixed(1)}%) — ` +
    `not shown as a bar; it is context, and at this scale it would flatten the rest.`;

  renderEvidence();
}

function renderEvidence() {
  const d = vizState.data;
  const name = vizState.selected;
  const spec = CLASS_EVIDENCE[name] || { keys: [], why: "" };
  const stats = (d.per_class || {})[name] || {};

  $("evidence-note").textContent = `${name} · ${commas(d.counts[name] || 0)} struts`;
  const box = $("evidence");
  box.replaceChildren();

  const present = spec.keys.filter((k) => stats[k]).slice(0, MAX_TILES);
  if (!present.length) {
    box.append(el("p", "empty-body",
      "This run's table does not carry the columns that drive this class. " +
      "outputs/lattice_iou/ has the full 16-column table."));
  }
  present.forEach((key) => {
    const s = stats[key];
    const tile = el("div", "metric");
    tile.append(el("span", "metric-key", key));
    tile.append(el("span", "metric-val", fmtNum(s.median)));
    tile.append(el("span", "metric-range", `${fmtNum(s.min)} – ${fmtNum(s.max)}  (n=${commas(s.n)})`));
    box.append(tile);
  });

  $("evidence-why").textContent = spec.why;
}

const fmtNum = (v) =>
  Math.abs(v) >= 1000 ? commas(Math.round(v))
    : Math.abs(v) >= 10 ? v.toFixed(1)
    : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);

const fmtBytes = (n) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${Math.round(n / 1e3)} KB` : `${n} B`;

function thumb(item) {
  const card = el("div", "thumb");
  if (item.kind === "image") {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/api/file?path=${encodeURIComponent(item.path)}`;
    img.alt = item.name;
    card.append(img);
    card.addEventListener("click", () => openLightbox(item));
  } else {
    const box = el("div", "thumb-page");
    box.innerHTML = `<b>3D</b>${item.embeddable ? "opens in the 3D tab"
                                                : `${fmtBytes(item.bytes)} — opens in the 3D tab`}`;
    card.append(box);
    card.addEventListener("click", () => openIn3d(item.path));
  }
  const meta = el("div", "thumb-meta");
  meta.append(el("span", "thumb-name", item.name));
  meta.append(el("span", "thumb-size", fmtBytes(item.bytes)));
  card.append(meta);
  card.title = item.path;
  return card;
}

function openLightbox(item) {
  $("lightbox-img").src = `/api/file?path=${encodeURIComponent(item.path)}`;
  $("lightbox-cap").textContent = item.path;
  $("lightbox").hidden = false;
}

function renderRunFigures() {
  const figures = vizState.data.figures || [];
  $("run-figures-note").textContent = `${figures.length} file${figures.length === 1 ? "" : "s"}`;
  const grid = $("run-figures");
  grid.replaceChildren();
  figures.forEach((f) => grid.append(thumb(f)));

  const s = vizState.data.summary || {};
  const design = (s.design || "").split("/").pop();
  $("run-provenance").textContent = design
    ? `design ${design} · correction ${s.correction_applied ? "applied" : "NOT applied"}`
    : "no summary.json for this run";
}

async function loadGallery() {
  const d = await (await fetch("/api/gallery")).json();
  $("gallery-note").textContent =
    `${d.total} files · pages over ${fmtBytes(d.embed_limit)} open in a tab`;
  const host = $("gallery");
  host.replaceChildren();
  d.groups.forEach((group) => {
    const details = el("details", "group");
    const summary = document.createElement("summary");
    summary.innerHTML = `<span class="group-path">${escapeHtml(group.group)}</span>` +
      `<span class="group-count">${group.items.length}</span>`;
    details.append(summary);
    const grid = el("div", "gallery-grid");
    // Build thumbnails only when the group is first opened — 138 eager <img> tags would
    // hit the server all at once on load.
    let built = false;
    details.addEventListener("toggle", () => {
      if (details.open && !built) {
        group.items.forEach((item) => grid.append(thumb(item)));
        built = true;
      }
    });
    details.append(grid);
    host.append(details);
  });
}

/* Render every figure for the selected run. The work happens in make_figures.py, which
 * calls the repo's own renderers; this only streams its progress. */
async function makeFigures() {
  if (!vizState.run || vizState.rendering) return;
  vizState.rendering = true;
  $("make-figures").disabled = true;
  $("figrun").hidden = false;
  $("figrun-dir").hidden = true;
  $("figrun-steps").replaceChildren();
  $("figrun-note").textContent = "starting…";

  const seen = new Map();
  const put = (name, status, message) => {
    let row = seen.get(name);
    if (!row) {
      row = el("div", "figstep");
      row.append(el("span", "figstep-name", name));
      row.append(el("span", "status", ""));
      row.append(el("span", "figstep-msg", ""));
      seen.set(name, row);
      $("figrun-steps").append(row);
    }
    const chip = row.querySelector(".status");
    const tone = { ok: "ok", failed: "fail", skipped: "skip" }[status] || "running";
    chip.className = `status status-${tone}`;
    chip.textContent = status === "start" ? "running" : status;
    row.querySelector(".figstep-msg").textContent = message || "";
    $("figrun-steps").scrollTop = $("figrun-steps").scrollHeight;
  };

  try {
    const response = await fetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: vizState.run }),
    });
    if (!response.ok) {
      put("request", "failed", await response.text());
      return;
    }
    let failures = 0;
    for await (const ev of sseEvents(response)) {
      if (ev.type === "progress") {
        put(ev.step || "step", ev.status || "start", ev.message || "");
        if (ev.status === "failed") failures += 1;
        if (ev.step === "done") {
          $("figrun-path").textContent = ev.directory || "";
          $("figrun-dir").hidden = false;
        }
      } else if (ev.type === "stderr") {
        addDiagnostic(ev.line);
      } else if (ev.type === "run.exited") {
        $("figrun-note").textContent =
          failures ? `finished with ${failures} step${failures === 1 ? "" : "s"} failed`
                   : "finished";
      }
    }
    await loadDefects(vizState.run); // pick up whatever was just written
    loadGallery();
  } catch (err) {
    put("stream", "failed", String(err));
  } finally {
    vizState.rendering = false;
    $("make-figures").disabled = false;
  }
}

/* Every .html the server can serve, newest run first, so the 3D tab can embed one
 * without leaving the dashboard. Sourced from the gallery, which already indexes them. */
async function load3dPages() {
  const select = $("view3d-select");
  const gallery = await (await fetch("/api/gallery")).json();
  const pages = [];
  gallery.groups.forEach((group) => group.items.forEach((item) => {
    if (item.kind === "page") pages.push(item);
  }));
  pages.sort((a, b) => b.mtime - a.mtime);
  vizState.pages = pages;

  select.replaceChildren();
  if (!pages.length) {
    select.append(el("option", null, "no .html pages found"));
    $("view3d-empty").hidden = false;
    return;
  }
  pages.forEach((page) => {
    const option = el("option", null, `${page.path}  ·  ${fmtBytes(page.bytes)}`);
    option.value = page.path;
    select.append(option);
  });

  // Prefer an interactive class viewer from the selected run over, say, a 60 MB
  // design-vs-as-built page that would take a while to paint.
  const preferred =
    pages.find((p) => vizState.run && p.path.startsWith(vizState.run) &&
                      p.name === "strut_classes_interactive.html") ||
    pages.find((p) => p.name === "strut_classes_interactive.html") ||
    pages[0];
  select.value = preferred.path;
  show3dPage(preferred.path);
}

/* Jump to the 3D tab with a specific page loaded, from anywhere. */
async function openIn3d(path) {
  showTab("3d");
  if (!vizState.pages) await load3dPages();
  const select = $("view3d-select");
  if ([...select.options].some((o) => o.value === path)) select.value = path;
  show3dPage(path);
}

function show3dPage(path) {
  const page = (vizState.pages || []).find((p) => p.path === path);
  $("view3d-empty").hidden = true;
  $("view3d").src = `/api/file?path=${encodeURIComponent(path)}`;
  $("view3d-note").textContent = page
    ? `${fmtBytes(page.bytes)} · ${new Date(page.mtime * 1000).toLocaleString()}`
    : path;
}

function showTab(which) {
  const panels = { chat: "panel-chat", viz: "panel-viz", "3d": "panel-3d" };
  const buttons = { chat: "tab-btn-chat", viz: "tab-btn-viz", "3d": "tab-btn-3d" };
  const active = panels[which] ? which : "chat";
  Object.entries(panels).forEach(([key, id]) => { $(id).hidden = key !== active; });
  Object.entries(buttons).forEach(([key, id]) => {
    $(id).classList.toggle("is-active", key === active);
    $(id).setAttribute("aria-selected", String(key === active));
  });

  if (active === "viz" && !vizState.data) {
    loadDefectRuns();
    loadGallery();
  }
  if (active === "3d" && !vizState.pages) {
    // The run list drives which page is preferred, so make sure it is loaded first.
    (vizState.data ? Promise.resolve() : loadDefectRuns()).then(load3dPages);
  }
}

/* ---------- wiring ---------- */

function exportTrace() {
  const body = state.trace.map((p) => JSON.stringify(p)).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([body], { type: "application/x-ndjson" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `codex_trace_${(state.threadId || "run").slice(0, 8)}.jsonl`;
  a.click();
  URL.revokeObjectURL(url);
}

function newThread() {
  if (state.trace.length && !confirm("Discard this trace and start a new thread?")) return;
  location.reload(); // all session state is client-side, so a reload is the reset
}

/* Replay a saved run through the same render path the live stream uses, so what
 * you see replayed is what you saw live. `?replay=<run_id>&speed=<ms>` */
async function replay(name, stepMs) {
  const response = await fetch(`/api/runs/${encodeURIComponent(name)}`);
  if (!response.ok) {
    addMessage("agent", `No saved run named ${name}.`);
    return;
  }
  $("chat-empty")?.remove();
  $("composer-hint").textContent = `replaying ${name}`;
  const lines = (await response.text()).split("\n").filter((l) => l.trim());
  for (const line of lines) {
    try {
      handleEvent(JSON.parse(line));
    } catch {
      addDiagnostic(`unparsed replay line: ${line.slice(0, 200)}`);
    }
    if (stepMs) await new Promise((r) => setTimeout(r, stepMs));
  }
  $("composer-hint").textContent = `replayed ${lines.length} events from ${name}`;
  refreshStats();
}

function init() {
  const box = $("suggestions");
  SUGGESTIONS.forEach((text) => {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => send(text));
    box.append(chip);
  });

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("prompt").value.trim();
    if (text) send(text);
  });

  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });

  $("cancel").addEventListener("click", async () => {
    if (!state.runId) return;
    await fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: state.runId }),
    });
  });

  $("export").addEventListener("click", exportTrace);
  $("new-thread").addEventListener("click", newThread);

  $("tab-btn-chat").addEventListener("click", () => showTab("chat"));
  $("tab-btn-viz").addEventListener("click", () => showTab("viz"));
  $("tab-btn-3d").addEventListener("click", () => showTab("3d"));
  $("view3d-select").addEventListener("change", (e) => show3dPage(e.target.value));
  $("view3d-reload").addEventListener("click", () => {
    if ($("view3d").src) $("view3d").src = $("view3d").src;
  });
  $("view3d-pop").addEventListener("click", () => {
    const path = $("view3d-select").value;
    if (path) window.open(`/api/file?path=${encodeURIComponent(path)}`, "_blank", "noopener");
  });
  $("run-select").addEventListener("change", (e) => loadDefects(e.target.value));

  // Detection is the MCP server's job, so this hands the canonical prompt to the agent
  // rather than computing anything here.
  $("run-detect").addEventListener("click", () => {
    if (state.running) return;
    if ($("sandbox").value === "read-only" &&
        !confirm("detect_lattice_defects writes a results directory, which the read-only " +
                 "sandbox forbids. Switch to workspace-write and run?")) return;
    if ($("sandbox").value === "read-only") $("sandbox").value = "workspace-write";
    showTab("chat");
    send(DETECT_PROMPT);
  });

  $("make-figures").addEventListener("click", makeFigures);
  $("figrun-hide").addEventListener("click", () => { $("figrun").hidden = true; });
  $("figrun-copy").addEventListener("click", () => {
    const path = $("figrun-path").textContent;
    navigator.clipboard?.writeText(path).then(
      () => { $("figrun-copy").textContent = "Copied"; },
      () => { $("figrun-copy").textContent = "Copy failed"; });
    setTimeout(() => { $("figrun-copy").textContent = "Copy path"; }, 1500);
  });

  $("lightbox-close").addEventListener("click", () => { $("lightbox").hidden = true; });
  $("lightbox").addEventListener("click", (e) => {
    if (e.target === $("lightbox")) $("lightbox").hidden = true;
  });
  addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("lightbox").hidden = true;
  });

  $("sandbox").addEventListener("change", () => {
    $("chat-note").textContent = `${$("sandbox").value} sandbox`;
  });

  $("theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme;
    const isDark = now
      ? now === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("codex-dash-theme", next); } catch {}
  });

  fetch("/api/config")
    .then((r) => r.json())
    .then((cfg) => {
      $("repo-line").textContent =
        `${cfg.repo_name} · ${cfg.branch || "?"} · ${cfg.dirty_files} uncommitted · ${cfg.codex_version}`;
      $("repo-line").title = `${cfg.repo}\n${cfg.head}`;
      $("sandbox").value = cfg.default_sandbox;
      $("chat-note").textContent = `${cfg.default_sandbox} sandbox`;
    })
    .catch(() => { $("repo-line").textContent = "could not reach the server"; });

  refreshStats();

  const params = new URLSearchParams(location.search);
  if (params.get("theme")) document.documentElement.dataset.theme = params.get("theme");
  if (params.get("class")) vizState.selected = params.get("class");
  if (params.get("tab")) showTab(params.get("tab"));
  if (params.get("replay")) {
    replay(params.get("replay"), Number(params.get("speed") || 0));
  }
}

init();
