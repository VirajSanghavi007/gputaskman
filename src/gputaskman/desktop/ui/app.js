/* GPU Task Manager window. Reads state from the Python side (pywebview js_api) and draws it.
   Provenance: null means unknown and is drawn as "n/a" or a gap, never as 0. Inferred VRAM is marked "~". */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  data: null, info: null, page: "processes", sortKey: "vram", sortDir: -1,
  filter: "", selected: null, perfSel: "gpu0", rendered: 0, timer: null, busy: false,
};

/* ---------- formatting ---------- */
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function fmtBytes(b) {
  if (!isNum(b)) return null;
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = b;
  while (Math.abs(v) >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v.toFixed(0) : v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " " + u[i];
}
const fmtPct = (v) => (isNum(v) ? (v >= 10 ? v.toFixed(0) : v.toFixed(1)) + "%" : null);
const orNA = (s) => (s === null || s === undefined || s === "" ? "n/a" : s);
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

/* Task Manager style heat tint: stronger yellow for a bigger share of the column's scale. */
function heat(frac) {
  if (!isNum(frac) || frac <= 0) return "";
  const a = Math.min(0.55, 0.06 + frac * 0.5);
  return `background-color: rgba(${css("--heat")}, ${a.toFixed(3)})`;
}

/* ---------- API ---------- */
function api() { return window.pywebview && window.pywebview.api; }

async function poll() {
  if (state.busy || !api()) return;
  state.busy = true;
  try {
    const data = await api().get_state();
    if (data) { state.data = data; render(); }
  } catch (e) {
    setStatus(`<span class="err">Could not read state: ${esc(e && e.message ? e.message : e)}</span>`);
  } finally {
    state.busy = false;
  }
}

async function start() {
  if (state.timer) return;
  try { state.info = await api().get_info(); } catch (_) { state.info = null; }
  if (state.info) $("version").textContent = "v" + state.info.version;
  const interval = Math.max(250, ((state.info && state.info.interval) || 1) * 1000);
  await poll();
  state.timer = setInterval(poll, interval);
}

/* ---------- processes ---------- */
function rowVram(r) { return isNum(r.vram) ? r.vram : isNum(r.vram_inferred) ? r.vram_inferred : null; }
function sortValue(r, key) {
  if (key === "vram") return rowVram(r);
  if (key === "engine") return r.engine || "";
  return r[key];
}
function compare(a, b, key, dir) {
  const va = sortValue(a, key), vb = sortValue(b, key);
  const na = va === null || va === undefined || va === "", nb = vb === null || vb === undefined || vb === "";
  if (na && nb) return a.pid - b.pid;
  if (na) return 1; // unknown always sorts last, whichever direction
  if (nb) return -1;
  if (typeof va === "string") return dir * va.localeCompare(vb, undefined, { sensitivity: "base" }) || a.pid - b.pid;
  return dir * (va - vb) || a.pid - b.pid;
}
function matches(r, q) {
  if (!q) return true;
  return [r.name, String(r.pid), r.runtime, r.kind, r.inside, (r.models || []).join(" ")]
    .some((s) => (s || "").toLowerCase().includes(q));
}
function rowKey(r) { return `${r.host || ""}|${r.pid}|${r.name}`; }

function visibleRows() {
  const rows = (state.data && state.data.processes) || [];
  const q = state.filter.trim().toLowerCase();
  return rows.filter((r) => matches(r, q)).sort((a, b) => compare(a, b, state.sortKey, state.sortDir));
}

function renderProcesses() {
  const data = state.data;
  const rows = visibleRows();
  const gpus = data.gpus || [];
  const vramTotal = gpus.reduce((s, g) => s + (isNum(g.vram_total) ? g.vram_total : 0), 0);
  const sharedTotal = gpus.reduce((s, g) => s + (isNum(g.shared_total) ? g.shared_total : 0), 0);

  // Column totals: the adapter-wide readings (measured), the same figures Task Manager puts on top.
  const utils = gpus.map((g) => g.util).filter(isNum);
  $("t-gpu").textContent = utils.length ? fmtPct(Math.max(...utils)) : "n/a";
  const vu = gpus.map((g) => g.vram_used).filter(isNum);
  $("t-vram").textContent = vu.length && vramTotal ? Math.round((vu.reduce((a, b) => a + b, 0) / vramTotal) * 100) + "%" : "n/a";
  const su = gpus.map((g) => g.shared_used).filter(isNum);
  $("t-shared").textContent = su.length && sharedTotal ? Math.round((su.reduce((a, b) => a + b, 0) / sharedTotal) * 100) + "%" : "n/a";

  document.querySelectorAll("#proc-table th").forEach((th) => {
    const on = th.dataset.key === state.sortKey;
    th.classList.toggle("sorted", on);
    th.dataset.dir = on ? (state.sortDir < 0 ? "▼" : "▲") : "";
  });

  const html = rows.map((r) => {
    const v = rowVram(r);
    const inferred = !isNum(r.vram) && isNum(r.vram_inferred);
    const vramText = v === null ? "n/a" : (inferred ? "~" : "") + fmtBytes(v);
    const vramTitle = inferred ? `Inferred${r.vram_inferred_by ? " from " + r.vram_inferred_by : ""}, not measured` : "Measured";
    const sh = fmtBytes(r.shared);
    const pct = fmtPct(r.gpu_pct);
    const badges = (r.models && r.models.length ? `<span class="badge ai" title="${esc(r.models.join(", "))}">model</span>` : "")
      + (r.host === "windows" || r.host === "wsl" ? `<span class="badge">${esc(r.host)}</span>` : "");
    const sel = state.selected === rowKey(r) ? " selected" : "";
    return `<tr data-key="${esc(rowKey(r))}" class="${sel.trim()}">`
      + `<td class="name" title="${esc(r.inside || r.name)}">${esc(r.name)}${badges}</td>`
      + `<td class="num">${esc(r.pid)}</td>`
      + `<td class="${r.kind ? "" : "na"}">${esc(r.kind || "")}</td>`
      + `<td class="runtime" title="${esc(r.runtime)}">${esc(r.runtime)}</td>`
      + `<td class="num${pct ? "" : " na"}" style="${heat(isNum(r.gpu_pct) ? r.gpu_pct / 100 : 0)}">${pct || "n/a"}</td>`
      + `<td class="engine${r.engine ? "" : " na"}" title="${esc(r.engine_reason || "")}">${esc(r.engine || "")}</td>`
      + `<td class="num${v === null ? " na" : ""}${inferred ? " inferred" : ""}" title="${esc(vramTitle)}" style="${heat(v !== null && vramTotal ? v / vramTotal : 0)}">${esc(vramText)}</td>`
      + `<td class="num${sh ? "" : " na"}" style="${heat(isNum(r.shared) && sharedTotal ? r.shared / sharedTotal : 0)}">${esc(sh || "n/a")}</td>`
      + `</tr>`;
  }).join("");
  $("proc-body").innerHTML = html;
  state.rendered = rows.length;
  $("proc-empty").hidden = rows.length > 0;
  $("proc-empty").textContent = data.ready ? (state.filter ? "No process matches the search." : "No GPU processes right now.") : "Waiting for the first sample...";

  const selRow = rows.find((r) => rowKey(r) === state.selected);
  if (!selRow) state.selected = null;
  $("end-task").disabled = !selRow;
  renderDetails(selRow);
}

const SEG_COLORS = { weights: "--seg-weights", kv: "--seg-kv", compute: "--seg-compute", runtime: "--seg-runtime" };
function renderDetails(r) {
  const box = $("details");
  if (!r) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const started = isNum(r.started_at) ? new Date(r.started_at * 1000).toLocaleString() : null;
  const facts = [
    ["Process", `${r.name} (PID ${r.pid})`],
    ["GPUs", (r.gpus || []).map((i) => "GPU " + i).join(", ")],
    ["Runtime", r.runtime],
    ["Type", r.kind],
    ["Dedicated (measured)", fmtBytes(r.vram)],
    ["Dedicated (inferred)", isNum(r.vram_inferred) ? "~" + fmtBytes(r.vram_inferred) + (r.vram_inferred_by ? " via " + r.vram_inferred_by : "") : null],
    ["Shared GPU memory", fmtBytes(r.shared)],
    ["Started", started],
    ["Inside", r.inside],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  let html = `<h3>${esc(r.name)}</h3><div class="facts">`
    + facts.map(([k, v]) => `<div class="fact"><div class="k">${esc(k)}</div><div class="v" title="${esc(v)}">${esc(v)}</div></div>`).join("")
    + `</div>`;
  if (r.models && r.models.length) html += `<div class="note">Models: ${esc(r.models.join(", "))}</div>`;
  const segs = (r.segments || []).filter((s) => isNum(s.bytes) && s.bytes > 0);
  if (segs.length) {
    const total = segs.reduce((a, s) => a + s.bytes, 0);
    html += `<div class="segbar">` + segs.map((s) =>
      `<div title="${esc(s.label)}: ${esc(fmtBytes(s.bytes))}${s.exact ? "" : " (estimate)"}" style="width:${(s.bytes / total * 100).toFixed(2)}%;background:var(${SEG_COLORS[s.kind] || "--seg-unknown"})"></div>`).join("") + `</div>`;
    html += `<div class="legend">` + segs.map((s) =>
      `<span><i style="background:var(${SEG_COLORS[s.kind] || "--seg-unknown"})"></i>${esc(s.label)} ${s.exact ? "" : "~"}${esc(fmtBytes(s.bytes))}</span>`).join("") + `</div>`;
    html += `<div class="note">~ marks an estimate. Segments are what the runtime reports or gputaskman estimates, not a measurement of each buffer.</div>`;
  }
  if (r.engine_reason) html += `<div class="note">GPU engines: ${esc(r.engine_reason)}</div>`;
  box.innerHTML = html;
}

/* ---------- charts ---------- */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

/* Draw a series scaled to [0, max]. null values are gaps: the line breaks and no fill is drawn there. */
function drawSeries(cv, values, max, color, { grid = true, size = 60 } = {}) {
  const { ctx, w, h, dpr } = setupCanvas(cv);
  if (grid) {
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) { const x = Math.round((w * i) / 10) + 0.5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let i = 1; i < 5; i++) { const y = Math.round((h * i) / 5) + 0.5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  }
  if (!isNum(max) || max <= 0) return;
  const n = Math.max(size, values.length);
  const step = w / (n - 1 || 1);
  const off = n - values.length;
  const pt = (i) => [(off + i) * step, h - Math.max(0, Math.min(1, values[i] / max)) * (h - 2 * dpr) - dpr];
  // Split into runs of known values.
  const runs = [];
  let run = [];
  values.forEach((v, i) => { if (isNum(v)) run.push(i); else if (run.length) { runs.push(run); run = []; } });
  if (run.length) runs.push(run);
  for (const r of runs) {
    ctx.beginPath();
    r.forEach((i, k) => { const [x, y] = pt(i); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    if (r.length === 1) { const [x, y] = pt(r[0]); ctx.lineTo(x + 1, y); }
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
    const [x0] = pt(r[0]), [x1] = pt(r[r.length - 1]);
    ctx.lineTo(x1, h); ctx.lineTo(x0, h); ctx.closePath();
    ctx.globalAlpha = 0.14; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
  }
}

/* ---------- performance ---------- */
function perfItems() {
  const d = state.data, items = [];
  for (const g of d.gpus || []) {
    items.push({
      id: "gpu" + g.index, title: "GPU " + g.index, sub: g.name,
      line: g.error ? "Unavailable" : [fmtPct(g.util), isNum(g.temp) ? g.temp + " °C" : null].filter(Boolean).join(" (") + (isNum(g.temp) && isNum(g.util) ? ")" : ""),
      series: (d.history || {})["gpu" + g.index + ".util"] || [], max: 100, color: "--c-gpu",
    });
  }
  const s = d.system;
  items.push({ id: "cpu", title: "CPU", sub: s && s.logical ? `${s.logical} logical processors` : "", line: s ? orNA(fmtPct(s.cpu_pct)) : "n/a",
    series: (d.history || {}).cpu || [], max: 100, color: "--c-cpu" });
  items.push({ id: "mem", title: "Memory", sub: s && isNum(s.ram_total) ? fmtBytes(s.ram_total) : "",
    line: s && isNum(s.ram_used) && isNum(s.ram_total) ? `${fmtBytes(s.ram_used)} / ${fmtBytes(s.ram_total)} (${Math.round(s.ram_used / s.ram_total * 100)}%)` : "n/a",
    series: (d.history || {}).ram || [], max: s ? s.ram_total : null, color: "--c-mem" });
  return items;
}

function renderPerformance() {
  const items = perfItems();
  if (!items.some((it) => it.id === state.perfSel)) state.perfSel = items[0].id;
  const list = $("perf-list");
  if (list.dataset.ids !== items.map((i) => i.id).join(",")) {
    list.dataset.ids = items.map((i) => i.id).join(",");
    list.innerHTML = items.map((it) => `<div class="perf-item" role="option" data-id="${it.id}"><canvas></canvas><div class="info"><div class="t"></div><div class="s sub"></div><div class="s line"></div></div></div>`).join("");
  }
  for (const it of items) {
    const el = list.querySelector(`[data-id="${it.id}"]`);
    el.classList.toggle("active", it.id === state.perfSel);
    el.setAttribute("aria-selected", it.id === state.perfSel);
    el.querySelector(".t").textContent = it.title;
    el.querySelector(".sub").textContent = it.sub;
    el.querySelector(".sub").title = it.sub;
    el.querySelector(".line").textContent = it.line;
    drawSeries(el.querySelector("canvas"), it.series, it.max, css(it.color), { grid: false, size: state.data.history_size || 60 });
  }
  const sel = state.perfSel;
  if (sel.startsWith("gpu")) renderGpuDetail(state.data.gpus.find((g) => "gpu" + g.index === sel));
  else if (sel === "cpu") renderCpuDetail();
  else renderMemDetail();
}

function detailShell(id, title, model, charts, stats, notes) {
  const box = $("perf-detail");
  if (box.dataset.view !== id) {
    box.dataset.view = id;
    box.innerHTML = `<div class="perf-head"><h2></h2><div class="model"></div></div><div class="charts"></div><div class="stats"></div><div class="notes" hidden></div>`;
    box.querySelector(".charts").innerHTML = charts.map((c) =>
      (c.pairStart ? `<div class="chart-pair">` : "")
      + `<div class="chart-box"><div class="chart-label"><span>${esc(c.label)}</span><span class="max"></span></div><canvas data-c="${c.key}" style="height:${c.height || 150}px"></canvas></div>`
      + (c.pairEnd ? `</div>` : "")).join("");
  }
  box.querySelector("h2").textContent = title;
  box.querySelector(".model").textContent = model || "";
  box.querySelector(".model").title = model || "";
  for (const c of charts) {
    const cv = box.querySelector(`canvas[data-c="${c.key}"]`);
    cv.parentElement.querySelector(".max").textContent = c.maxLabel || "";
    drawSeries(cv, c.series, c.max, css(c.color), { size: state.data.history_size || 60 });
  }
  box.querySelector(".stats").innerHTML = stats.map(([k, v, small]) =>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v${small ? " small" : ""}">${esc(orNA(v))}</div></div>`).join("");
  const n = box.querySelector(".notes");
  n.hidden = !notes || !notes.length;
  n.innerHTML = (notes || []).map(esc).join("<br>");
}

function renderGpuDetail(g) {
  const h = state.data.history || {};
  const i = g.index;
  const charts = [
    { key: "util", label: "Utilization (60 s)", maxLabel: "100%", series: h[`gpu${i}.util`] || [], max: 100, color: "--c-gpu", height: 190 },
    { key: "vram", label: "Dedicated GPU memory usage", maxLabel: orNA(fmtBytes(g.vram_total)), series: h[`gpu${i}.vram`] || [], max: g.vram_total, color: "--c-gpu", pairStart: true },
    { key: "shared", label: "Shared GPU memory usage", maxLabel: orNA(fmtBytes(g.shared_total)), series: h[`gpu${i}.shared`] || [], max: g.shared_total, color: "--c-gpu-2", pairEnd: true },
  ];
  const mem = (u, t) => (isNum(u) ? `${fmtBytes(u)}${isNum(t) ? " / " + fmtBytes(t) : ""}` : null);
  const power = isNum(g.power_w) ? `${g.power_w.toFixed(1)} W${isNum(g.power_limit_w) ? " / " + g.power_limit_w.toFixed(0) + " W" : ""}` : null;
  const clock = isNum(g.sm_clock) ? `${g.sm_clock}${isNum(g.sm_clock_max) ? " / " + g.sm_clock_max : ""} MHz` : null;
  const pcie = isNum(g.pcie_rx_kbs) || isNum(g.pcie_tx_kbs)
    ? `${isNum(g.pcie_rx_kbs) ? fmtBytes(g.pcie_rx_kbs * 1024) + "/s" : "n/a"} in, ${isNum(g.pcie_tx_kbs) ? fmtBytes(g.pcie_tx_kbs * 1024) + "/s" : "n/a"} out` : null;
  const stats = [
    ["Utilization", fmtPct(g.util)],
    ["Dedicated GPU memory", mem(g.vram_used, g.vram_total)],
    ["Shared GPU memory", mem(g.shared_used, g.shared_total)],
    ["Attributed to processes", fmtBytes(g.attributed)],
    ["Temperature", isNum(g.temp) ? g.temp + " °C" : null],
    ["Power", power],
    ["Graphics clock", clock],
    ["Memory clock", isNum(g.mem_clock) ? g.mem_clock + " MHz" : null],
    ["Memory controller", fmtPct(g.util_mem)],
    ["Video encode / decode", isNum(g.enc) || isNum(g.dec) ? `${orNA(fmtPct(g.enc))} / ${orNA(fmtPct(g.dec))}` : null],
    ["Fan", fmtPct(g.fan)],
    ["Performance state", isNum(g.pstate) ? "P" + g.pstate : g.pstate],
    ["PCIe", pcie, true],
    ["Driver", [g.driver, g.driver_model].filter(Boolean).join(" ") || null, true],
    ["CUDA", g.cuda, true],
    ["Location", g.pci, true],
    ["Backend", g.backend, true],
    ["Processes", String((g.processes || []).length)],
  ];
  detailShell("gpu" + i, "GPU " + i, g.name, charts, stats, g.notes);
}

function renderCpuDetail() {
  const s = state.data.system, h = state.data.history || {};
  const charts = [{ key: "cpu", label: "% Utilization (60 s)", maxLabel: "100%", series: h.cpu || [], max: 100, color: "--c-cpu", height: 220 }];
  const stats = [
    ["Utilization", s ? fmtPct(s.cpu_pct) : null],
    ["Logical processors", s && s.logical != null ? String(s.logical) : null],
    ["Cores", s && s.physical != null ? String(s.physical) : null],
    ["Busiest core", s && s.per_core && s.per_core.filter(isNum).length ? fmtPct(Math.max(...s.per_core.filter(isNum))) : null],
  ];
  detailShell("cpu", "CPU", s ? "" : "System readings unavailable", charts, stats, s ? [] : ["System RAM/CPU could not be read."]);
}

function renderMemDetail() {
  const s = state.data.system, h = state.data.history || {};
  const charts = [{ key: "ram", label: "Memory usage (60 s)", maxLabel: s ? orNA(fmtBytes(s.ram_total)) : "", series: h.ram || [], max: s ? s.ram_total : null, color: "--c-mem", height: 220 }];
  const stats = [
    ["In use", s ? fmtBytes(s.ram_used) : null],
    ["Available", s ? fmtBytes(s.ram_available) : null],
    ["Total", s ? fmtBytes(s.ram_total) : null],
    ["Swap / page file", s && isNum(s.swap_used) ? `${fmtBytes(s.swap_used)} / ${orNA(fmtBytes(s.swap_total))}` : null],
  ];
  detailShell("mem", "Memory", s && isNum(s.ram_total) ? fmtBytes(s.ram_total) : "", charts, stats, []);
}

/* ---------- shell ---------- */
function setStatus(html) { $("status").innerHTML = html; }

function render() {
  const d = state.data;
  if (!d) return;
  if (state.page === "processes") renderProcesses();
  else renderPerformance();
  const n = (d.processes || []).length;
  const taken = isNum(d.taken) ? new Date(d.taken * 1000).toLocaleTimeString() : "n/a";
  const failed = (d.gpus || []).filter((g) => g.error).map((g) => `GPU ${g.index}: ${esc(g.error)}`);
  setStatus(`<span>GPU processes: ${n}</span><span>Updated ${esc(taken)}</span>`
    + (d.error ? `<span class="err">${esc(d.error)}</span>` : "")
    + failed.map((f) => `<span class="err">${f}</span>`).join(""));
}

function showPage(page) {
  state.page = page;
  document.querySelectorAll(".nav").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  $("processes").hidden = page !== "processes";
  $("performance").hidden = page !== "performance";
  $("page-title").textContent = page === "processes" ? "Processes" : "Performance";
  document.querySelector(".search").style.visibility = page === "processes" ? "" : "hidden";
  $("end-task").style.visibility = page === "processes" ? "" : "hidden";
  $("perf-detail").dataset.view = "";
  render();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 4500);
}

async function endSelected() {
  const rows = (state.data && state.data.processes) || [];
  const r = rows.find((x) => rowKey(x) === state.selected);
  if (!r || !api()) return;
  if (r.host === "wsl") { toast("This process runs inside WSL. End it from the Linux side."); return; }
  $("confirm-title").textContent = `End ${r.name}?`;
  $("confirm-text").textContent = `PID ${r.pid}. Unsaved work in this program will be lost. gputaskman checks the PID still belongs to ${r.name} before ending it.`;
  const dlg = $("confirm");
  dlg.returnValue = "";
  dlg.showModal();
  const answer = await new Promise((res) => dlg.addEventListener("close", () => res(dlg.returnValue), { once: true }));
  if (answer !== "ok") return;
  try {
    const res = await api().end_task(r.pid, r.name, isNum(r.started_at) ? r.started_at : null);
    toast(res.message);
    if (res.ok) { state.selected = null; poll(); }
  } catch (e) {
    toast("End task failed: " + (e && e.message ? e.message : e));
  }
}

function wire() {
  document.querySelectorAll(".nav").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));
  document.querySelectorAll("#proc-table th").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.key;
    if (state.sortKey === k) state.sortDir = -state.sortDir;
    else { state.sortKey = k; state.sortDir = ["name", "kind", "runtime", "engine", "pid"].includes(k) ? 1 : -1; }
    render();
  }));
  $("search").addEventListener("input", (e) => { state.filter = e.target.value; render(); });
  $("proc-body").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    state.selected = state.selected === tr.dataset.key ? null : tr.dataset.key;
    render();
  });
  $("end-task").addEventListener("click", endSelected);
  $("perf-list").addEventListener("click", (e) => {
    const it = e.target.closest(".perf-item");
    if (it) { state.perfSel = it.dataset.id; render(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Delete" && state.page === "processes" && state.selected && document.activeElement !== $("search")) endSelected();
    if (e.key === "Escape" && state.selected) { state.selected = null; render(); }
    if ((e.ctrlKey && e.key.toLowerCase() === "f") && state.page === "processes") { e.preventDefault(); $("search").focus(); }
  });
  window.addEventListener("resize", () => { if (state.page === "performance") render(); });
}

/* Smoke tests (gputaskman-app --selftest) ask the page what it drew. */
window.gtmSelfTest = () => ({
  ready: !!(state.data && state.data.ready),
  gpus: state.data ? (state.data.gpus || []).length : 0,
  processes: state.data ? (state.data.processes || []).length : 0,
  rows: document.querySelectorAll("#proc-body tr").length,
  title: document.title,
  error: state.data ? state.data.error : null,
});
window.gtmState = state; // for debugging in the inspector

wire();
if (api()) start();
else window.addEventListener("pywebviewready", start);
