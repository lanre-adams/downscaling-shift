/* Downscaling shift test - dashboard. Plain JS, no build step, no dependencies. */
"use strict";

const MODELS = [
  { key: "qm", label: "Quantile mapping", color: "var(--qm)" },
  { key: "gbm", label: "ML downscaler", color: "var(--gbm)" },
  { key: "gbm_phys", label: "ML + warming prior", color: "var(--phys)" },
];
const S = { index: null, runs: {}, current: null, staticMode: false };
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";

/* ---------- helpers ---------- */
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function txt(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }
const fmt = (v, d = 1, sign = true) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a"
  : (sign && v > 0 ? "+" : "") + Number(v).toFixed(d);
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
const log = (d0, d1, r0, r1) => (v) => r0 + ((Math.log(v) - Math.log(d0)) / (Math.log(d1) - Math.log(d0) || 1)) * (r1 - r0);
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function svgFor(container, w, h) {
  container.innerHTML = "";
  return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, container);
}
const tip = $("tooltip");
function bindTip(node, html) {
  const show = (e) => { tip.innerHTML = html; tip.hidden = false; move(e); };
  const move = (e) => {
    const x = (e.clientX ?? 0) + 14, y = (e.clientY ?? 0) + 14;
    tip.style.left = Math.min(x, window.innerWidth - 280) + "px"; tip.style.top = y + "px";
  };
  node.addEventListener("mouseenter", show); node.addEventListener("mousemove", move);
  node.addEventListener("mouseleave", () => (tip.hidden = true));
  node.addEventListener("focus", (e) => { const r = node.getBoundingClientRect(); show({ clientX: r.right, clientY: r.top }); });
  node.addEventListener("blur", () => (tip.hidden = true));
  node.setAttribute("tabindex", "0");
}
function legend(container, items) {
  container.innerHTML = items.map((i) =>
    `<span><i class="${i.kind || ""}" style="${i.kind === "dash" ? "" : `background:${i.color}`}"></i>${i.label}</span>`).join("");
}
async function getJSON(url) { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

/* ---------- loading ---------- */
async function loadAll() {
  S.index = await getJSON("results/index.json");
  await Promise.all(S.index.runs.map(async (r) => { S.runs[r.id] = await getJSON("results/" + r.path); }));
  renderHero();
  renderTabs();
  select(S.current && S.runs[S.current] ? S.current : S.index.runs[0].id);
}

/* ---------- hero: bias vs warming ---------- */
const STANDARD = ["default", "future", "stress"];
function shiftedRuns() {
  // The hero compares the standard scenarios only (same grid and settings);
  // extra runs (yours, or real data) appear as tabs below.
  const all = S.index.runs.filter((r) => S.runs[r.id]);
  const std = all.filter((r) => STANDARD.includes(r.id));
  return (std.length >= 2 ? std : all).sort((a, b) => a.shift_K - b.shift_K);
}
function renderHero() {
  const runs = shiftedRuns();
  const src = S.runs[runs[0].id].meta.source;
  const status = $("data-status");
  status.hidden = false;
  status.textContent = src === "synthetic" ? "Synthetic data" : "Real data";
  status.classList.toggle("real", src !== "synthetic");

  // change in 99th-percentile rainfall, early -> late, for each scenario
  const series = [{ key: "obs", label: "Observed", color: "var(--obs)", dash: true }, ...MODELS].map((m) => ({
    ...m,
    pts: runs.map((r) => {
      const c = S.runs[r.id].change[m.key === "obs" ? "gbm" : m.key];
      return { K: r.shift_K, v: m.key === "obs" ? c.obs_p99_change_pct : c.pred_p99_change_pct, run: r };
    }),
  }));
  legend($("hero-legend"), series.map((m) => ({ color: m.color, label: m.label, kind: m.dash ? "dash" : "line" })));

  const W = 640, H = 360, l = 56, r = 120, t = 16, b = 44;
  const svg = svgFor($("hero-svg"), W, H);
  const allV = series.flatMap((s) => s.pts.map((p) => p.v));
  const kMax = Math.max(...runs.map((r) => r.shift_K)) * 1.08;
  const lo = Math.min(0, ...allV), hi = Math.max(...allV) * 1.08 + 1;
  const x = lin(0, kMax, l, W - r), y = lin(lo, hi, H - b, t);
  niceTicks(lo, hi, 6).forEach((v) => {
    el("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }, svg);
    txt(svg, l - 8, y(v) + 4, `${v > 0 ? "+" : ""}${v}%`, { "text-anchor": "end" });
  });
  niceTicks(0, kMax, 5).forEach((k) => txt(svg, x(k), H - b + 18, `${k} K`, { "text-anchor": "middle" }));
  txt(svg, (l + W - r) / 2, H - 6, "how much warmer the test years are than the training climate", { "text-anchor": "middle" });

  const ends = [];
  series.forEach((s) => {
    const d = s.pts.map((p, i) => `${i ? "L" : "M"}${x(p.K)},${y(p.v)}`).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.dash ? 2.5 : 2, "stroke-dasharray": s.dash ? "6 4" : "none" }, svg);
    s.pts.forEach((p) => {
      el("circle", { cx: x(p.K), cy: y(p.v), r: 5, fill: s.dash ? "#fff" : s.color, stroke: s.dash ? s.color : "#fff", "stroke-width": 2 }, svg);
      const hit = el("circle", { cx: x(p.K), cy: y(p.v), r: 12, class: "hit", "aria-label": `${s.label}, ${fmt(p.K, 1)} K: ${fmt(p.v, 0)}%` }, svg);
      bindTip(hit, `<b>${s.label}</b><br>${p.run.label}<br>${fmt(p.K, 2)} K warmer<br>Change in heavy rain ${fmt(p.v, 1)}%`);
    });
    const last = s.pts[s.pts.length - 1];
    ends.push({ y: y(last.v), x: x(last.K), label: `${s.label} ${fmt(last.v, 0)}%` });
  });
  // direct end labels, nudged apart so they never overlap
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
  ends.forEach((e) => txt(svg, e.x + 10, e.y + 4, e.label, { class: "ink", "font-size": 12 }));

  // hero sentence, computed from the numbers
  const first = runs[0], lastRun = runs[runs.length - 1];
  const g = (run, k) => S.runs[run.id].change[k];
  $("hero-finding").textContent =
    `With ${fmt(first.shift_K, 1, false)} K of warming, the ML downscaler predicts a ${fmt(g(first, "gbm").pred_p99_change_pct, 0)}% ` +
    `change in heavy rain against ${fmt(g(first, "gbm").obs_p99_change_pct, 0)}% observed. With ${fmt(lastRun.shift_K, 1, false)} K it predicts ` +
    `${fmt(g(lastRun, "gbm").pred_p99_change_pct, 0)}% against ${fmt(g(lastRun, "gbm").obs_p99_change_pct, 0)}%. ` +
    `Adding a warming prior gives ${fmt(g(lastRun, "gbm_phys").pred_p99_change_pct, 0)}%.` +
    (src === "synthetic" ? " These are results from a synthetic world: a test of the method, not of the real climate." : "");
}

/* ---------- scenario tabs ---------- */
function renderTabs() {
  const tabs = $("scenario-tabs");
  tabs.innerHTML = "";
  S.index.runs.forEach((r) => {
    const b = document.createElement("button");
    b.type = "button"; b.role = "tab"; b.dataset.id = r.id;
    b.textContent = r.id === "custom" ? "Your run" : r.label.replace(/ \(.*\)$/, "");
    b.addEventListener("click", () => select(r.id));
    tabs.appendChild(b);
  });
}

function select(id) {
  S.current = id;
  document.querySelectorAll("#scenario-tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.id === id));
  const res = S.runs[id], entry = S.index.runs.find((r) => r.id === id);
  const base = "results/" + entry.path.replace(/results\.json$/, "");
  const sp = res.splits, oor = res.out_of_range || {};
  $("scenario-summary").textContent =
    `${entry.label}. Trained on ${sp.train.n_days} days (${sp.train.years[0]}–${sp.train.years[sp.train.years.length - 1]}, minus held-out years); ` +
    `tested on ${sp.in_dist.n_days} held-out early days and ${sp.shifted.n_days} days from ${sp.shifted.years[0]}–${sp.shifted.years[sp.shifted.years.length - 1]}, ` +
    `${fmt(entry.shift_K, 2)} K warmer.` + (oor.shifted !== undefined ? ` ${fmt(100 * oor.shifted, 1, false)}% of shifted-period moisture values lie beyond the training range.` : "");
  renderTable(res);
  renderChange(res);
  renderQQ(res);
  renderSpectra(res);
  $("findings-list").innerHTML = res.findings.map((f) => `<li>${f}</li>`).join("");
  $("figures").innerHTML = res.figures.map((f) =>
    `<figure><img src="${base}${f.file}" alt="${f.title}" loading="lazy"><figcaption><strong>${f.title}.</strong> ${f.caption}</figcaption></figure>`).join("");
  loadReport(base);
  $("dl-md").href = base + "report.md";
  $("dl-json").href = base + "results.json";
}

function renderTable(res) {
  const s = res.scores;
  const head = `<thead><tr><th>Model</th><th>Mean bias, early</th><th>Mean bias, shifted</th><th>Heavy-rain bias, early</th><th>Heavy-rain bias, shifted</th><th>RMSE early (mm/day)</th><th>RMSE shifted</th><th>Fine-scale detail kept</th></tr></thead>`;
  const rows = MODELS.map((m) => {
    const a = s.in_dist[m.key], b = s.shifted[m.key];
    const worse = Math.abs(b.p99_bias_pct) > Math.abs(a.p99_bias_pct) + 5;
    return `<tr><td><span class="swatch" style="background:${m.color}"></span>${m.label}</td>` +
      `<td>${fmt(a.mean_bias_pct)}%</td><td>${fmt(b.mean_bias_pct)}%</td>` +
      `<td>${fmt(a.p99_bias_pct)}%</td><td class="${worse ? "worse" : ""}">${fmt(b.p99_bias_pct)}%${worse ? " ▲" : ""}</td>` +
      `<td>${fmt(a.rmse, 2, false)}</td><td>${fmt(b.rmse, 2, false)}</td><td>${fmt(100 * b.small_scale_ratio, 0, false)}%</td></tr>`;
  }).join("");
  $("score-table").innerHTML = head + `<tbody>${rows}</tbody>`;
  const ia = s.in_dist.interval, ib = s.shifted.interval;
  $("interval-note").textContent =
    `Heavy-rain bias uses the 99th percentile of all pixel-day rainfall. ▲ marks a bias at least 5 points larger on the shifted years. ` +
    `The ML 10–90% interval contains ${fmt(100 * ia.coverage_wet, 0, false)}% of wet-day observations early and ${fmt(100 * ib.coverage_wet, 0, false)}% shifted (target 80%).`;
}

function renderChange(res) {
  const groups = [["Mean rainfall", "mean"], ["Heavy rain (99th percentile)", "p99"]];
  const bars = [{ key: "obs", label: "Observed", color: "var(--obs)" }, ...MODELS];
  legend($("change-legend"), bars.map((b) => ({ color: b.color, label: b.label, kind: "bar" })));
  const W = 560, bh = 16, gap = 4, gh = bars.length * (bh + gap) + 34, left = 12, H = groups.length * gh + 30;
  const svg = svgFor($("change-svg"), W, H);
  const vals = [];
  groups.forEach(([, k]) => bars.forEach((b) => vals.push(b.key === "obs" ? res.change.gbm[`obs_${k}_change_pct`] : res.change[b.key][`pred_${k}_change_pct`])));
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals) * 1.15 + 1;
  const x = lin(lo, hi, left + 4, W - 60);
  niceTicks(lo, hi, 5).forEach((t) => {
    el("line", { x1: x(t), x2: x(t), y1: 6, y2: H - 22, class: t === 0 ? "axis" : "grid" }, svg);
    txt(svg, x(t), H - 6, `${t > 0 ? "+" : ""}${t}%`, { "text-anchor": "middle" });
  });
  groups.forEach(([title, k], gi) => {
    const y0 = gi * gh + 18;
    txt(svg, left, y0, title, { class: "ink", "font-weight": 600 });
    bars.forEach((b, i) => {
      const v = b.key === "obs" ? res.change.gbm[`obs_${k}_change_pct`] : res.change[b.key][`pred_${k}_change_pct`];
      const y = y0 + 10 + i * (bh + gap);
      const x0 = x(Math.min(0, v)), w = Math.max(2, Math.abs(x(v) - x(0)));
      el("rect", { x: x0, y, width: w, height: bh, rx: 3, fill: b.color }, svg);
      txt(svg, x(Math.max(0, v)) + 6, y + 12, `${fmt(v, 0)}%`, { class: "ink", "font-size": 12 });
      const hit = el("rect", { x: left, y, width: W - left, height: bh, class: "hit" }, svg);
      const cap = b.key === "obs" ? "" : `<br>${fmt(res.change[b.key][`${k}_captured_pct`], 0, false)}% of observed change`;
      bindTip(hit, `<b>${b.label}</b><br>${title}: ${fmt(v)}%${cap}`);
    });
  });
}

function renderQQ(res) {
  const keep = res.qq_levels.map((q, i) => (q >= 0.9 ? i : -1)).filter((i) => i >= 0);
  const lv = keep.map((i) => res.qq_levels[i]);
  const s0 = res.scores.shifted, s = {};
  MODELS.forEach((m) => { s[m.key] = { qq_pred: keep.map((i) => s0[m.key].qq_pred[i]), qq_obs: keep.map((i) => s0[m.key].qq_obs[i]) }; });
  legend($("qq-legend"), MODELS.map((m) => ({ color: m.color, label: m.label, kind: "line" })));
  const W = 560, H = 280, l = 52, r = 16, t = 12, b = 40;
  const svg = svgFor($("qq-svg"), W, H);
  const series = MODELS.map((m) => ({ m, v: s[m.key].qq_pred.map((p, i) => 100 * (p - s[m.key].qq_obs[i]) / (s[m.key].qq_obs[i] || 1)) }));
  const all = series.flatMap((d) => d.v);
  const lo = Math.min(-10, ...all) - 5, hi = Math.max(10, ...all) + 5;
  const x = lin(0, lv.length - 1, l, W - r), y = lin(lo, hi, H - b, t);
  niceTicks(lo, hi, 5).forEach((v) => {
    el("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: v === 0 ? "zero" : "grid" }, svg);
    txt(svg, l - 8, y(v) + 4, `${v > 0 ? "+" : ""}${v}%`, { "text-anchor": "end" });
  });
  lv.forEach((q, i) => txt(svg, x(i), H - b + 18, `${+(q * 100).toFixed(1)}`, { "text-anchor": "middle" }));
  txt(svg, (l + W - r) / 2, H - 4, "percentile of daily rainfall", { "text-anchor": "middle" });
  series.forEach(({ m, v }) => {
    el("path", { d: v.map((val, i) => `${i ? "L" : "M"}${x(i)},${y(val)}`).join(""), fill: "none", stroke: m.color, "stroke-width": 2 }, svg);
    v.forEach((val, i) => {
      el("circle", { cx: x(i), cy: y(val), r: 3.5, fill: m.color, stroke: "#fff", "stroke-width": 1.5 }, svg);
    });
  });
  lv.forEach((q, i) => {
    const hit = el("rect", { x: x(i) - (W - l - r) / lv.length / 2, y: t, width: (W - l - r) / lv.length, height: H - t - b, class: "hit" }, svg);
    bindTip(hit, `<b>${+(q * 100).toFixed(1)}th percentile</b><br>Observed ${fmt(s.qm.qq_obs[i], 1, false)} mm/day<br>` +
      series.map(({ m, v }) => `${m.label}: ${fmt(v[i], 0)}%`).join("<br>"));
  });
}

function renderSpectra(res) {
  const sp = res.spectra.shifted, cfg = res.meta.config, W0 = res.meta.grid.fine[1];
  const kmPerPx = (cfg.lon_max - cfg.lon_min) / W0 * 111;
  legend($("spec-legend"), [{ kind: "dash", label: "Observed" }, ...MODELS.map((m) => ({ color: m.color, label: m.label, kind: "line" }))]);
  const W = 1100, H = 300, l = 58, r = 16, t = 12, b = 40;
  const svg = svgFor($("spec-svg"), W, H);
  const wl = sp.k.map((k) => kmPerPx / k);
  const all = [sp.obs, ...MODELS.map((m) => sp[m.key])].flat().filter((v) => v > 0);
  const ymin = Math.max(Math.min(...all), Math.max(...all) * 1e-6), ymax = Math.max(...all);
  const x = log(Math.max(...wl), Math.min(...wl), l, W - r), y = log(ymin, ymax, H - b, t);
  const coarseKm = kmPerPx * res.meta.grid.factor;
  [2000, 1000, 500, 200, 100, 50, 20].filter((v) => v <= Math.max(...wl) && v >= Math.min(...wl)).forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: t, y2: H - b, class: "grid" }, svg);
    txt(svg, x(v), H - b + 18, `${v}`, { "text-anchor": "middle" });
  });
  txt(svg, (l + W - r) / 2, H - 4, "wavelength (km)", { "text-anchor": "middle" });
  txt(svg, 6, t + 10, "power", {});
  if (coarseKm <= Math.max(...wl)) {
    el("line", { x1: x(coarseKm), x2: x(coarseKm), y1: t, y2: H - b, class: "zero" }, svg);
    txt(svg, x(coarseKm) + 4, t + 10, "coarse grid", { "font-size": 11 });
  }
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(wl[i])},${y(Math.max(v, ymin))}`).join("");
  MODELS.forEach((m) => el("path", { d: path(sp[m.key]), fill: "none", stroke: m.color, "stroke-width": 2 }, svg));
  el("path", { d: path(sp.obs), fill: "none", stroke: "var(--obs)", "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
  const n = wl.length;
  wl.forEach((w, i) => {
    const x0 = i === 0 ? l : (x(wl[i - 1]) + x(w)) / 2, x1 = i === n - 1 ? W - r : (x(w) + x(wl[i + 1])) / 2;
    const hit = el("rect", { x: Math.min(x0, x1), y: t, width: Math.abs(x1 - x0), height: H - t - b, class: "hit" }, svg);
    bindTip(hit, `<b>${fmt(w, 0, false)} km</b><br>` + MODELS.map((m) => `${m.label}: ${fmt(100 * sp[m.key][i] / sp.obs[i], 0, false)}% of observed`).join("<br>"));
  });
}

async function loadReport(base) {
  const body = $("report-body");
  try {
    const r = await fetch(base + "report.html", { cache: "no-store" });
    body.innerHTML = await r.text();
    body.querySelectorAll("img").forEach((img) => { img.src = base + img.getAttribute("src"); });
  } catch (e) { body.textContent = "The report could not be loaded. Run the pipeline to create it."; }
}

/* ---------- running (server mode only) ---------- */
function setStatus(msg, frac, error) {
  const s = $("run-status");
  s.textContent = msg; s.classList.toggle("error", !!error);
  $("run-bar").style.width = `${Math.round(100 * (frac || 0))}%`;
}
async function poll(onDone) {
  const st = await getJSON("api/status");
  if (st.state === "running") {
    setStatus(`${st.message} (${Math.round(100 * st.progress)}%)`, st.progress);
    $("run-btn").disabled = true;
    setTimeout(() => poll(onDone), 1500);
  } else {
    $("run-btn").disabled = false;
    if (st.state === "error") setStatus(`${st.message}: ${st.error}`, 0, true);
    else if (st.state === "done") { setStatus(st.message, 1); onDone && onDone(st); }
  }
}
function wireForm() {
  const range = $("f-warming"), out = $("f-warming-out");
  range.addEventListener("input", () => (out.textContent = Number(range.value).toFixed(2)));
  $("run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { preset: $("f-preset").value, warming_rate: Number(range.value), seed: Number($("f-seed").value), run_drywet: $("f-drywet").checked };
    const r = await fetch("api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { const d = await r.json().catch(() => ({})); setStatus(d.detail ? String(d.detail) : `Could not start the run (${r.status})`, 0, true); return; }
    setStatus("Run started", 0.01);
    poll(async () => { S.current = "custom"; await loadAll(); document.getElementById("explore").scrollIntoView(); });
  });
}

/* ---------- start ---------- */
(async function init() {
  try { const h = await fetch("api/health"); S.staticMode = !h.ok; } catch { S.staticMode = true; }
  document.body.classList.toggle("static", S.staticMode);
  if (!S.staticMode) wireForm();
  try { await loadAll(); }
  catch (e) {
    $("hero-finding").textContent = "No results yet. They are being prepared now; this page updates when they are ready.";
    $("report-body").textContent = "The report appears here once the first run finishes.";
    if (!S.staticMode) poll(() => loadAll());
  }
  if (!S.staticMode) {
    const st = await getJSON("api/status").catch(() => null);
    if (st && st.state === "running") poll(() => loadAll());
  }
})();
