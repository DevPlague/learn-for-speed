// learn-for-speed dashboard. Reads window.DATA written by dashboard/build.py, no server needed.
"use strict";

const D = window.DATA;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ICONS = {
  flag: '<path d="M4 22V4"/><path d="M4 4h12l-2 4 2 4H4"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  home: '<path d="M3 11 12 4l9 7"/><path d="M5 10v10h14V10"/>',
  chart: '<path d="M3 3v18h18"/><path d="m7 15 4-5 4 3 5-7"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  play: '<path d="M7 4v16l13-8z"/>',
  pause: '<path d="M7 4h4v16H7zM14 4h4v16h-4z"/>',
  gauge: '<path d="M12 14l4-4"/><path d="M3.3 17a9 9 0 1 1 17.4 0"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  monitor: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
  chevron: '<path d="m15 18-6-6 6-6"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h10"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  car: '<path d="M5 16h14l-1.5-6h-11z"/><circle cx="8" cy="17" r="2"/><circle cx="16" cy="17" r="2"/>',
  bot: '<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01"/>',
  shuffle: '<path d="M3 6h4l10 12h4M3 18h4l3-3.6M14 9.6 17 6h4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  steps: '<path d="M4 20h4v-4h4v-4h4V8h4"/>',
  trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
  route: '<circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M8 19h8a3 3 0 0 0 0-6H8a3 3 0 0 1 0-6h8"/>',
  alert: '<path d="M12 3 2 21h20z"/><path d="M12 10v4M12 18h.01"/>',
};

const VIEWS = [
  ["overview", "Overview", "home"],
  ["training", "Training", "chart"],
  ["validation", "Validation", "check"],
  ["statistics", "Statistics", "target"],
  ["test", "Test", "lock"],
  ["simulation", "Simulation", "play"],
  ["physics", "Physics", "gauge"],
];
const RUN_PALETTE = ["#2f6bff", "#db2777", "#0891b2", "#7c3aed", "#ea580c", "#0d9488"];
const FIXED_COLORS = { pure_pursuit: "#16a34a", random: "#d97706" };
const OUTCOME = {
  lap: ["Lap", "#16a34a"],
  off_road: ["Off road", "#dc2626"],
  stuck: ["Stuck", "#d97706"],
  timeout: ["Timeout", "#7b8088"],
};
const METRIC_LABELS = {
  lap_success_rate: "laps completed",
  off_track_rate: "went off road",
  stuck_rate: "got stuck",
  mean_lap_time: "lap time (s)",
  mean_speed: "avg speed (km/h)",
  mean_progress: "lap driven",
};

const state = { view: "overview", group: "all", hidden: new Set(), charts: {}, search: "", level: 0 };

// ---------- small helpers ----------
const icon = (name) =>
  `<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x) => (x == null ? "-" : `${Math.round(100 * x)} %`);
const num = (x, d = 1) => (x == null ? "-" : Number(x).toFixed(d));
const last = (arr) => (arr && arr.length ? arr[arr.length - 1] : null);
const groupOf = (name) => name.replace(/_s\d+$/, "").split("_")[0];
const configOf = (name) => name.replace(/_s\d+$/, "");
const BASELINE_NAMES = { pure_pursuit: "Pure pursuit (baseline)", random: "Random (baseline)" };
function policyName(key) {
  if (BASELINE_NAMES[key]) return BASELINE_NAMES[key];
  if (key.startsWith("pooled:")) return `PPO ${key.slice(7)} (all seeds)`;
  const run = key.replace(/^ppo:/, "");
  const match = run.match(/^(.*)_s(\d+)$/);
  return match ? `PPO ${match[1]} (seed ${match[2]})` : `PPO ${run}`;
}

function fillIcons(root = document) {
  $$("[data-icon]", root).forEach((el) => {
    if (!el.dataset.done) {
      el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon));
      el.dataset.done = "1";
    }
  });
}

function policyColor(key) {
  if (FIXED_COLORS[key]) return FIXED_COLORS[key];
  const ppo = Object.keys(D.validation).filter((k) => k.startsWith("ppo:"));
  if (key.startsWith("pooled:")) key = ppo.find((k) => configOf(k.slice(4)) === key.slice(7)) || key;
  return RUN_PALETTE[Math.max(ppo.indexOf(key), 0) % RUN_PALETTE.length];
}

function runColor(name) {
  const configs = [...new Set(D.runs.map((r) => configOf(r.name)))];
  return RUN_PALETTE[configs.indexOf(configOf(name)) % RUN_PALETTE.length];
}

function alpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function visibleRuns() {
  return D.runs.filter(
    (r) =>
      (state.group === "all" || groupOf(r.name) === state.group) &&
      r.name.toLowerCase().includes(state.search.toLowerCase())
  );
}

// validation and statistics of the difficulty picked in the level tabs
function level() {
  if (!D.levels) return { difficulty: D.meta.difficulty, validation: D.validation, stats: null };
  return D.levels[Math.min(state.level, D.levels.length - 1)];
}

function levelTabs(el) {
  const levels = D.levels || [];
  el.innerHTML = levels.length < 2 ? "" : levels
    .map((l, i) => `<button data-level="${i}" class="${i === state.level ? "active" : ""}">difficulty ${num(l.difficulty)}</button>`)
    .join("");
  $$("button", el).forEach((b) =>
    b.addEventListener("click", () => {
      state.level = Number(b.dataset.level);
      RENDER[state.view]();
    })
  );
}

function mainPolicy() {
  return D.meta.selected.find((k) => D.validation[k]) || Object.keys(D.validation)[0];
}

// ---------- charts ----------
function chartDefaults() {
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 11;
  Chart.defaults.locale = "en-US";
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.filter = (item) => !(item.text || "").startsWith("_");
}

function makeChart(canvas, config) {
  const id = canvas.id;
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart(canvas, config);
  return state.charts[id];
}

// info is {what, good, read}, shown in a popover next to the title
function infoButton(info) {
  if (!info) return "";
  const rows = [
    ["What it is", info.what],
    ["Good result", info.good],
    ["How to read it", info.read],
  ];
  return `<span class="info" tabindex="0" aria-label="About this chart">i<span class="tip">${rows
    .map(([label, text]) => `<b>${label}</b><span>${text}</span>`)
    .join("")}</span></span>`;
}

function chartCard(parent, id, title, info) {
  parent.insertAdjacentHTML(
    "beforeend",
    `<div class="card chart-card"><div class="chart-title">${title}${infoButton(info)}</div><div class="chart-box"><canvas id="${id}"></canvas></div></div>`
  );
  return $(`#${id}`);
}

function lineOptions(xTitle, extra = {}) {
  return {
    parsing: false,
    scales: {
      x: { type: "linear", title: { display: true, text: xTitle }, grid: { display: false } },
      y: { beginAtZero: false, ...extra.y },
    },
    elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.25 } },
    plugins: { legend: { display: extra.legend !== false, position: "bottom" } },
  };
}

// mean line plus a band of one std for every config with several seeds
function groupedSeries(runs, source, key) {
  const groups = {};
  runs.forEach((r) => {
    const table = r[source];
    if (table && table[key] && table.timesteps) (groups[configOf(r.name)] ||= []).push(table);
  });
  const datasets = [];
  Object.entries(groups).forEach(([config, tables]) => {
    const n = Math.min(...tables.map((t) => t.timesteps.length));
    const color = runColor(config);
    const mean = [], lo = [], hi = [];
    for (let i = 0; i < n; i++) {
      const ys = tables.map((t) => t[key][i]).filter((v) => v != null);
      if (!ys.length) continue;
      const m = ys.reduce((a, b) => a + b, 0) / ys.length;
      const sd = Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / ys.length);
      const x = tables[0].timesteps[i];
      mean.push({ x, y: m });
      lo.push({ x, y: m - sd });
      hi.push({ x, y: m + sd });
    }
    const label = tables.length > 1 ? `${config} (${tables.length} seeds)` : config;
    if (tables.length > 1) {
      datasets.push({ label: `_lo${config}`, data: lo, borderWidth: 0, pointRadius: 0, fill: false });
      datasets.push({ label: `_hi${config}`, data: hi, borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: alpha(color, 0.15) });
    }
    datasets.push({ label, data: mean, borderColor: color, backgroundColor: color });
  });
  return datasets;
}

// ---------- track drawing ----------
function bounds(track) {
  const pts = track.left_wall.concat(track.right_wall);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

function fitView(track, w, h, pad = 12) {
  const b = bounds(track);
  const s = Math.min((w - 2 * pad) / (b.x1 - b.x0), (h - 2 * pad) / (b.y1 - b.y0));
  const ox = (w - s * (b.x1 - b.x0)) / 2, oy = (h - s * (b.y1 - b.y0)) / 2;
  return { s, map: (x, y) => [ox + (x - b.x0) * s, h - oy - (y - b.y0) * s] };
}

function followView(cx, cy, w, h, s) {
  return { s, map: (x, y) => [w / 2 + (x - cx) * s, h / 2 - (y - cy) * s] };
}

function pathOf(ctx, pts, view, close = true) {
  pts.forEach((p, i) => {
    const [x, y] = view.map(p[0], p[1]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  if (close) ctx.closePath();
}

function drawTrack(ctx, track, view, w, h) {
  ctx.fillStyle = css("--grass");
  ctx.fillRect(0, 0, w, h);
  ctx.beginPath();
  pathOf(ctx, track.left, view);
  pathOf(ctx, track.right, view);
  ctx.fillStyle = css("--road");
  ctx.fill("evenodd");
  const lw = Math.max(1, view.s * 0.25);
  ctx.lineWidth = lw;
  ctx.strokeStyle = css("--edge");
  [track.left, track.right].forEach((edge) => {
    ctx.beginPath();
    pathOf(ctx, edge, view);
    ctx.stroke();
  });
  ctx.strokeStyle = css("--wall");
  ctx.lineWidth = lw * 1.6;
  [track.left_wall, track.right_wall].forEach((wall) => {
    ctx.beginPath();
    pathOf(ctx, wall, view);
    ctx.stroke();
  });
  ctx.setLineDash([view.s * 2, view.s * 2]);
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = Math.max(1, view.s * 0.15);
  ctx.beginPath();
  pathOf(ctx, track.centerline, view);
  ctx.stroke();
  ctx.setLineDash([]);
  const a = view.map(...track.right[0]), b = view.map(...track.left[0]);
  ctx.lineWidth = Math.max(3, view.s * 1.2);
  ctx.strokeStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
  ctx.setLineDash([ctx.lineWidth, ctx.lineWidth]);
  ctx.strokeStyle = "#111214";
  ctx.stroke();
  ctx.setLineDash([]);
}

function speedColor(v) {
  const t = Math.min(Math.abs(v) / D.meta.max_speed, 1);
  return `hsl(${220 - 220 * t}, 85%, 55%)`;
}

function drawTrail(ctx, f, upto, view) {
  ctx.lineWidth = Math.max(1.5, view.s * 0.5);
  ctx.lineCap = "round";
  const step = Math.max(1, Math.floor(upto / 600));
  for (let i = step; i <= upto; i += step) {
    ctx.strokeStyle = speedColor(f.v[i]);
    ctx.beginPath();
    ctx.moveTo(...view.map(f.x[i - step], f.y[i - step]));
    ctx.lineTo(...view.map(f.x[i], f.y[i]));
    ctx.stroke();
  }
}

function drawCar(ctx, f, i, view, color, lidar) {
  const x = f.x[i], y = f.y[i], yaw = f.yaw[i];
  if (lidar) {
    ctx.strokeStyle = "rgba(255, 210, 0, 0.45)";
    ctx.fillStyle = "rgba(255, 210, 0, 0.9)";
    ctx.lineWidth = 1;
    D.meta.lidar_angles.forEach((ang, k) => {
      const d = f.hits[i][k];
      const end = view.map(x + d * Math.cos(yaw + ang), y + d * Math.sin(yaw + ang));
      ctx.beginPath();
      ctx.moveTo(...view.map(x, y));
      ctx.lineTo(...end);
      ctx.stroke();
      if (d < D.meta.lidar_range) ctx.fillRect(end[0] - 1.5, end[1] - 1.5, 3, 3);
    });
  }
  const scale = Math.max(view.s, 2.2);
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const body = [[-1, -0.9], [3, -0.9], [3, 0.9], [-1, 0.9]].map(([bx, by]) => {
    const [px, py] = view.map(x, y);
    return [px + (bx * c - by * s) * scale, py - (bx * s + by * c) * scale];
  });
  ctx.beginPath();
  body.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = f.on_grass[i] ? "#dc2626" : "#ffffff";
  ctx.stroke();
}

function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320, h = canvas.clientHeight || 240;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawStatic(canvas, track, ep, color) {
  const { ctx, w, h } = sizeCanvas(canvas);
  const view = fitView(track, w, h);
  drawTrack(ctx, track, view, w, h);
  const n = ep.frames.x.length - 1;
  drawTrail(ctx, ep.frames, n, view);
  drawCar(ctx, ep.frames, n, view, color, false);
}

function statusHtml(ep) {
  const [text, color] = OUTCOME[ep.outcome];
  return `<span class="status"><span class="dot" style="background:${color}"></span>${text}</span>`;
}

function episodeNote(ep) {
  return ep.outcome === "lap" ? `lap in ${num(ep.lap_time)} s` : `${pct(ep.progress)} of the lap`;
}

// ---------- overview ----------
function renderHero() {
  const seeds = Object.keys(D.tracks);
  const svg = $("#hero-track");
  if (!seeds.length) return;
  const track = D.tracks[seeds[0]];
  const view = fitView(track, 300, 190, 16);
  const pts = (arr) => arr.map((p) => view.map(p[0], p[1]).map((v) => v.toFixed(1)).join(",")).join(" ");
  const shift = "translate(90,20)";
  svg.innerHTML = `
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#bcd7f5"/><stop offset="1" stop-color="#eef5e6"/>
      </linearGradient>
    </defs>
    <rect width="400" height="220" fill="url(#sky)"/>
    <path d="M0 150 Q80 90 160 140 T320 120 T400 130 V220 H0z" fill="#9ccf86"/>
    <path d="M0 175 Q120 130 230 170 T400 160 V220 H0z" fill="#6fae5c"/>
    <g transform="${shift}">
      <polygon points="${pts(track.left_wall)}" fill="none" stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>
      <polyline points="${pts(track.centerline)}" fill="none" stroke="#3f444b" stroke-width="7" stroke-linejoin="round"/>
      <polyline points="${pts(track.centerline)}" fill="none" stroke="#ffffff" stroke-width="1" stroke-dasharray="4 4"/>
      <circle cx="${view.map(...track.centerline[0])[0]}" cy="${view.map(...track.centerline[0])[1]}" r="5" fill="#dc2626" stroke="#fff" stroke-width="2"/>
    </g>`;
}

function renderTabs() {
  const groups = ["all", ...new Set(D.runs.map((r) => groupOf(r.name)))];
  $("#group-tabs").innerHTML = groups
    .map((g) => `<button data-group="${g}" class="${g === state.group ? "active" : ""}">${g === "all" ? "All runs" : g}</button>`)
    .join("");
  $$("#group-tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      state.group = b.dataset.group;
      renderAll();
    })
  );
  $("#difficulty-pill").innerHTML = `${icon("route")} difficulty ${num(D.meta.difficulty)} &middot; ${D.meta.eval_tracks} val tracks`;
}

function kpiCard(parent, { id, title, sub, value, note, iconName }) {
  parent.insertAdjacentHTML(
    "beforeend",
    `<div class="card">
      <div class="kpi-head"><div><div class="kpi-title">${title}</div><div class="kpi-sub">${sub}</div></div>
      <div class="icon-box">${icon(iconName)}</div></div>
      <div class="kpi-value">${value}</div><div class="kpi-note">${note}</div>
      <div class="kpi-chart"><canvas id="${id}"></canvas></div>
    </div>`
  );
  return $(`#${id}`);
}

function renderKpis() {
  const el = $("#kpis");
  el.innerHTML = "";
  const key = mainPolicy();
  const v = D.validation[key] || {};
  const pp = D.validation.pure_pursuit || {};
  const run = D.runs.find((r) => `ppo:${r.name}` === key);
  const curve = run && run.val_curve;
  const color = policyColor(key);
  const eps = D.replays[key] || [];

  const c1 = kpiCard(el, { id: "k-lap", title: "Lap success", sub: `${policyName(key)} on validation`, value: pct(v.lap_success_rate), note: `pure pursuit ${pct(pp.lap_success_rate)}`, iconName: "trophy" });
  const curveData = curve ? curve.timesteps.map((x, i) => ({ x, y: curve.lap_success_rate[i] })) : [];
  makeChart(c1, {
    type: "line",
    data: { datasets: [{ data: curveData, borderColor: color, fill: true, backgroundColor: alpha(color, 0.08) }] },
    options: { ...lineOptions("", { legend: false }), scales: { x: { type: "linear", display: false }, y: { display: false, min: 0, max: 1 } } },
  });

  const c2 = kpiCard(el, { id: "k-time", title: "Lap time", sub: "mean of completed laps", value: `${num(v.mean_lap_time)} s`, note: `pure pursuit ${num(pp.mean_lap_time)} s`, iconName: "clock" });
  makeChart(c2, {
    type: "bar",
    data: { labels: eps.map((e) => e.seed), datasets: [{ data: eps.map((e) => e.progress), backgroundColor: eps.map((e) => alpha(OUTCOME[e.outcome][1], 0.7)), borderRadius: 3 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false, min: 0, max: 1 } } },
  });

  const c3 = kpiCard(el, { id: "k-speed", title: "Mean speed", sub: "over every step", value: `${num((v.mean_speed || 0) * 3.6)} km/h`, note: `pure pursuit ${num((pp.mean_speed || 0) * 3.6)} km/h`, iconName: "gauge" });
  const ep = eps[0];
  makeChart(c3, {
    type: "line",
    data: { datasets: ep ? [{ data: ep.frames.v.map((y, i) => ({ x: i * ep.dt, y: y * 3.6 })), borderColor: color, borderWidth: 1.5 }] : [] },
    options: { ...lineOptions("", { legend: false }), scales: { x: { type: "linear", display: false }, y: { display: false } } },
  });

  const c4 = kpiCard(el, { id: "k-out", title: "Failures", sub: "off track and stuck", value: pct((v.off_track_rate || 0) + (v.stuck_rate || 0)), note: `off ${pct(v.off_track_rate)} &middot; stuck ${pct(v.stuck_rate)}`, iconName: "alert" });
  makeChart(c4, {
    type: "bar",
    data: {
      labels: Object.keys(D.validation).map(policyName),
      datasets: [{ data: Object.values(D.validation).map((m) => m.lap_success_rate), backgroundColor: Object.keys(D.validation).map((k) => alpha(policyColor(k), 0.75)), borderRadius: 3 }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { display: false, min: 0, max: 1 }, y: { grid: { display: false } } } },
  });
}

function renderBoard() {
  const el = $("#board");
  el.innerHTML = "";
  const seeds = Object.keys(D.tracks);
  if (!seeds.length) return;
  Object.keys(D.replays).slice(0, 3).forEach((key, i) => {
    const ep = D.replays[key][0];
    el.insertAdjacentHTML(
      "beforeend",
      `<div class="card tile"><canvas id="board-${i}"></canvas><div class="tile-body">
        <div class="tile-title">${policyName(key)}</div>
        <div class="kpi-note">seed ${ep.seed}, ${episodeNote(ep)}</div>
        <div class="tile-foot">${statusHtml(ep)}<span>${(ep.frames.x.length * ep.dt).toFixed(1)} s</span></div>
      </div></div>`
    );
    drawStatic($(`#board-${i}`), D.tracks[ep.seed], ep, policyColor(key));
  });
}

function grade(rate) {
  return rate >= 0.9 ? "A+" : rate >= 0.75 ? "A" : rate >= 0.5 ? "B" : rate >= 0.25 ? "C" : "D";
}

function renderPolicies() {
  $("#policies").innerHTML = Object.entries(D.validation)
    .map(([key, m]) => {
      const iconName = key.startsWith("ppo:") ? "bot" : key === "random" ? "shuffle" : "car";
      return `<div class="list-item" data-policy="${key}">
        <div class="icon-box" style="color:${policyColor(key)}">${icon(iconName)}</div>
        <div class="grow"><div>${policyName(key)}</div><div class="sub">lap ${pct(m.lap_success_rate)} &middot; ${num(m.mean_speed)} m/s</div></div>
        <span class="grade">${grade(m.lap_success_rate)}</span></div>`;
    })
    .join("");
  $$("#policies .list-item").forEach((item) =>
    item.addEventListener("click", () => {
      simState.mode = "policy";
      simState.pick = item.dataset.policy;
      go("simulation");
    })
  );

  const runs = visibleRuns();
  const steps = runs.reduce((a, r) => a + (last(r.progress && r.progress.timesteps) || 0), 0);
  const best = Math.max(0, ...runs.map((r) => Math.max(0, ...((r.val_curve && r.val_curve.lap_success_rate) || [0]))));
  const v = D.validation[mainPolicy()] || {};
  const stats = [
    ["steps", `${(steps / 1e6).toFixed(2)} M`, "steps trained"],
    ["layers", runs.length, "runs"],
    ["trophy", pct(best), "best val lap rate"],
    ["route", pct(v.mean_progress), "mean progress"],
  ];
  $("#stats").innerHTML = stats
    .map(([ic, n, label]) => `<div class="card"><div class="icon-box">${icon(ic)}</div><div><div class="stat-num">${n}</div><div class="stat-label">${label}</div></div></div>`)
    .join("");
}

// ---------- training ----------
const TRAIN_METRICS = [
  {
    source: "val_curve",
    key: "lap_success_rate",
    title: "Laps completed on unseen tracks",
    info: {
      what: "Share of the validation tracks (seeds from 1000, never trained on) where the car finished a full lap. Checked every eval.freq steps with deterministic actions.",
      good: "0.7 or more is the MVP target. 1.0 means every track was lapped. Pure pursuit laps all of them at difficulty 0.",
      read: "It should climb and then stay high. Plateaus are normal, PPO often jumps after one. A drop late in training means the policy got worse, best_model.zip keeps the best point.",
    },
  },
  {
    source: "val_curve",
    key: "mean_progress",
    title: "Share of the lap driven on unseen tracks",
    info: {
      what: "How far around the lap the car got before the attempt ended, averaged over the validation tracks. 1.0 is a full lap.",
      good: "Above 0.9 means it fails only near the end, if at all. Near 0 means it does not leave the start line.",
      read: "Moves before the lap rate does, so it shows learning even while no lap is finished yet. High progress with a low lap rate means it crashes late on most tracks.",
    },
  },
  {
    source: "progress",
    key: "ep_rew_mean",
    title: "Score per attempt (training tracks)",
    info: {
      what: "Total reward of one attempt, averaged over the last 100 attempts on random training tracks. Progress and the lap bonus add, time, steering jitter, grass, idling, crashing and getting stuck subtract.",
      good: "With the current weights a clean lap scores about plus 10. Crashing early is about minus 5 and standing still until stuck is about minus 18.",
      read: "Rising means the agent is learning what the reward asks for. Only compare runs that use the same reward weights, since the scale changes with them.",
    },
  },
  {
    source: "progress",
    key: "ep_len_mean",
    title: "Attempt length in steps (training tracks)",
    info: {
      what: "Steps per attempt on training tracks. One step is 0.05 s, so 20 steps are 1 s.",
      good: "A lap at difficulty 0 takes about 480 steps (24 s). Exactly 200 for a long time is a warning, it is the 10 s stuck limit, so the car stands still.",
      read: "Read it with the score. Length falling while the score rises means faster laps. Length falling while the score falls means earlier crashes.",
    },
  },
  {
    source: "progress",
    key: "entropy_loss",
    title: "Exploration level (entropy loss)",
    info: {
      what: "Minus the entropy of the action distribution. It measures how random the actions still are during training.",
      good: "A slow steady rise. The agent explores first and becomes more decided as it learns.",
      read: "Higher means less random. A fast rise very early means it stopped exploring too soon, which is how the failed MVP got stuck standing still.",
    },
  },
  {
    source: "progress",
    key: "approx_kl",
    title: "Policy change per update (approx KL)",
    info: {
      what: "How much the policy changed in one PPO update, measured as the KL divergence between the old and new action distributions.",
      good: "Between about 0.005 and 0.02.",
      read: "Regular spikes above 0.05 mean the updates are too big, lower the learning rate. Values near 0 for a long time mean the policy has almost stopped changing.",
    },
  },
  {
    source: "progress",
    key: "value_loss",
    title: "Critic error (value loss)",
    info: {
      what: "Squared error of the critic, the network that predicts how much reward is still to come. Rewards are normalized by VecNormalize, so the scale is relative.",
      good: "Falls and stays low.",
      read: "Spikes are normal when the behaviour changes, for example at the first finished laps. A loss that keeps growing means the critic cannot keep up.",
    },
  },
  {
    source: "progress",
    key: "explained_variance",
    title: "Critic accuracy (explained variance)",
    info: {
      what: "How much of the real returns the critic explains. 1 is perfect, 0 is no better than always guessing the average, below 0 is worse than that.",
      good: "Above 0.5 while the agent is still learning to survive. Once almost every attempt is a full lap, 0.2 to 0.3 is normal here.",
      read: "The critic only sees lidar, speed, steering and slip, never where the finish line is, so it cannot predict when the lap bonus arrives. When nearly all attempts end in a lap, that bonus is most of what is left to predict and the value drops. Worry only if it stays near 0 or negative while the laps completed chart is still low.",
    },
  },
];

function renderTraining() {
  const runs = visibleRuns();
  $("#run-chips").innerHTML = runs
    .map((r) => `<span class="chip ${state.hidden.has(r.name) ? "off" : ""}" data-run="${r.name}"><span class="dot" style="background:${runColor(r.name)}"></span>${r.name}</span>`)
    .join("");
  $$("#run-chips .chip").forEach((c) =>
    c.addEventListener("click", () => {
      const name = c.dataset.run;
      state.hidden.has(name) ? state.hidden.delete(name) : state.hidden.add(name);
      renderTraining();
      renderRunList();
    })
  );
  const shown = runs.filter((r) => !state.hidden.has(r.name));
  const el = $("#train-charts");
  el.innerHTML = "";
  TRAIN_METRICS.forEach(({ source, key, title, info }, i) => {
    const datasets = groupedSeries(shown, source, key);
    const canvas = chartCard(el, `train-${i}`, title, info);
    makeChart(canvas, { type: "line", data: { datasets }, options: lineOptions("training steps") });
  });
}

// ---------- validation and test ----------
function metricsTable(rows) {
  const keys = Object.keys(METRIC_LABELS);
  const fmt = (k, v) => {
    if (v == null) return "-";
    if (k.endsWith("rate") || k === "mean_progress") return pct(v);
    return k === "mean_speed" ? num(v * 3.6, 0) : num(v);
  };
  return `<table><thead><tr><th>policy</th>${keys.map((k) => `<th>${METRIC_LABELS[k]}</th>`).join("")}</tr></thead><tbody>${Object.entries(rows)
    .map(
      ([name, m]) =>
        `<tr><td><span class="dot" style="display:inline-block;background:${policyColor(name)}"></span> ${policyName(name)}</td>${keys
          .map((k) => `<td>${fmt(k, m[k])}${m[k + "_std"] != null ? ` &plusmn; ${fmt(k, m[k + "_std"])}` : ""}</td>`)
          .join("")}</tr>`
    )
    .join("")}</tbody></table>`;
}

function renderValidation() {
  levelTabs($("#val-level"));
  const { difficulty, validation: V } = level();
  $("#val-note").textContent = `First ${D.meta.eval_tracks} validation seeds from 1000, difficulty ${num(difficulty)}, deterministic actions, one episode per track.`;
  $("#val-table").innerHTML = metricsTable(V);
  const keys = Object.keys(V);
  const el = $("#val-charts");
  el.innerHTML = "";
  const pp = V.pure_pursuit || {};
  const bar = (id, title, values, max, info) =>
    makeChart(chartCard(el, id, title, info), {
      type: "bar",
      data: { labels: keys.map(policyName), datasets: [{ data: values, backgroundColor: keys.map((k) => alpha(policyColor(k), 0.8)), borderRadius: 4 }] },
      options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max } } },
    });
  bar("val-lap", "Laps completed", keys.map((k) => V[k].lap_success_rate), 1, {
    what: "Share of the validation tracks where each driver finished a full lap. Same tracks and difficulty for everyone.",
    good: `Higher is better. 0.7 is the MVP target and pure pursuit, the hand written driver, reaches ${pct(pp.lap_success_rate)}. Random should be near 0.`,
    read: "This is the main number. PPO beating random proves it learned something, matching pure pursuit means it drives as reliably as a driver that knows the track.",
  });
  bar("val-time", "Lap time, finished laps only (s)", keys.map((k) => V[k].mean_lap_time), undefined, {
    what: "Average time of the laps that were completed. Failed attempts are left out.",
    good: `Lower is better. Pure pursuit needs ${num(pp.mean_lap_time)} s.`,
    read: "Only fair next to the laps completed chart. A driver that finishes few laps can look fast because only its easy tracks count.",
  });
  bar("val-speed", "Average speed (km/h)", keys.map((k) => V[k].mean_speed * 3.6), undefined, {
    what: "Forward speed averaged over every step of every attempt, crashes included.",
    good: `Higher is better if laps are still finished. Pure pursuit averages ${num((pp.mean_speed || 0) * 3.6, 0)} km/h and the car tops out at ${num(D.meta.max_speed * 3.6, 0)} km/h.`,
    read: "High speed with many crashes means it is reckless. Very low speed means it hesitates or stands still.",
  });
  const outcomes = [
    ["lap", (m) => m.lap_success_rate],
    ["off_road", (m) => m.off_track_rate],
    ["stuck", (m) => m.stuck_rate],
    ["timeout", (m) => Math.max(0, 1 - m.lap_success_rate - m.off_track_rate - m.stuck_rate)],
  ];
  makeChart(chartCard(el, "val-out", "How each attempt ends", {
    what: "Share of attempts that ended with a finished lap, off road (barrier or over 10 s on grass), stuck (under 2 m gained in 10 s) or the 120 s time limit.",
    good: "All green. Some red is acceptable while learning, orange should never be large.",
    read: "Red means it drives too fast or turns too late. Orange means it gives up and stands still, the reward should make that the worst option. Grey means it drives but too slowly.",
  }), {
    type: "bar",
    data: {
      labels: keys.map(policyName),
      datasets: outcomes.map(([o, f]) => ({ label: OUTCOME[o][0], data: keys.map((k) => f(V[k])), backgroundColor: OUTCOME[o][1], borderRadius: 2 })),
    },
    options: { plugins: { legend: { position: "bottom" } }, scales: { x: { stacked: true }, y: { stacked: true, min: 0, max: 1 } } },
  });
}

// ---------- statistics ----------
const fmtCi = (ci, f = pct) => (ci && ci[0] != null ? `${f(ci[0])} to ${f(ci[1])}` : "-");
const fmtP = (p) => (p == null ? "n/a" : p < 0.0001 ? "< 0.0001" : num(p, 4));

function intervalChart(el, id, title, S, key, info) {
  const keys = Object.keys(S.intervals);
  const colors = keys.map(policyColor);
  makeChart(chartCard(el, id, title, info), {
    data: {
      labels: keys.map(policyName),
      datasets: [
        { type: "bar", label: "95% interval", data: keys.map((k) => S.intervals[k][key + "_ci"]), backgroundColor: colors.map((c) => alpha(c, 0.25)), borderColor: colors, borderWidth: 1, borderSkipped: false, barPercentage: 0.4 },
        { type: "line", label: "estimate", data: keys.map((k) => S.intervals[k][key]), showLine: false, pointRadius: 5, pointBackgroundColor: colors, pointBorderColor: colors },
      ],
    },
    options: { plugins: { legend: { position: "bottom" } }, scales: { y: { min: 0, max: 1 } } },
  });
}

function infoCard(title, info, body) {
  return `<div class="card"><div class="chart-title">${title}${infoButton(info)}</div><div class="table-wrap">${body}</div></div>`;
}

function renderStatistics() {
  levelTabs($("#stat-level"));
  const L = level();
  const S = L.stats;
  const body = $("#stat-body");
  if (!S) {
    body.innerHTML = `<div class="card empty">${icon("alert")}<p>This data.js was built before statistics existed. Rebuild with <code>make dashboard</code>.</p></div>`;
    return;
  }
  $("#stat-note").textContent = `Difficulty ${num(L.difficulty)}, first ${D.meta.eval_tracks} validation tracks from 1000. Every policy drove the same tracks, so every test is paired by track.`;
  body.innerHTML = `<div class="grid-2" id="stat-charts"></div>`;
  const charts = $("#stat-charts");
  intervalChart(charts, "stat-lap", "Laps completed with 95% interval", S, "lap_rate", {
    what: "Dot is the share of tracks lapped, the bar is the range where the true rate likely lies. One model uses the Wilson interval, pooled seeds a bootstrap over tracks.",
    good: "A high dot with a short bar. Bars of two drivers that do not overlap mean a clear difference.",
    read: "With few tracks the bars are wide. 20 of 20 laps is still compatible with about 84%, so a perfect score is not proof of perfection.",
  });
  intervalChart(charts, "stat-progress", "Share of the lap driven with 95% interval", S, "progress", {
    what: "Mean progress over tracks with a bootstrap interval, 1 is a full lap.",
    good: "Close to 1 with a short bar.",
    read: "Separates a driver that almost finishes from one that never starts, even when both lap 0%.",
  });

  const rows = Object.entries(S.intervals).map(([k, s]) => {
    const seeds = s.lap_rate_per_seed ? s.lap_rate_per_seed.map(pct).join(", ") : "-";
    return `<tr><td><span class="dot" style="display:inline-block;background:${policyColor(k)}"></span> ${policyName(k)}</td><td>${s.n_models}</td><td>${s.n_tracks}</td><td>${pct(s.lap_rate)}</td><td>${fmtCi(s.lap_rate_ci)}</td><td>${seeds}</td><td>${fmtCi(s.lap_rate_seed_ci)}</td><td>${pct(s.progress)}</td><td>${fmtCi(s.progress_ci)}</td></tr>`;
  });
  const intervals = `<table><thead><tr><th>policy</th><th>models</th><th>tracks</th><th>laps</th><th>95% interval</th><th>per seed</th><th>seed t interval</th><th>lap driven</th><th>95% interval</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;

  const tests = S.tests.map((t) => {
    const pa = t.a.startsWith("pooled:"), pb = t.b.startsWith("pooled:");
    const seeds = !t.models ? "" : pa && pb ? `, seed ${t.models[0]} of each` : pa ? `, seed ${t.models[0]}` : pb ? `, seed ${t.models[1]}` : "";
    const verdict = t.p_holm == null ? `<span class="sig">no pairs</span>` : t.p_holm < 0.05 ? `<span class="sig yes">differ</span>` : `<span class="sig">no evidence</span>`;
    return `<tr><td>${policyName(t.a)} vs ${policyName(t.b)}</td><td>${t.test}${seeds}</td><td>${t.detail}</td><td>${fmtP(t.p)}</td><td>${fmtP(t.p_holm)}</td><td>${verdict}</td></tr>`;
  });
  const testTable = `<table><thead><tr><th>pair</th><th>test</th><th>detail</th><th>p</th><th>p Holm</th><th>result</th></tr></thead><tbody>${tests.join("")}</tbody></table>`;
  const friedman = S.friedman == null ? "" : `<p class="muted">Friedman test on per track progress over all drivers above, p ${fmtP(S.friedman)}. Under 0.05 means at least one driver differs from the rest.</p>`;
  const note = S.note ? `<p class="muted">${S.note}</p>` : "";

  body.insertAdjacentHTML("beforeend", infoCard("Intervals per policy", {
    what: "Each row is one driver. Pooled rows join every seed of one PPO config, with the lap rate of each seed and a Student t interval over seeds.",
    good: "Per seed values close together. The seed t interval tells whether the training method is reliable, not just one lucky model.",
    read: "The track interval measures how well these models drive. The seed interval measures how much a new training run could differ, it is wide with 3 seeds.",
  }, intervals));
  body.insertAdjacentHTML("beforeend", infoCard("Paired tests", {
    what: "McNemar compares laps using only the tracks where one driver lapped and the other did not. Wilcoxon compares progress and lap time track by track. Holm raises every p to account for running many tests at once.",
    good: "Result differ means p Holm under 0.05, a difference unlikely to come from luck on these tracks.",
    read: "No evidence is not proof of equality, with few tracks real small differences can hide. Detail shows the discordant tracks or the median difference.",
  }, testTable + friedman + note));
}

function renderTest() {
  const rows = D.test;
  $("#test-body").innerHTML = Object.keys(rows).length
    ? `<div class="card"><div class="table-wrap">${metricsTable(rows)}</div></div>`
    : `<div class="card empty">${icon("lock")}<p>Test seeds (10000 and up) are run once by S3 for the final models of T11.<br>Nothing is reported here before that, and nothing is ever tuned on them.</p></div>`;
}

// ---------- simulation ----------
const simState = { mode: "policy", pick: null, playing: false, t: 0, speed: 1, tiles: [], maxT: 0, lastTs: 0 };

function simTiles() {
  if (simState.mode === "policy") {
    const key = D.replays[simState.pick] ? simState.pick : mainPolicy();
    simState.pick = key;
    return D.replays[key].map((ep) => ({ ep, key, title: `seed ${ep.seed}` }));
  }
  const seed = D.tracks[simState.pick] ? simState.pick : Object.keys(D.tracks)[0];
  simState.pick = seed;
  return Object.entries(D.replays).map(([key, eps]) => ({ ep: eps.find((e) => String(e.seed) === String(seed)), key, title: policyName(key) }));
}

function renderSimulation() {
  $("#sim-mode").innerHTML = [["policy", "One policy, many tracks"], ["track", "One track, all policies"]]
    .map(([m, t]) => `<button data-mode="${m}" class="${simState.mode === m ? "active" : ""}">${t}</button>`)
    .join("");
  $$("#sim-mode button").forEach((b) =>
    b.addEventListener("click", () => {
      simState.mode = b.dataset.mode;
      simState.pick = null;
      renderSimulation();
    })
  );
  const tiles = simTiles();
  const options = simState.mode === "policy" ? Object.keys(D.replays).map((k) => [k, policyName(k)]) : Object.keys(D.tracks).map((s) => [s, `track seed ${s}`]);
  $("#sim-pick").innerHTML = options.map(([v, t]) => `<option value="${v}" ${String(v) === String(simState.pick) ? "selected" : ""}>${t}</option>`).join("");

  const grid = $("#sim-grid");
  grid.innerHTML = "";
  simState.tiles = tiles.filter((t) => t.ep).map((t, i) => {
    grid.insertAdjacentHTML(
      "beforeend",
      `<div class="card tile"><canvas id="sim-${i}"></canvas><div class="tile-body">
        <div class="row between"><span class="tile-title">${t.title}</span><span id="sim-status-${i}"></span></div>
        <div class="tile-foot"><span id="sim-info-${i}"></span><span>${episodeNote(t.ep)}</span></div></div></div>`
    );
    return { ...t, canvas: $(`#sim-${i}`), i, color: policyColor(t.key) };
  });
  simState.tiles.forEach((tile) => Object.assign(tile, sizeCanvas(tile.canvas)));
  simState.maxT = Math.max(0, ...simState.tiles.map((t) => t.ep.frames.x.length * t.ep.dt));
  $("#speed").innerHTML = [1, 2, 4, 8].map((s) => `<button data-speed="${s}" class="${s === simState.speed ? "active" : ""}">${s}x</button>`).join("");
  $$("#speed button").forEach((b) =>
    b.addEventListener("click", () => {
      simState.speed = Number(b.dataset.speed);
      $$("#speed button").forEach((x) => x.classList.toggle("active", x === b));
    })
  );
  drawSim();
}

function drawSim() {
  const lidarOn = $("#opt-lidar").checked, follow = $("#opt-follow").checked, trail = $("#opt-trail").checked;
  simState.tiles.forEach((tile) => {
    const { ctx, w, h, ep } = tile;
    const track = D.tracks[ep.seed];
    const f = ep.frames;
    const n = f.x.length;
    const i = Math.max(0, Math.min(Math.floor(simState.t / ep.dt), n - 1));
    const view = follow ? followView(f.x[i], f.y[i], w, h, 4) : fitView(track, w, h);
    drawTrack(ctx, track, view, w, h);
    if (trail) drawTrail(ctx, f, i, view);
    drawCar(ctx, f, i, view, tile.color, lidarOn);
    const speed = Math.hypot(f.v[i], f.v_lat[i]) * 3.6;
    $(`#sim-info-${tile.i}`).textContent = `${speed.toFixed(0)} km/h, ${pct(Math.max(0, f.progress[i]))}`;
    $(`#sim-status-${tile.i}`).innerHTML = i === n - 1 ? statusHtml(ep) : `<span class="status"><span class="dot" style="background:${css("--accent")}"></span>Driving</span>`;
  });
  $("#clock").textContent = `${simState.t.toFixed(1)} s`;
  $("#scrub").value = simState.maxT ? Math.round((1000 * simState.t) / simState.maxT) : 0;
}

function setPlaying(on) {
  simState.playing = on;
  if (on && simState.t >= simState.maxT) simState.t = 0;
  $("#play").innerHTML = `${icon(on ? "pause" : "play")}${on ? "Pause" : "Play"}`;
  simState.lastTs = performance.now();
  if (on) requestAnimationFrame(tick);
}

function tick(ts) {
  if (!simState.playing || state.view !== "simulation") return;
  // The first frame timestamp can be older than performance.now() in setPlaying, so never step back
  const dt = Math.max(0, ts - simState.lastTs) / 1000;
  simState.t = Math.min(simState.t + dt * simState.speed, simState.maxT);
  simState.lastTs = ts;
  drawSim();
  if (simState.t >= simState.maxT) setPlaying(false);
  else requestAnimationFrame(tick);
}

// ---------- physics ----------
function renderPhysicsSelectors() {
  const keys = Object.keys(D.replays);
  const pol = $("#phys-policy"), seed = $("#phys-seed");
  if (!pol.options.length) {
    pol.innerHTML = keys.map((k) => `<option value="${k}">${policyName(k)}</option>`).join("");
    seed.innerHTML = Object.keys(D.tracks).map((s) => `<option value="${s}">track seed ${s}</option>`).join("");
    pol.value = mainPolicy();
  }
}

function renderPhysics() {
  renderPhysicsSelectors();
  const key = $("#phys-policy").value;
  const seed = $("#phys-seed").value;
  const ep = D.replays[key].find((e) => String(e.seed) === seed);
  const track = D.tracks[seed];
  const f = ep.frames;
  const t = f.x.map((_, i) => +(i * ep.dt).toFixed(2));
  const color = policyColor(key);
  const series = (arr, label, c = color, extra = {}) => ({ label, data: arr.map((y, i) => ({ x: t[i], y })), borderColor: c, backgroundColor: c, borderWidth: 1.5, ...extra });
  const flat = (y, label, c) => series(t.map(() => y), label, c, { borderDash: [4, 4], borderWidth: 1 });
  const el = $("#phys-charts");
  el.innerHTML = "";
  const half = track.width / 2, wall = half + track.width;

  makeChart(chartCard(el, "ph-speed", "Speed [km/h]"), {
    type: "line",
    data: { datasets: [series(f.v.map((v) => v * 3.6), "forward"), series(f.v.map((v, i) => Math.hypot(v, f.v_lat[i]) * 3.6), "total", "#94a3b8")] },
    options: lineOptions("time [s]"),
  });
  makeChart(chartCard(el, "ph-slip", "Sideways slip [km/h], drift when large"), {
    type: "line",
    data: { datasets: [series(f.v_lat.map((v) => v * 3.6), "slip", "#ea580c")] },
    options: lineOptions("time [s]", { legend: false }),
  });
  makeChart(chartCard(el, "ph-steer", "Steering, command in [-1, 1] and wheel angle [rad]"), {
    type: "line",
    data: { datasets: [series(f.steer_cmd, "command", "#94a3b8"), series(f.steer, "wheel angle")] },
    options: lineOptions("time [s]"),
  });
  makeChart(chartCard(el, "ph-lat", "Lateral offset [m], grass past the dashed lines"), {
    type: "line",
    data: { datasets: [series(f.lateral, "offset"), flat(half, "_g1", "#16a34a"), flat(-half, "_g2", "#16a34a"), flat(wall, "_w1", "#dc2626"), flat(-wall, "_w2", "#dc2626")] },
    options: lineOptions("time [s]", { legend: false }),
  });
  makeChart(chartCard(el, "ph-throttle", "Throttle and brake command"), {
    type: "line",
    data: { datasets: [series(f.throttle, "throttle", "#0891b2", { fill: true, backgroundColor: alpha("#0891b2", 0.12) })] },
    options: lineOptions("time [s]", { legend: false, y: { min: -1, max: 1 } }),
  });

  // jitter is the step to step change of the steering command, compared across policies
  const bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 2.01];
  const labels = bins.slice(0, -1).map((b, i) => `${b}-${bins[i + 1] > 2 ? "2" : bins[i + 1]}`);
  const datasets = Object.entries(D.replays).map(([k, eps]) => {
    const counts = new Array(labels.length).fill(0);
    let total = 0;
    eps.forEach((e) =>
      e.frames.steer_cmd.forEach((s, i) => {
        if (!i) return;
        const d = Math.abs(s - e.frames.steer_cmd[i - 1]);
        const b = bins.findIndex((edge, j) => d >= edge && d < bins[j + 1]);
        if (b >= 0) counts[b]++;
        total++;
      })
    );
    return { label: policyName(k), data: counts.map((c) => c / Math.max(total, 1)), backgroundColor: alpha(policyColor(k), 0.8), borderRadius: 3 };
  });
  makeChart(chartCard(el, "ph-jitter", "Steering change per step (jitter), share of steps"), {
    type: "bar",
    data: { labels, datasets },
    options: { plugins: { legend: { position: "bottom" } }, scales: { y: { min: 0 } } },
  });

  const canvas = $("#phys-map");
  const { ctx, w, h } = sizeCanvas(canvas);
  const view = fitView(track, w, h);
  drawTrack(ctx, track, view, w, h);
  drawTrail(ctx, f, f.x.length - 1, view);
  drawCar(ctx, f, f.x.length - 1, view, color, false);
  Object.entries(D.replays).forEach(([k, eps]) => {
    const e = eps.find((x) => String(x.seed) === seed);
    if (!e || e.outcome === "lap") return;
    const n = e.frames.x.length - 1;
    const [px, py] = view.map(e.frames.x[n], e.frames.y[n]);
    ctx.strokeStyle = policyColor(k);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px - 6, py - 6);
    ctx.lineTo(px + 6, py + 6);
    ctx.moveTo(px + 6, py - 6);
    ctx.lineTo(px - 6, py + 6);
    ctx.stroke();
  });
  $("#phys-legend").innerHTML = `<span style="color:${speedColor(0)}">slow</span><span>trail colored by speed, X marks failures</span><span style="color:${speedColor(D.meta.max_speed)}">fast</span>`;

  const fails = [];
  Object.entries(D.replays).forEach(([k, eps]) => eps.forEach((e) => e.outcome !== "lap" && fails.push([k, e])));
  $("#failures").innerHTML = fails.length
    ? fails
        .map(([k, e]) => `<div class="list-item" data-k="${k}" data-seed="${e.seed}"><div class="icon-box" style="color:${OUTCOME[e.outcome][1]}">${icon("alert")}</div>
          <div class="grow"><div>${policyName(k)}, seed ${e.seed}</div><div class="sub">${OUTCOME[e.outcome][0]} at ${pct(e.progress)} after ${(e.frames.x.length * e.dt).toFixed(1)} s</div></div></div>`)
        .join("")
    : `<div class="empty">No failures on the replay tracks</div>`;
  $$("#failures .list-item").forEach((item) =>
    item.addEventListener("click", () => {
      $("#phys-policy").value = item.dataset.k;
      $("#phys-seed").value = item.dataset.seed;
      renderPhysics();
    })
  );
}

// ---------- shell ----------
function renderNav() {
  $("#nav").innerHTML = VIEWS.map(([id, label, ic]) => {
    const count = id === "training" ? D.runs.length : id === "simulation" ? Object.keys(D.replays).length : "";
    return `<button data-view="${id}" class="${state.view === id ? "active" : ""}">${icon(ic)}${label}${count ? `<span class="badge">${count}</span>` : ""}</button>`;
  }).join("");
  $$("#nav button").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
}

function renderRunList() {
  const runs = visibleRuns();
  $("#run-count").textContent = runs.length;
  $("#run-list").innerHTML = runs
    .map((r) => `<li data-run="${r.name}" class="${state.hidden.has(r.name) ? "" : "selected"}"><span class="dot" style="background:${runColor(r.name)}"></span>${r.name}</li>`)
    .join("");
  $$("#run-list li").forEach((li) =>
    li.addEventListener("click", () => {
      state.hidden.clear();
      runs.filter((r) => r.name !== li.dataset.run).forEach((r) => state.hidden.add(r.name));
      go("training");
    })
  );
}

const RENDER = {
  overview: () => {
    renderHero();
    renderTabs();
    renderKpis();
    renderBoard();
    renderPolicies();
  },
  training: renderTraining,
  validation: renderValidation,
  statistics: renderStatistics,
  test: renderTest,
  simulation: renderSimulation,
  physics: renderPhysics,
};

function go(view) {
  state.view = view;
  $$(".view").forEach((v) => (v.style.display = v.dataset.view === view ? "" : "none"));
  $("#crumb-text").textContent = VIEWS.find((v) => v[0] === view)[1];
  if (view !== "simulation") setPlaying(false);
  renderNav();
  RENDER[view]();
}

function renderAll() {
  chartDefaults();
  renderRunList();
  go(state.view);
}

function applyTheme(choice) {
  localStorage.setItem("theme", choice);
  const dark = choice === "dark" || (choice === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $$("#theme button").forEach((b) => b.classList.toggle("active", b.dataset.themeSet === choice));
  renderAll();
}

function bindShell() {
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => go(b.dataset.goto)));
  $$("#theme button").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeSet)));
  $("#crumb").addEventListener("click", () => go("overview"));
  $("#toggle-side").addEventListener("click", () => document.body.classList.toggle("side-hidden"));
  $("#search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderRunList();
    if (state.view === "training" || state.view === "overview") RENDER[state.view]();
  });
  $("#play").addEventListener("click", () => setPlaying(!simState.playing));
  $("#scrub").addEventListener("input", (e) => {
    simState.t = (Number(e.target.value) / 1000) * simState.maxT;
    drawSim();
  });
  $("#sim-pick").addEventListener("change", (e) => {
    simState.pick = e.target.value;
    renderSimulation();
  });
  ["#opt-lidar", "#opt-follow", "#opt-trail"].forEach((id) => $(id).addEventListener("change", drawSim));
  $("#phys-policy").addEventListener("change", renderPhysics);
  $("#phys-seed").addEventListener("change", renderPhysics);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;
    if (e.key === "/") {
      e.preventDefault();
      $("#search").focus();
    }
    if (e.key === " " && state.view === "simulation") {
      e.preventDefault();
      setPlaying(!simState.playing);
    }
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => RENDER[state.view](), 150);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (localStorage.getItem("theme") === "system") applyTheme("system");
  });
}

if (!D) {
  document.body.innerHTML = '<div class="empty">No data.js found. Run <code>uv run python -m dashboard.build --runs &lt;run&gt;</code> first.</div>';
} else {
  fillIcons();
  $("#meta-line").textContent = `${D.runs.length} runs, ${Object.keys(D.validation).length} policies`;
  bindShell();
  applyTheme(localStorage.getItem("theme") || "light");
}
