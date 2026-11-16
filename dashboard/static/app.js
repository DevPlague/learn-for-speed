// learn-for-speed dashboard. Reads window.DATA written by dashboard/build.py, no server needed.
// Page copy is in Spanish on purpose, code and comments stay in English.
"use strict";

const D = window.DATA;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const ICONS = {
  flag: '<path d="M4 22V4"/><path d="M4 4h12l-2 4 2 4H4"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  play: '<path d="M7 4v16l13-8z"/>',
  pause: '<path d="M7 4h4v16H7zM14 4h4v16h-4z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  monitor: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
  car: '<path d="M5 16h14l-1.5-6h-11z"/><circle cx="8" cy="17" r="2"/><circle cx="16" cy="17" r="2"/>',
  bot: '<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01"/>',
  shuffle: '<path d="M3 6h4l10 12h4M3 18h4l3-3.6M14 9.6 17 6h4"/>',
  alert: '<path d="M12 3 2 21h20z"/><path d="M12 10v4M12 18h.01"/>',
};

// id, menu label, line under it
const VIEWS = [
  ["overview", "Inicio", "resumen"],
  ["race", "Carrera", "arcade contra el modelo"],
  ["training", "Entrenamiento", "curvas de aprendizaje"],
  ["validation", "Validación", "pistas 1000 a 1999"],
  ["statistics", "Estadística", "intervalos y tests"],
  ["test", "Test", "resultado final"],
  ["simulation", "Repeticiones", "vueltas grabadas"],
  ["physics", "Física", "volante y derrape"],
];
// families of model versions for the training picker, first match wins
const FAMILIES = [
  ["final", "Final", (c) => c === "final"],
  ["curriculum", "Curriculum", (c) => c.startsWith("mvp_curriculum")],
  ["exp", "Experimentos", (c) => c.startsWith("exp")],
  ["mvp", "MVP", (c) => c.startsWith("mvp")],
  ["probe", "Sondas", (c) => c.startsWith("probe")],
  ["other", "Otras", () => true],
];
const OUTCOME = {
  lap: ["Vuelta", "--out-lap"],
  off_road: ["Fuera de pista", "--out-off"],
  stuck: ["Atasco", "--out-stuck"],
  timeout: ["Tiempo agotado", "--out-time"],
};
const METRIC_LABELS = {
  lap_success_rate: "vueltas completadas",
  off_track_rate: "salidas de pista",
  stuck_rate: "atascos",
  mean_lap_time: "tiempo de vuelta (s)",
  mean_speed: "velocidad media (km/h)",
  mean_progress: "vuelta recorrida",
};
const TEST_NAMES = {
  "lap McNemar": "vueltas (McNemar)",
  "progress Wilcoxon": "vuelta recorrida (Wilcoxon)",
  "lap time both lapped Wilcoxon": "tiempo de vuelta en pistas que acaban los dos (Wilcoxon)",
};
const CAR_YELLOW = "#ffd23f";
const PLAYER_BLUE = "#2f6fd6";

const state = { view: "overview", hidden: new Set(), charts: {}, search: "", level: 0 };

// ---------- small helpers ----------
const icon = (name) =>
  `<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const palette = () => [1, 2, 3, 4, 5, 6].map((i) => css(`--c${i}`));
const outcomeColor = (o) => css(OUTCOME[o][1]);
const pct = (x) => (x == null ? "-" : `${Math.round(100 * x)} %`);
const num = (x, d = 1) => (x == null ? "-" : Number(x).toFixed(d).replace(".", ","));
const configOf = (name) => name.replace(/_s\d+$/, "");
const familyOf = (name) => FAMILIES.find(([, , match]) => match(configOf(name)))[0];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const mod1 = (a) => ((a % 1) + 1) % 1;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const isPpo = (key) => key.startsWith("ppo:") || key.startsWith("pooled:");
const timeoutRate = (m) => Math.max(0, 1 - m.lap_success_rate - m.off_track_rate - m.stuck_rate);
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
// seven segment readouts keep the dot, the DSEG face has no comma
const digit = (x, d = 1) => (x == null ? "--" : Number(x).toFixed(d));
const LEVEL_NAMES = { 0: "Fácil", 0.5: "Media", 1: "Difícil" };
const levelName = (d) => LEVEL_NAMES[Number(d)] || `dificultad ${num(d)}`;

function runLabel(name) {
  const run = D.runs.find((r) => r.name === name);
  return run && run.label ? run.label : name;
}

function configLabel(config) {
  const run = D.runs.find((r) => configOf(r.name) === config);
  return run && run.label ? run.label.replace(/ · semilla \d+$/, "") : config;
}

function policyName(key) {
  if (key === "pure_pursuit") return "Pure pursuit (referencia)";
  if (key === "random") return "Random (referencia)";
  if (key.startsWith("pooled:")) return `PPO ${configLabel(key.slice(7))} (todas las semillas)`;
  const test = key.match(/^eval_(.+)_test(?:_d(\d+))?$/);
  if (test) {
    const who = test[1] === "pure_pursuit" ? "Pure pursuit" : test[1] === "random" ? "Random" : `PPO ${configLabel(test[1])}`;
    return test[2] != null ? `${who} · ${levelName(test[2]).toLowerCase()}` : who;
  }
  return `PPO ${runLabel(key.replace(/^ppo:/, ""))}`;
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
  if (key === "pure_pursuit" || /^eval_pure_pursuit/.test(key)) return css("--pp");
  if (key === "random" || /^eval_random/.test(key)) return css("--rnd");
  const ppo = Object.keys(D.validation).filter((k) => k.startsWith("ppo:"));
  if (key.startsWith("pooled:")) key = ppo.find((k) => configOf(k.slice(4)) === key.slice(7)) || key;
  const colors = palette();
  return colors[Math.max(ppo.indexOf(key), 0) % colors.length];
}

function runColor(name) {
  const configs = [...new Set(D.runs.map((r) => configOf(r.name)))];
  const colors = palette();
  return colors[configs.indexOf(configOf(name)) % colors.length];
}

function hex(color) {
  return color.startsWith("#") ? color : "#8a817d";
}

function alpha(color, a) {
  const n = parseInt(hex(color).slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function mix(a, b, t) {
  const pa = parseInt(hex(a).slice(1), 16), pb = parseInt(hex(b).slice(1), 16);
  const ch = (p, k) => (p >> k) & 255;
  return `rgb(${[16, 8, 0].map((k) => Math.round(ch(pa, k) + (ch(pb, k) - ch(pa, k)) * t)).join(",")})`;
}

function level() {
  if (!D.levels) return { difficulty: D.meta.difficulty, validation: D.validation, stats: null };
  return D.levels[Math.min(state.level, D.levels.length - 1)];
}

function levelTabs(el) {
  const levels = D.levels || [];
  el.innerHTML = levels.length < 2 ? "" : levels
    .map((l, i) => `<button data-level="${i}" class="${i === state.level ? "active" : ""}">${levelName(l.difficulty)}</button>`)
    .join("");
  $$("button", el).forEach((b) =>
    b.addEventListener("click", () => {
      state.level = Number(b.dataset.level);
      safeRender(state.view);
    })
  );
}

function tabs(el, options, current, onPick) {
  el.innerHTML = options.map(([v, t]) => `<button data-v="${v}" class="${v === current ? "active" : ""}">${t}</button>`).join("");
  $$("button", el).forEach((b) => b.addEventListener("click", () => onPick(b.dataset.v)));
}

function mainPolicy() {
  return D.meta.selected.find((k) => D.validation[k]) || Object.keys(D.validation)[0];
}

const store = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch (e) {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (e) {
      /* storage blocked, the page works without it */
    }
  },
};

// ---------- charts ----------
function chartDefaults() {
  Chart.defaults.font.family = '"Oxygen", "Segoe UI", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.locale = "es-ES";
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = alpha(css("--muted"), 0.22);
  Chart.defaults.maintainAspectRatio = false;
  // liquid entrance, values pour up from the baseline in a wave from left to right and settle softly
  const wave = (c) => {
    if (c.type !== "data" || c.mode !== "default") return 0;
    const n = c.chart.data.datasets[c.datasetIndex].data.length || 1;
    return (c.dataIndex / n) * 520 + c.datasetIndex * 80;
  };
  // no overshoot, a bar that passes the top of the axis gets clipped and looks cut
  Chart.defaults.animation = REDUCED ? false : { duration: 700, easing: "easeOutQuart", delay: wave };
  // hovering only highlights, quickly and without overshoot
  Chart.defaults.transitions.active.animation.duration = REDUCED ? 0 : 120;
  Chart.defaults.interaction = { mode: "nearest", intersect: false, axis: "x" };
  const tip = Chart.defaults.plugins.tooltip;
  Object.assign(tip, {
    backgroundColor: css("--text"),
    titleColor: css("--bg"),
    bodyColor: css("--bg"),
    borderColor: css("--accent"),
    borderWidth: 2,
    cornerRadius: 12,
    padding: 10,
    caretSize: 6,
    boxPadding: 4,
    usePointStyle: true,
    position: "nearest",
  });
  tip.titleFont = { family: '"Barlow Condensed", sans-serif', weight: "700", size: 15 };
  tip.bodyFont = { family: '"Oxygen", sans-serif', size: 12 };
  tip.animation = REDUCED ? false : { duration: 120, easing: "easeOutQuad" };
  Chart.defaults.elements.line.borderWidth = 2.5;
  Chart.defaults.elements.line.tension = 0.3;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 5;
  Chart.defaults.elements.point.hoverBorderWidth = 2;
  Chart.defaults.elements.bar.borderWidth = 2;
  Chart.defaults.elements.bar.borderColor = css("--line");
  Chart.defaults.elements.bar.borderRadius = 8;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.boxHeight = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.legend.labels.font = { family: '"Barlow Condensed", sans-serif', weight: "600", size: 14 };
  Chart.defaults.plugins.legend.labels.filter = (item) => !(item.text || "").startsWith("_");
}

function pruneCharts() {
  Object.entries(state.charts).forEach(([id, chart]) => {
    if (!chart.canvas || !chart.canvas.isConnected) {
      chart.destroy();
      delete state.charts[id];
    }
  });
}

// charts inside a container must die before its HTML is wiped, or they keep animating on dead canvases
function destroyChartsIn(el) {
  Object.entries(state.charts).forEach(([id, chart]) => {
    if (!chart.canvas || !chart.canvas.isConnected || el.contains(chart.canvas)) {
      chart.destroy();
      delete state.charts[id];
    }
  });
  el.innerHTML = "";
}

function chartError(canvas, err) {
  console.error(err);
  const box = canvas.closest(".chart-box") || canvas.parentElement;
  if (box) box.innerHTML = `<div class="empty">No se pudo dibujar esta gráfica. ${esc(err.message)}</div>`;
}

function makeChart(canvas, config) {
  const id = canvas.id;
  if (state.charts[id]) state.charts[id].destroy();
  try {
    state.charts[id] = new Chart(canvas, config);
  } catch (err) {
    delete state.charts[id];
    chartError(canvas, err);
    return null;
  }
  return state.charts[id];
}

// charts measure their box when created, so measure again once the view has settled
function resizeCharts() {
  Object.values(state.charts).forEach((c) => {
    if (c.canvas && c.canvas.isConnected && c.canvas.offsetParent !== null) c.resize();
  });
}

// info is {what, good, read}, only the parts given are shown
function infoButton(info) {
  if (!info) return "";
  const rows = [
    ["Qué es", info.what],
    ["Buen resultado", info.good],
    ["Cómo leerlo", info.read],
  ].filter(([, text]) => text);
  const body = rows.length === 1 ? `<span>${rows[0][1]}</span>` : rows.map(([label, text]) => `<b>${label}</b><span>${text}</span>`).join("");
  return `<span class="info" tabindex="0" aria-label="Más información">i<span class="tip">${body}</span></span>`;
}

function chartCard(parent, id, title, info) {
  parent.insertAdjacentHTML(
    "beforeend",
    `<div class="card chart-card"><div class="chart-title">${title}${infoButton(info)}</div><div class="chart-box"><canvas id="${id}"></canvas></div></div>`
  );
  return $(`#${id}`);
}

// extra.fast draws without animation and decimates long series, for the per step physics traces
function lineOptions(xTitle, extra = {}) {
  return {
    parsing: false,
    normalized: true,
    ...(extra.fast ? { animation: false } : {}),
    scales: {
      x: { type: "linear", title: { display: !!xTitle, text: xTitle }, grid: { display: false } },
      y: { beginAtZero: false, ...extra.y },
    },
    plugins: {
      legend: { display: extra.legend !== false, position: "bottom" },
      ...(extra.fast ? { decimation: { enabled: true, algorithm: "min-max", samples: 400 } } : {}),
    },
  };
}

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
    const name = configLabel(config);
    const label = tables.length > 1 ? `${name} (${tables.length} semillas)` : name;
    if (tables.length > 1) {
      datasets.push({ label: `_lo${config}`, data: lo, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false });
      datasets.push({ label: `_hi${config}`, data: hi, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: "-1", backgroundColor: alpha(color, 0.15) });
    }
    datasets.push({ label, data: mean, borderColor: color, backgroundColor: color });
  });
  return datasets;
}

// ---------- pixel art ----------
// Tracks are drawn into a canvas at 1/PX resolution and scaled up without smoothing. Curbs, tire
// walls, ground and the flag are decoration on top of the real geometry.
const PX = 3;
const lowCanvas = document.createElement("canvas");
const decorCache = new WeakMap();

// world palettes for the arcade, the first one follows the page colors elsewhere
const WORLD_THEMES = [
  { grass: "#9fc27a", grass2: "#93b76e", tuft: "#6c9150", road: "#4a4548", edge: "#f7f0e1", wall: "#2d2a2e", curb: "#d9483f", curb2: "#f7f0e1", center: "#8f8a85", deco: "grass" },
  { grass: "#e8eff4", grass2: "#d8e3ec", tuft: "#2e5a4c", road: "#596273", edge: "#ffffff", wall: "#24324a", curb: "#3f73d6", curb2: "#ffffff", center: "#a3afbf", deco: "snow" },
  { grass: "#2b1d1a", grass2: "#36231f", tuft: "#ff6a1a", road: "#3b3535", edge: "#ffb347", wall: "#0c0808", curb: "#ff6a1a", curb2: "#1b1414", center: "#75655f", deco: "lava" },
];
const WORLD_NAMES = ["Valle verde", "Puerto nevado", "Volcán"];

function pageTheme() {
  return {
    grass: css("--grass"), grass2: css("--grass2"), tuft: css("--tuft"), road: css("--road"), edge: css("--edge"),
    wall: css("--wall"), curb: css("--curb"), curb2: css("--edge"), center: css("--center"), deco: "grass",
  };
}

function bounds(track) {
  const pts = track.left_wall.concat(track.right_wall);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

function decor(track) {
  if (decorCache.has(track)) return decorCache.get(track);
  const c = track.centerline, n = c.length;
  const curb = c.map((p, i) => {
    const a = c[(i - 1 + n) % n], b = c[(i + 1) % n];
    const h1 = Math.atan2(p[1] - a[1], p[0] - a[0]), h2 = Math.atan2(b[1] - p[1], b[0] - p[0]);
    const turn = Math.abs(Math.atan2(Math.sin(h2 - h1), Math.cos(h2 - h1)));
    const seg = Math.hypot(b[0] - p[0], b[1] - p[1]);
    return seg / Math.max(turn, 1e-6) < 28; // tighter than 28 m gets curbs
  });
  const b = bounds(track);
  let seed = Math.round(track.length * 1000) % 2147483647 || 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const tufts = Array.from({ length: 140 }, () => [b.x0 + rnd() * (b.x1 - b.x0), b.y0 + rnd() * (b.y1 - b.y0), rnd() * 6.28]);
  const out = { curb, tufts };
  decorCache.set(track, out);
  return out;
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

function drawGround(o, d, lv, lw, lh, s, th, time) {
  o.fillStyle = th.grass;
  o.fillRect(0, 0, lw, lh);
  // checker tiles anchored to the world so they scroll with the camera
  const tile = Math.max(2, Math.round(6 * s));
  const [ax, ay] = lv.map(0, 0);
  const sx = ((ax % (2 * tile)) + 2 * tile) % (2 * tile) - 2 * tile;
  const sy = ((ay % tile) + tile) % tile - tile;
  o.fillStyle = th.grass2;
  for (let y = sy; y < lh; y += tile) {
    const odd = Math.round((y - ay) / tile) & 1;
    for (let x = sx + (odd ? tile : 0); x < lw; x += 2 * tile) o.fillRect(x, y, tile, tile);
  }
  const t = REDUCED ? 0 : time;
  d.tufts.forEach(([x, y, ph], k) => {
    const [u, v] = lv.map(x, y);
    if (u < -3 || v < -6 || u > lw + 3 || v > lh + 2) return;
    if (th.deco === "snow") {
      if (k % 3) return;
      // a small pine with its trunk
      o.fillStyle = th.tuft;
      o.fillRect(u, v - 4, 1, 1);
      o.fillRect(u - 1, v - 3, 3, 1);
      o.fillRect(u - 1, v - 2, 3, 1);
      o.fillRect(u - 2, v - 1, 5, 1);
      o.fillStyle = "#6b4a2f";
      o.fillRect(u, v, 1, 1);
    } else if (th.deco === "lava") {
      if (k % 2) return;
      // cracks that glow and fade
      o.fillStyle = mix("#ff6a1a", "#ffd23f", 0.5 + 0.5 * Math.sin(t / 380 + ph));
      o.fillRect(u, v, 2, 1);
      o.fillRect(u + 1, v + 1, 2, 1);
    } else {
      const sway = Math.round(Math.sin(t / 420 + ph));
      o.fillStyle = th.tuft;
      o.fillRect(u + sway, v - 1, 1, 2);
      o.fillRect(u + 2 + sway, v, 1, 1);
    }
  });
}

// weather in screen space, snow falls and embers rise
function drawWeather(o, lw, lh, th, time) {
  if (REDUCED || (th.deco !== "snow" && th.deco !== "lava")) return;
  for (let k = 0; k < 70; k++) {
    const speed = 0.012 + (k % 4) * 0.006;
    if (th.deco === "snow") {
      const y = (k * 37 + time * speed) % (lh + 4);
      const x = (k * 53.7 + Math.sin(time / 900 + k) * 4 + lw) % lw;
      o.fillStyle = k % 5 ? "#ffffff" : "#cfe3f3";
      o.fillRect(Math.round(x), Math.round(y), 1, 1);
    } else if (k % 2) {
      const y = lh - ((k * 41 + time * speed) % (lh + 4));
      const x = (k * 67.3 + Math.sin(time / 500 + k) * 3 + lw) % lw;
      o.fillStyle = k % 3 ? "#ff8a2a" : "#ffd23f";
      o.fillRect(Math.round(x), Math.round(y), 1, 1);
    }
  }
}

function drawTrack(ctx, track, view, w, h, time = performance.now(), theme = null) {
  const th = theme || pageTheme();
  const lw = Math.max(1, Math.ceil(w / PX)), lh = Math.max(1, Math.ceil(h / PX));
  if (lowCanvas.width !== lw || lowCanvas.height !== lh) [lowCanvas.width, lowCanvas.height] = [lw, lh];
  const o = lowCanvas.getContext("2d");
  o.imageSmoothingEnabled = false;
  const lv = { s: view.s / PX, map: (x, y) => view.map(x, y).map((v) => Math.round(v / PX)) };
  const s = Math.max(lv.s, 0.35);
  const d = decor(track);
  drawGround(o, d, lv, lw, lh, s, th, time);

  // tire walls
  o.lineJoin = "round";
  o.lineWidth = Math.max(2, s * 1.6);
  o.strokeStyle = th.wall;
  [track.left_wall, track.right_wall].forEach((wall) => {
    o.beginPath();
    pathOf(o, wall, lv);
    o.stroke();
  });
  o.setLineDash([Math.max(2, 2 * s), Math.max(2, 2 * s)]);
  o.lineWidth = Math.max(1, s * 0.6);
  o.strokeStyle = th.curb;
  [track.left_wall, track.right_wall].forEach((wall) => {
    o.beginPath();
    pathOf(o, wall, lv);
    o.stroke();
  });
  o.setLineDash([]);

  // asphalt
  o.beginPath();
  pathOf(o, track.left, lv);
  pathOf(o, track.right, lv);
  o.fillStyle = th.road;
  o.fill("evenodd");

  // edges, and two color curbs on tight corners
  const ew = Math.max(1, Math.round(s * 0.8));
  [track.left, track.right].forEach((edge) => {
    o.lineWidth = ew;
    o.strokeStyle = th.edge;
    o.beginPath();
    pathOf(o, edge, lv);
    o.stroke();
    o.lineWidth = Math.max(2, Math.round(s * 1.4));
    edge.forEach((p, i) => {
      if (!d.curb[i] || i % 2) return;
      const q = edge[(i + 1) % edge.length];
      o.strokeStyle = (i >> 1) % 2 ? th.curb : th.curb2;
      o.beginPath();
      o.moveTo(...lv.map(p[0], p[1]));
      o.lineTo(...lv.map(q[0], q[1]));
      o.stroke();
    });
  });
  o.setLineDash([Math.max(1, 3 * s), Math.max(1, 3 * s)]);
  o.strokeStyle = th.center;
  o.lineWidth = Math.max(1, Math.round(s * 0.4));
  o.beginPath();
  pathOf(o, track.centerline, lv);
  o.stroke();
  o.setLineDash([]);

  // checkered start line and a waving flag beside it
  const a = track.right[0], b2 = track.left[0];
  const c0 = track.centerline[0], c1 = track.centerline[1];
  const hd = Math.atan2(c1[1] - c0[1], c1[0] - c0[0]);
  o.lineWidth = Math.max(1, Math.round(s * 1.2));
  for (let k = 0; k < 6; k++)
    for (let r = 0; r < 2; r++) {
      const t0 = k / 6, t1 = (k + 1) / 6, off = r * 1.2;
      const p0 = [a[0] + (b2[0] - a[0]) * t0 + Math.cos(hd) * off, a[1] + (b2[1] - a[1]) * t0 + Math.sin(hd) * off];
      const p1 = [a[0] + (b2[0] - a[0]) * t1 + Math.cos(hd) * off, a[1] + (b2[1] - a[1]) * t1 + Math.sin(hd) * off];
      o.strokeStyle = (k + r) % 2 ? "#f7f0e1" : "#2d2a2e";
      o.beginPath();
      o.moveTo(...lv.map(...p0));
      o.lineTo(...lv.map(...p1));
      o.stroke();
    }
  const [fx, fy] = lv.map(...track.left_wall[0]);
  o.fillStyle = "#2d2a2e";
  o.fillRect(fx, fy - 9, 1, 10);
  for (let x = 0; x < 6; x++)
    for (let y = 0; y < 4; y++) {
      const wave = REDUCED ? 0 : Math.round(Math.sin(time / 160 - x * 0.9));
      o.fillStyle = (x + y) % 2 ? "#f7f0e1" : "#2d2a2e";
      o.fillRect(fx + 1 + x, fy - 9 + y + wave, 1, 1);
    }
  drawWeather(o, lw, lh, th, time);

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(lowCanvas, 0, 0, lw, lh, 0, 0, lw * PX, lh * PX);
  ctx.restore();
}

// Top down race car facing +x, the outline is added around it. t tire, r rim, b body, d shade,
// l highlight, w wings and stripe, g cockpit, h helmet.
const CAR_ROWS = [
  "..tttttt.......ttttt..",
  "..trrrrt.......trrrt..",
  "ww..bbbbbbbbbbbbb.....",
  "wwddbbblllllbbbbbbbw..",
  "wwdbbbgghhbbwwwwbbbbww",
  "wwdbbbgghhbbwwwwbbbbww",
  "wwddddbbbbbbbbbbbbbw..",
  "ww..ddddbbbbbbbbb.....",
  "..trrrrt.......trrrt..",
  "..tttttt.......ttttt..",
];
const REAR_AXLE_PX = 6; // tire center plus the outline
const spriteCache = new Map();

function outlined(rows) {
  const w = Math.max(...rows.map((r) => r.length)) + 2, h = rows.length + 2;
  const grid = Array.from({ length: h }, (_, j) => Array.from({ length: w }, (_, i) => (rows[j - 1] || "")[i - 1] || "."));
  const filled = (i, j) => j >= 0 && j < h && i >= 0 && i < w && grid[j][i] !== "." && grid[j][i] !== "k";
  return grid.map((row, j) =>
    row.map((ch, i) => (ch === "." && (filled(i - 1, j) || filled(i + 1, j) || filled(i, j - 1) || filled(i, j + 1)) ? "k" : ch)).join("")
  );
}
const CAR_SPRITE = outlined(CAR_ROWS);

function spriteImage(rows, colors) {
  const key = rows.join("") + JSON.stringify(colors);
  if (spriteCache.has(key)) return spriteCache.get(key);
  const c = document.createElement("canvas");
  c.width = Math.max(...rows.map((r) => r.length));
  c.height = rows.length;
  const x = c.getContext("2d");
  rows.forEach((row, j) =>
    [...row].forEach((ch, i) => {
      if (!colors[ch]) return;
      x.fillStyle = colors[ch];
      x.fillRect(i, j, 1, 1);
    })
  );
  spriteCache.set(key, c);
  return c;
}

function carColors(body, outline) {
  const b = hex(body);
  return {
    k: outline || "#1b191c", b, d: mix(b, "#000000", 0.28), l: mix(b, "#ffffff", 0.45), w: "#f7f0e1",
    t: "#1b191c", r: "#77727a", g: "#2b3550", h: b === CAR_YELLOW ? "#d9483f" : CAR_YELLOW,
  };
}

// the car origin is its rear axle, the body spans 1 m behind to 3 m ahead of it
function drawSprite(ctx, x, y, yaw, view, colors, glow = null) {
  const img = spriteImage(CAR_SPRITE, colors);
  const art = Math.max(1, Math.round((4.4 * Math.max(view.s, 2.6)) / 22));
  const [px, py] = view.map(x, y);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.translate(Math.round(px), Math.round(py));
  ctx.rotate(-yaw);
  if (glow) {
    ctx.shadowColor = glow;
    ctx.shadowBlur = 12;
  }
  ctx.drawImage(img, -REAR_AXLE_PX * art, Math.round((-img.height * art) / 2), img.width * art, img.height * art);
  ctx.restore();
}

function speedColor(v) {
  return mix(css("--slow"), css("--accent"), Math.min(Math.abs(v) / D.meta.max_speed, 1));
}

function drawTrail(ctx, f, upto, view) {
  ctx.lineWidth = PX;
  ctx.lineCap = "square";
  const step = Math.max(1, Math.floor(upto / 600));
  for (let i = step; i <= upto; i += step) {
    ctx.strokeStyle = speedColor(f.v[i]);
    ctx.beginPath();
    ctx.moveTo(...view.map(f.x[i - step], f.y[i - step]));
    ctx.lineTo(...view.map(f.x[i], f.y[i]));
    ctx.stroke();
  }
}

// replay cars are always bright yellow with a glow so they stand out from any track
function drawCar(ctx, f, i, view, lidar) {
  const x = f.x[i], y = f.y[i], yaw = f.yaw[i];
  if (lidar) {
    ctx.strokeStyle = alpha(css("--accent"), 0.45);
    ctx.fillStyle = css("--accent");
    ctx.lineWidth = 1;
    D.meta.lidar_angles.forEach((ang, k) => {
      const dist = f.hits[i][k];
      const end = view.map(x + dist * Math.cos(yaw + ang), y + dist * Math.sin(yaw + ang));
      ctx.beginPath();
      ctx.moveTo(...view.map(x, y));
      ctx.lineTo(...end);
      ctx.stroke();
      if (dist < D.meta.lidar_range) ctx.fillRect(end[0] - 2, end[1] - 2, 4, 4);
    });
  }
  drawSprite(ctx, x, y, yaw, view, carColors(CAR_YELLOW, f.on_grass[i] ? "#d9483f" : "#1b191c"), "rgba(255, 210, 63, 0.9)");
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

function drawStatic(canvas, track, ep) {
  const { ctx, w, h } = sizeCanvas(canvas);
  const view = fitView(track, w, h);
  drawTrack(ctx, track, view, w, h);
  const n = ep.frames.x.length - 1;
  drawTrail(ctx, ep.frames, n, view);
  drawCar(ctx, ep.frames, n, view, false);
}

function episodeNote(ep) {
  return ep.outcome === "lap" ? `vuelta en ${num(ep.lap_time)} s` : `${OUTCOME[ep.outcome][0].toLowerCase()} al ${pct(ep.progress)}`;
}

// ---------- header art, flag and kart ----------
function drawFlag(t = 0) {
  const c = $("#flag");
  if (!c) return;
  const x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  x.fillStyle = css("--text");
  x.fillRect(0, 0, 1, 9);
  for (let i = 0; i < 12; i++)
    for (let j = 0; j < 6; j++) {
      const wave = REDUCED ? 0 : Math.round(Math.sin(t / 220 - i * 0.7) * 0.8);
      x.fillStyle = (Math.floor(i / 2) + Math.floor(j / 2)) % 2 ? css("--bg") : css("--text");
      x.fillRect(1 + i, j + 1 + wave, 1, 1);
    }
}

function drawKart(dir = 1) {
  const c = $("#kart");
  const x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  x.save();
  x.imageSmoothingEnabled = false;
  if (dir < 0) {
    x.translate(c.width, 0);
    x.scale(-1, 1);
  }
  x.drawImage(spriteImage(CAR_SPRITE, carColors(css("--accent"), css("--line"))), 0, 0);
  x.restore();
}

// ---------- hero, the user's coupe driving in its own scene ----------
// The picture is drawn untouched at 1:1. Everything around it is drawn on a grid of 7 px blocks,
// the size of one pixel of the picture, so the whole scene reads as one piece of pixel art.
const HERO = { W: 692, H: 381, B: 7, raf: 0, img: new Image(), low: document.createElement("canvas") };
HERO.img.src = "assets/hero-car.png";
HERO.img.addEventListener("load", () => state.view === "overview" && startHero());
// wheel ellipses and contact points in picture pixels, measured on the image
const HERO_WHEELS = [
  { x: 346, y: 284, rx: 24, ry: 46 },
  { x: 582, y: 252, rx: 18, ry: 40 },
];
const groundY = (u) => 49 - (u - 47) * 0.173; // ground line through both tires, in blocks

function drawHero(ts) {
  const canvas = $("#hero-car");
  if (!canvas || state.view !== "overview" || !canvas.isConnected) {
    HERO.raf = 0;
    return;
  }
  const { W, H, B, low } = HERO;
  const cols = Math.ceil(W / B), rows = Math.ceil(H / B);
  if (low.width !== cols) [low.width, low.height] = [cols, rows];
  const o = low.getContext("2d");
  const t = REDUCED ? 0 : ts / 1000;
  const ink = css("--line"), paper = css("--panel"), road = css("--road"), edge = css("--edge"), curb = css("--curb");

  // paper with a halftone that drifts slowly, the far background
  o.fillStyle = paper;
  o.fillRect(0, 0, cols, rows);
  o.fillStyle = alpha(css("--muted"), 0.18);
  const drift = Math.floor(t * 3) % 4;
  for (let j = 0; j < rows; j += 2) for (let i = ((j / 2) % 2) * 2 + drift - 4; i < cols; i += 4) if (i >= 0) o.fillRect(i, j, 1, 1);

  // speed lines streaming past, parallel to the road
  o.fillStyle = alpha(css("--muted"), 0.5);
  for (let k = 0; k < 9; k++) {
    const len = 6 + (k % 3) * 5, y0 = 4 + k * 3.4;
    const u = ((t * (55 + k * 7) + k * 23) % (cols + len + 20)) - len;
    for (let q = 0; q < len; q++) o.fillRect(Math.round(u + q), Math.round(y0 - (u + q - 47) * 0.173 - 8), 1, 1);
  }

  // the road band under the car, with curbs on its far edge and dashes on the near side
  for (let u = 0; u < cols; u++) {
    const g = groundY(u);
    o.fillStyle = road;
    o.fillRect(u, Math.round(g - 15), 1, rows);
    o.fillStyle = (Math.floor((u + t * 40) / 3) % 2) ? curb : edge;
    o.fillRect(u, Math.round(g - 16), 1, 1);
    o.fillStyle = Math.floor((u + t * 40) / 6) % 2 ? edge : road;
    o.fillRect(u, Math.round(g + 4), 1, 1);
  }

  // shadow along the ground, it shivers a little with the engine
  o.fillStyle = "rgba(0, 0, 0, 0.32)";
  const shake = Math.sin(t * 40) > 0 ? 1 : 0;
  for (let u = 3 + shake; u < 92; u++) o.fillRect(u, Math.round(groundY(u) - 1), 1, 2);

  // dust behind the rear wheel and exhaust puffs from the back, both drift to the right
  for (let p = 0; p < 7; p++) {
    const age = (t * 1.8 + p / 7) % 1;
    const size = 1 + Math.round(age * 2);
    o.fillStyle = alpha(css("--muted"), 0.7 * (1 - age));
    o.fillRect(Math.round(86 + age * 12), Math.round(groundY(86) - 2 - age * 4 - (p % 3)), size, size);
    o.fillStyle = `rgba(200, 196, 190, ${0.75 * (1 - age)})`;
    o.fillRect(Math.round(92 + age * 9), Math.round(34 - age * 5 + (p % 2)), size + 1, size);
  }

  const x = canvas.getContext("2d");
  x.imageSmoothingEnabled = false;
  x.clearRect(0, 0, W, H);
  x.drawImage(low, 0, 0, cols * B, rows * B);
  if (HERO.img.complete && HERO.img.naturalWidth) {
    const bob = Math.sin(t * 22) > 0.4 ? -2 : 0;
    x.drawImage(HERO.img, 0, bob);
    // spinning wheels, three dark blocks running round each rim
    x.fillStyle = alpha(ink, 0.55);
    HERO_WHEELS.forEach((wh) => {
      for (let k = 0; k < 3; k++) {
        const a = -t * 14 + (k * 2 * Math.PI) / 3;
        const px = wh.x + Math.cos(a) * wh.rx * 0.55, py = wh.y + bob + Math.sin(a) * wh.ry * 0.55;
        x.fillRect(Math.round(px / B) * B - 3, Math.round(py / B) * B - 3, B, B);
      }
    });
  }
  HERO.raf = requestAnimationFrame(drawHero);
}

function startHero() {
  if (!HERO.raf) HERO.raf = requestAnimationFrame(drawHero);
}

// ---------- overview, the instrument cluster ----------
const gauges = [];

function drawGauge(canvas, value, max, unit, digits, majors) {
  const dpr = window.devicePixelRatio || 1;
  const S = canvas.clientWidth || 220;
  if (canvas.width !== Math.round(S * dpr)) [canvas.width, canvas.height] = [Math.round(S * dpr), Math.round(S * dpr)];
  const x = canvas.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  x.clearRect(0, 0, S, S);
  const c = S / 2, R = S * 0.44;
  const a0 = Math.PI * 0.75, sweep = Math.PI * 1.5;
  const ang = (v) => a0 + sweep * clamp(v / max, 0, 1.04);
  // face
  x.beginPath();
  x.arc(c, c, R + 6, 0, 2 * Math.PI);
  x.fillStyle = "#151317";
  x.fill();
  x.lineWidth = 3;
  x.strokeStyle = "#3a353b";
  x.stroke();
  // red zone at the top of the scale
  x.beginPath();
  x.arc(c, c, R - 6, ang(max * 0.85), ang(max));
  x.lineWidth = 8;
  x.strokeStyle = "#d9483f";
  x.stroke();
  // ticks and numbers
  for (let i = 0; i <= majors * 5; i++) {
    const a = a0 + (sweep * i) / (majors * 5), major = i % 5 === 0;
    const r1 = R - (major ? 16 : 9);
    x.beginPath();
    x.moveTo(c + Math.cos(a) * r1, c + Math.sin(a) * r1);
    x.lineTo(c + Math.cos(a) * (R - 2), c + Math.sin(a) * (R - 2));
    x.lineWidth = major ? 3 : 1.5;
    x.strokeStyle = major ? "#efe6d2" : "#8a817d";
    x.stroke();
    if (major) {
      x.fillStyle = "#d8ccb6";
      x.font = `600 ${Math.round(S * 0.075)}px "Barlow Condensed", sans-serif`;
      x.textAlign = "center";
      x.textBaseline = "middle";
      x.fillText(String(Math.round((max * i) / (majors * 5))), c + Math.cos(a) * (R - 30), c + Math.sin(a) * (R - 30));
    }
  }
  // digital readout and unit
  x.fillStyle = "#efe6d2";
  x.font = `700 ${Math.round(S * 0.17)}px "Barlow Condensed", sans-serif`;
  x.fillText(value.toFixed(digits).replace(".", ","), c, c + R * 0.5);
  x.fillStyle = "#a69a8e";
  x.font = `600 ${Math.round(S * 0.07)}px "Barlow Condensed", sans-serif`;
  x.fillText(unit, c, c + R * 0.72);
  // needle with a soft glow and the hub
  const a = ang(value);
  x.save();
  x.shadowColor = "rgba(255, 90, 70, 0.7)";
  x.shadowBlur = 10;
  x.beginPath();
  x.moveTo(c - Math.cos(a) * 12, c - Math.sin(a) * 12);
  x.lineTo(c + Math.cos(a) * (R - 12), c + Math.sin(a) * (R - 12));
  x.lineWidth = 4;
  x.lineCap = "round";
  x.strokeStyle = "#ff5a46";
  x.stroke();
  x.restore();
  x.beginPath();
  x.arc(c, c, 9, 0, 2 * Math.PI);
  x.fillStyle = "#2d2a2e";
  x.fill();
  x.lineWidth = 2;
  x.strokeStyle = "#8a817d";
  x.stroke();
}

// the needle sweeps to the top like a car starting up, then springs back to its value
function animateGauge(canvas, target, max, unit, digits, majors) {
  const start = performance.now();
  const frame = (ts) => {
    if (!canvas.isConnected) return;
    const t = (ts - start) / 1000;
    let v;
    if (REDUCED) v = target;
    else if (t < 0.45) v = max * (1 - (1 - t / 0.45) ** 3);
    else {
      const k = t - 0.45;
      v = target + (max - target) * Math.exp(-5 * k) * Math.cos(9 * k);
    }
    drawGauge(canvas, v, max, unit, digits, majors);
    if (!REDUCED && t < 2) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function renderCluster() {
  const key = mainPolicy();
  const v = D.validation[key] || {};
  const pp = D.validation.pure_pursuit || {};
  const maxKmh = Math.ceil((D.meta.max_speed * 3.6) / 10) * 10;
  $("#cluster").innerHTML = `
    <div class="gauge"><canvas id="g-speed"></canvas><span>velocidad media</span></div>
    <div class="lcd">
      <div class="cells">
        <div class="cell"><small>Tiempo de vuelta</small><b>${digit(v.mean_lap_time)}<span class="unit">s</span></b></div>
        <div class="cell"><small>Pure pursuit</small><b>${digit(pp.mean_lap_time)}<span class="unit">s</span></b></div>
        <div class="cell wide"><small>Progreso conseguido de media por pista</small><b>${digit(100 * (v.mean_progress || 0), 0)}<span class="unit">%</span></b></div>
      </div>
      <div class="foot">${esc(policyName(key))}, ${D.meta.eval_tracks} pistas de validación, nivel ${levelName(D.meta.difficulty).toLowerCase()}</div>
    </div>
    <div class="gauge"><canvas id="g-laps"></canvas><span>vueltas completadas</span></div>`;
  animateGauge($("#g-speed"), (v.mean_speed || 0) * 3.6, maxKmh, "km/h", 0, maxKmh / 10);
  animateGauge($("#g-laps"), 100 * (v.lap_success_rate || 0), 100, "%", 0, 5);
}

// one digital counter per difficulty with the share of validation tracks the model laps
function renderCounters() {
  const key = mainPolicy();
  const levels = D.levels || [{ difficulty: D.meta.difficulty, validation: D.validation }];
  $("#counters").innerHTML = levels
    .map((l) => {
      const m = l.validation[key] || {}, pp = l.validation.pure_pursuit || {};
      return `<div class="counter"><h4>${levelName(l.difficulty)}<span>dificultad ${num(l.difficulty)}</span></h4>
        <div class="lcd"><div class="cells">
          <div class="cell wide"><small>Vueltas completadas</small><b>${digit(100 * (m.lap_success_rate || 0), 0)}<span class="unit">%</span></b></div>
        </div><div class="foot">Pure pursuit ${pct(pp.lap_success_rate)} · progreso medio ${pct(m.mean_progress)}</div></div></div>`;
    })
    .join("");
}

// ---------- training ----------
const TRAIN_METRICS = [
  {
    source: "val_curve",
    key: "lap_success_rate",
    title: "Vueltas en pistas no vistas",
    info: {
      what: "Qué parte de las pistas de validación termina el coche. Son las semillas de pista desde la 1000, que nunca se usan para entrenar. Se mide cada cierto número de pasos, con el modelo conduciendo sin azar.",
      good: "El objetivo mínimo era un 70 %. Pure pursuit, que ve el mapa, las termina todas.",
      read: "Tiene que subir y quedarse arriba. Las mesetas son normales y PPO suele dar un salto después. Con curriculum, cada bajada coincide con una subida de dificultad.",
    },
  },
  {
    source: "val_curve",
    key: "mean_progress",
    title: "Parte de la vuelta recorrida",
    info: {
      what: "Hasta dónde llega el coche, de media, antes de terminar o fallar. 1 es la vuelta entera.",
      good: "Por encima de 0,9 solo falla cerca del final, si es que falla.",
      read: "Se mueve antes que la de vueltas, así que enseña que el coche aprende aunque todavía no termine ninguna.",
    },
  },
  {
    source: "progress",
    key: "ep_rew_mean",
    title: "Recompensa por intento",
    info: {
      what: "Los puntos que suma un intento entero en las pistas de entrenamiento, media de los últimos 100. Avanzar y completar la vuelta suman. El tiempo, los volantazos, pisar hierba, ir a paso de tortuga o pasarse de velocidad y salirse restan.",
      good: "Una vuelta limpia vale unos 10 puntos y estrellarse pronto, unos −5.",
      read: "Si sube, el coche aprende lo que le pide la recompensa. Solo tiene sentido comparar runs que usan la misma recompensa.",
    },
  },
  {
    source: "progress",
    key: "ep_len_mean",
    title: "Duración del intento en pasos",
    info: {
      what: "Cuántos pasos dura cada intento. Un paso son 0,05 s, así que 20 pasos son un segundo.",
      good: "Una vuelta en dificultad 0 son unos 480 pasos, 24 s.",
      read: "Se lee junto a la recompensa. Más corto y con más recompensa son vueltas más rápidas. Más corto y con menos, choques más tempranos.",
    },
  },
  {
    source: "progress",
    key: "entropy_loss",
    title: "Exploración (entropy loss)",
    info: {
      what: "Cuánto azar queda en las decisiones del coche mientras entrena. Es la entropía con el signo cambiado.",
      good: "Que suba despacio. Primero prueba cosas y luego se decide.",
      read: "Más alto es menos azar. Si sube de golpe al principio, el coche ha dejado de probar cosas demasiado pronto.",
    },
  },
  {
    source: "progress",
    key: "approx_kl",
    title: "Cambio por actualización (KL)",
    info: {
      what: "Cuánto cambia la forma de conducir en cada actualización de PPO.",
      good: "Entre 0,005 y 0,02, más o menos.",
      read: "Los picos por encima de 0,05 son cambios demasiado bruscos. Casi 0 durante mucho tiempo es que apenas aprende.",
    },
  },
  {
    source: "progress",
    key: "value_loss",
    title: "Error del crítico (value loss)",
    info: {
      what: "El error del crítico, la parte de la red que intenta adivinar cuántos puntos le quedan por ganar al coche.",
      good: "Que baje y se quede abajo.",
      read: "Los picos son normales cuando cambia la forma de conducir, por ejemplo con las primeras vueltas completas.",
    },
  },
  {
    source: "progress",
    key: "explained_variance",
    title: "Acierto del crítico (explained variance)",
    info: {
      what: "Qué parte de los puntos reales acierta el crítico. 1 es acertar siempre y 0 es no hacerlo mejor que decir la media.",
      good: "Más de 0,5 mientras aprende a no salirse. Cuando casi todo son vueltas, entre 0,2 y 0,3 es normal aquí.",
      read: "El crítico no ve dónde está la meta, así que no puede saber cuándo llega el premio por vuelta.",
    },
  },
];

function runGroups() {
  return FAMILIES.map(([id, name]) => ({ id, name, runs: D.runs.filter((r) => familyOf(r.name) === id) })).filter((g) => g.runs.length);
}

function setRuns(names, on) {
  names.forEach((n) => (on ? state.hidden.delete(n) : state.hidden.add(n)));
  renderTraining();
}

function renderTraining() {
  const q = state.search.trim().toLowerCase();
  const match = (r) => !q || `${r.name} ${r.label || ""}`.toLowerCase().includes(q);
  const groups = runGroups();
  $("#run-groups").innerHTML = groups
    .map((g) => {
      const on = g.runs.filter((r) => !state.hidden.has(r.name)).length;
      const chips = g.runs
        .filter(match)
        .map((r) => `<button class="chip ${state.hidden.has(r.name) ? "off" : ""}" data-run="${r.name}" title="${r.name}"><span class="dot" style="background:${runColor(r.name)}"></span>${esc(r.label || r.name)}</button>`)
        .join("");
      return `<div class="card run-group" data-family="${g.id}"><header><div><h4>${g.name}</h4><span>${on} de ${g.runs.length} a la vista</span></div>
        <div class="acts"><button class="link-btn" data-act="all">Todas</button><button class="link-btn" data-act="none">Ninguna</button></div></header>
        <div class="chips">${chips || '<span class="muted">nada coincide con la búsqueda</span>'}</div></div>`;
    })
    .join("");
  $$("#run-groups .chip").forEach((c) =>
    c.addEventListener("click", () => setRuns([c.dataset.run], state.hidden.has(c.dataset.run)))
  );
  $$("#run-groups .run-group").forEach((card) => {
    const g = groups.find((x) => x.id === card.dataset.family);
    $$("[data-act]", card).forEach((b) => b.addEventListener("click", () => setRuns(g.runs.map((r) => r.name), b.dataset.act === "all")));
  });
  const shown = D.runs.filter((r) => !state.hidden.has(r.name));
  $("#run-count").textContent = `${shown.length} de ${D.runs.length} runs a la vista`;

  const el = $("#train-charts");
  const built = TRAIN_METRICS.every((_, i) => state.charts[`train-${i}`] && state.charts[`train-${i}`].canvas.isConnected);
  if (built) {
    // only the data changes, the charts stay and morph to the new selection
    TRAIN_METRICS.forEach(({ source, key }, i) => {
      const chart = state.charts[`train-${i}`];
      chart.data.datasets = groupedSeries(shown, source, key);
      chart.update();
    });
    return;
  }
  destroyChartsIn(el);
  TRAIN_METRICS.forEach(({ source, key, title, info }, i) => {
    const canvas = chartCard(el, `train-${i}`, title, info);
    makeChart(canvas, { type: "line", data: { datasets: groupedSeries(shown, source, key) }, options: lineOptions("pasos de entrenamiento") });
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
  return `<table><thead><tr><th>Piloto</th>${keys.map((k) => `<th>${cap(METRIC_LABELS[k])}</th>`).join("")}</tr></thead><tbody>${Object.entries(rows)
    .map(
      ([name, m]) =>
        `<tr><td><span class="dot" style="background:${policyColor(name)}"></span> ${esc(policyName(name))}</td>${keys
          .map((k) => `<td>${fmt(k, m[k])}${m[k + "_std"] != null && m.n_models > 1 ? ` ± ${fmt(k, m[k + "_std"])}` : ""}</td>`)
          .join("")}</tr>`
    )
    .join("")}</tbody></table>`;
}

function renderValidation() {
  levelTabs($("#val-level"));
  const { difficulty, validation: V } = level();
  const n = D.meta.eval_tracks;
  $("#val-note").textContent = `Cada piloto corre una vez las mismas ${n} pistas de validación (semillas de pista 1000 a ${999 + n}) en el nivel ${levelName(difficulty).toLowerCase()} (dificultad ${num(difficulty)}), sin azar en sus decisiones.`;
  $("#val-table").innerHTML = metricsTable(V);
  const keys = Object.keys(V);
  const el = $("#val-charts");
  destroyChartsIn(el);
  const pp = V.pure_pursuit || {};
  const K = D.meta.car;
  const bar = (id, title, values, max, info) =>
    makeChart(chartCard(el, id, title, info), {
      type: "bar",
      data: { labels: keys.map(policyName), datasets: [{ label: title, data: values, backgroundColor: keys.map((k) => alpha(policyColor(k), 0.85)) }] },
      options: { interaction: { mode: "index", intersect: false }, plugins: { legend: { display: false } }, scales: { y: { min: 0, max } } },
    });
  bar("val-lap", "Vueltas completadas", keys.map((k) => V[k].lap_success_rate), 1, {
    what: "Qué parte de las pistas termina cada piloto. Todos corren las mismas pistas en la misma dificultad.",
    good: `Más es mejor. Pure pursuit llega al ${pct(pp.lap_success_rate)} y random debería quedarse cerca de 0.`,
    read: "Es el número principal. Ganar a random demuestra que ha aprendido, e igualar a pure pursuit es conducir tan bien como un piloto que ve el mapa.",
  });
  bar("val-time", "Tiempo de vuelta, solo vueltas terminadas (s)", keys.map((k) => V[k].mean_lap_time), undefined, {
    what: "Tiempo medio de las vueltas terminadas. Los intentos fallidos no cuentan.",
    good: `Menos es mejor. Pure pursuit tarda ${num(pp.mean_lap_time)} s.`,
    read: "Mírala junto a las vueltas completadas. Quien termina pocas puede parecer rápido porque solo cuentan sus pistas fáciles.",
  });
  bar("val-speed", "Velocidad media (km/h)", keys.map((k) => V[k].mean_speed * 3.6), undefined, {
    what: "Velocidad hacia delante en todos los pasos de todos los intentos, choques incluidos.",
    good: `Más es mejor si sigue terminando vueltas. Pure pursuit va a ${num((pp.mean_speed || 0) * 3.6, 0)} km/h y el coche no pasa de ${num(D.meta.max_speed * 3.6, 0)} km/h.`,
    read: "Mucha velocidad con muchos choques es conducir a lo loco. Muy poca es que duda o se queda parado.",
  });
  const outcomes = [
    ["lap", (m) => m.lap_success_rate],
    ["off_road", (m) => m.off_track_rate],
    ["stuck", (m) => m.stuck_rate],
    ["timeout", timeoutRate],
  ];
  makeChart(chartCard(el, "val-out", "Cómo acaba cada intento", {
    what: `Vuelta completa, fuera de pista (toca el muro o pasa más de ${num(K.GRASS_TIME, 0)} s en la hierba), atasco (avanza menos de 2 m en 10 s) o tiempo agotado (${num(K.MAX_TIME, 0)} s).`,
    good: "Todo vuelta. Alguna salida es aceptable mientras aprende, pero los atascos nunca deberían ser muchos.",
    read: "Salirse suele ser ir demasiado rápido o girar tarde. Atascarse es rendirse y quedarse quieto.",
  }), {
    type: "bar",
    data: {
      labels: keys.map(policyName),
      datasets: outcomes.map(([o, f]) => ({ label: OUTCOME[o][0], data: keys.map((k) => f(V[k])), backgroundColor: outcomeColor(o) })),
    },
    options: { interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } }, scales: { x: { stacked: true }, y: { stacked: true, min: 0, max: 1 } } },
  });
}

// ---------- statistics ----------
const fmtCi = (ci, f = pct) => (ci && ci[0] != null ? `${f(ci[0])} a ${f(ci[1])}` : "-");
const fmtP = (p) => (p == null ? "-" : p < 0.0001 ? "< 0,0001" : num(p, 4));

function testDetail(t) {
  let m = t.detail.match(/^(\d+) vs (\d+) discordant$/);
  if (m) return `${m[1]} pistas solo las acaba el primero y ${m[2]} solo el segundo`;
  m = t.detail.match(/^n (\d+), median diff ([+-]?[\d.]+)$/);
  if (m) return `${m[1]} pistas, diferencia mediana ${num(Number(m[2]), 2)}`;
  return esc(t.detail);
}

function intervalChart(el, id, title, S, key, info) {
  const keys = Object.keys(S.intervals);
  const colors = keys.map(policyColor);
  makeChart(chartCard(el, id, title, info), {
    data: {
      labels: keys.map(policyName),
      datasets: [
        { type: "bar", label: "intervalo al 95 %", data: keys.map((k) => S.intervals[k][key + "_ci"]), backgroundColor: colors.map((c) => alpha(c, 0.3)), borderColor: colors, borderSkipped: false, barPercentage: 0.4 },
        { type: "line", label: "estimación", data: keys.map((k) => S.intervals[k][key]), showLine: false, pointRadius: 6, pointHoverRadius: 8, pointBackgroundColor: colors, pointBorderColor: css("--line"), pointBorderWidth: 2 },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (c) => (c.dataset.type === "bar" && Array.isArray(c.raw) ? `intervalo: ${pct(c.raw[0])} a ${pct(c.raw[1])}` : `estimación: ${pct(c.raw)}`),
          },
        },
      },
      scales: { y: { min: 0, max: 1, ticks: { callback: (v) => pct(v) } } },
    },
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
    body.innerHTML = `<div class="card empty">${icon("alert")}<p>Este data.js es anterior a la estadística. Vuelve a generarlo con <code>make dashboard</code>.</p></div>`;
    return;
  }
  $("#stat-note").textContent = `Nivel ${levelName(L.difficulty).toLowerCase()} (dificultad ${num(L.difficulty)}). Todos los pilotos corrieron las mismas ${D.meta.eval_tracks} pistas de validación, así que cada comparación se hace pista a pista.`;
  destroyChartsIn(body);
  body.innerHTML = `<div class="card seed-explain">
      <div><h4>Semilla de <em>pista</em></h4><p>Decide qué circuito se genera. La misma semilla de pista da siempre el mismo trazado, así que todos los pilotos se miden en las mismas curvas.</p></div>
      <div><h4>Semilla de <em>entrenamiento</em></h4><p>Decide el azar de un entrenamiento, como los pesos con los que arranca la red o el orden de las pistas. Cada versión del modelo se entrenó 3 veces, con las semillas 0, 1 y 2.</p></div>
    </div><div class="grid-2" id="stat-charts"></div>`;
  const charts = $("#stat-charts");
  intervalChart(charts, "stat-lap", "Vueltas con intervalo al 95 %", S, "lap_rate", {
    what: "El punto es la parte de pistas completadas y la barra, el rango donde probablemente esté el valor real.",
  });
  intervalChart(charts, "stat-progress", "Vuelta recorrida con intervalo al 95 %", S, "progress", {
    what: "Qué parte de la vuelta recorre de media cada piloto antes de terminar o fallar, donde 1 es la vuelta entera. Para la barra se repite el cálculo miles de veces eligiendo pistas al azar de entre las 20 (bootstrap), y se queda con el 95 % central de los resultados.",
    good: "Cerca de 1 y con la barra corta.",
    read: "Distingue a un piloto que casi acaba de otro que no arranca, aunque ninguno de los dos complete vueltas.",
  });

  const rows = Object.entries(S.intervals).map(([k, s]) => {
    const seeds = s.lap_rate_per_seed ? s.lap_rate_per_seed.map(pct).join(", ") : "-";
    return `<tr><td><span class="dot" style="background:${policyColor(k)}"></span> ${esc(policyName(k))}</td><td>${s.n_models}</td><td>${s.n_tracks}</td><td>${pct(s.lap_rate)}</td><td>${fmtCi(s.lap_rate_ci)}</td><td>${seeds}</td><td>${fmtCi(s.lap_rate_seed_ci)}</td><td>${pct(s.progress)}</td><td>${fmtCi(s.progress_ci)}</td></tr>`;
  });
  const intervals = `<table><thead><tr><th>Piloto</th><th>Entrenamientos</th><th>Pistas</th><th>Vueltas</th><th>Intervalo al 95 %</th><th>Por semilla de entrenamiento</th><th>Intervalo t de Student</th><th>Vuelta recorrida</th><th>Intervalo al 95 %</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;

  const tests = S.tests.map((t) => {
    const pa = t.a.startsWith("pooled:"), pb = t.b.startsWith("pooled:");
    const which = !t.models ? "" : pa && pb ? `, semilla de entrenamiento ${t.models[0]} de cada uno` : pa ? `, semilla de entrenamiento ${t.models[0]}` : pb ? `, semilla de entrenamiento ${t.models[1]}` : "";
    const verdict = t.p_holm == null ? `<span class="sig">sin pistas que comparar</span>` : t.p_holm < 0.05 ? `<span class="sig yes">distintos</span>` : `<span class="sig">sin pruebas de diferencia</span>`;
    return `<tr><td class="t">${esc(policyName(t.a))} contra ${esc(policyName(t.b))}</td><td class="t">${TEST_NAMES[t.test] || esc(t.test)}${which}</td><td class="t">${testDetail(t)}</td><td>${fmtP(t.p)}</td><td>${fmtP(t.p_holm)}</td><td>${verdict}</td></tr>`;
  });
  const testTable = `<table><thead><tr><th style="text-align:left">Pareja</th><th style="text-align:left">Test</th><th style="text-align:left">Detalle</th><th>p</th><th>p con Holm</th><th>Resultado</th></tr></thead><tbody>${tests.join("")}</tbody></table>`;
  const friedman = S.friedman == null ? "" : `<p class="test-note">Friedman compara a todos los pilotos a la vez con la vuelta recorrida en cada pista y da p = ${fmtP(S.friedman)}. ${S.friedman < 0.05 ? "Al menos uno rinde distinto al resto." : "No hay pruebas de que alguno rinda distinto al resto."}</p>`;
  const note = S.note ? `<p class="test-note">Algunas versiones tienen distinto número de semillas de entrenamiento y no se pueden emparejar, así que aquí no hay tests.</p>` : "";

  body.insertAdjacentHTML("beforeend", infoCard("Intervalos por piloto", {
    what: "Cada fila es un piloto. Las filas «todas las semillas» juntan los 3 entrenamientos de una misma versión del modelo (sus parámetros y pesos) y muestran las vueltas de cada uno.",
    good: "Que los valores por semilla de entrenamiento se parezcan. Así el resultado depende del método y no de un entrenamiento con suerte.",
    read: "El intervalo t de Student responde a una pregunta: si se entrenara esta versión otra vez desde cero, ¿entre qué valores caería su porcentaje de vueltas? Sale de la media y la dispersión de los 3 entrenamientos. Como 3 datos son muy pocos, la t de Student ensancha el intervalo para no prometer más precisión de la que hay.",
  }, intervals));
  body.insertAdjacentHTML("beforeend", infoCard("Tests pareados", {
    what: "Son pareados porque los dos pilotos se comparan sobre las mismas pistas, una a una. McNemar compara vueltas y solo cuenta las pistas en las que uno acabó y el otro no. Wilcoxon compara la vuelta recorrida o el tiempo pista a pista, y mira si las diferencias se inclinan claramente hacia un lado. Como se hacen muchas comparaciones a la vez, la corrección de Holm sube cada p para no dar por buena una diferencia que solo es casualidad.",
    good: "«Distintos» quiere decir que la p con Holm es menor que 0,05, es decir, que es muy poco probable que una diferencia así salga por azar.",
    read: "«Sin pruebas de diferencia» no demuestra que sean iguales. Solo dice que con estas pistas no se nota la diferencia.",
  }, testTable + friedman + note));
}

function renderTest() {
  const rows = D.test;
  const n = Object.values(rows)[0];
  const tracks = n && n.n_models ? Math.round(n.n_episodes / n.n_models) : 200;
  $("#test-body").innerHTML = Object.keys(rows).length
    ? `<div class="card"><div class="chart-title">Resultados en las pistas de test${infoButton({
        what: "En las filas de PPO cada cifra es la media de los 3 modelos entrenados, uno por semilla de entrenamiento. El ± es la desviación típica entre esos 3 modelos, o sea, cuánto cambia el resultado de un entrenamiento a otro. Si vale 0, los 3 sacaron exactamente lo mismo.",
      })}</div><div class="table-wrap">${metricsTable(rows)}</div></div>
      <p class="test-note">Los resultados de test salen de evaluar el modelo final, con sus parámetros y pesos ya entrenados y congelados, sobre ${tracks} pistas generadas al azar (semillas de pista 10000 a ${9999 + tracks}). Esas pistas no se usaron ni para entrenar ni para elegir el modelo, y se corrieron una sola vez.</p>`
    : `<div class="card empty">${icon("lock")}<p>Las pistas de test (semillas de pista desde la 10000) se corren una sola vez con los modelos finales.<br>Hasta entonces no se publica nada y nunca se ajusta nada con ellas.</p></div>`;
}

// ---------- steering wheel picker ----------
// A pixel steering wheel peeks out from the left and turns one notch per item. Arrows, the mouse
// wheel and the up and down keys move it. items is a list of [value, label].
function drawWheelArt(canvas) {
  const N = 96, C = 47.5;
  canvas.width = canvas.height = N;
  const x = canvas.getContext("2d");
  const img = x.createImageData(N, N);
  const put = (i, j, color) => {
    const n = parseInt(color.slice(1), 16), k = (j * N + i) * 4;
    [img.data[k], img.data[k + 1], img.data[k + 2], img.data[k + 3]] = [n >> 16, (n >> 8) & 255, n & 255, 255];
  };
  const spokes = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const dx = i - C, dy = j - C, r = Math.hypot(dx, dy), a = Math.atan2(dy, dx);
      if (r > 47) continue;
      if (r > 45.5 || (r > 36 && r < 37.2)) put(i, j, "#0d0c0e");
      else if (r >= 37.2) {
        const top = Math.abs(Math.atan2(Math.sin(a + Math.PI / 2), Math.cos(a + Math.PI / 2))) < 0.13;
        const grip = Math.sin(a * 14) > 0.55 && r > 41;
        put(i, j, top ? "#d9483f" : grip ? "#1b191c" : r < 39 ? "#4a454c" : "#2b282d");
      } else if (r < 11) {
        if (r > 9.5) put(i, j, "#0d0c0e");
        else put(i, j, (Math.floor((i + 64) / 3) + Math.floor((j + 64) / 3)) % 2 ? "#f7f0e1" : "#2d2a2e");
      } else {
        for (const s of spokes) {
          const da = Math.atan2(Math.sin(a - s), Math.cos(a - s));
          const off = r * Math.sin(da);
          if (Math.cos(da) > 0 && Math.abs(off) < 4.2) {
            put(i, j, Math.abs(off) > 3.2 ? "#0d0c0e" : off > 0 ? "#6f6865" : "#9a928e");
            break;
          }
        }
      }
    }
  x.putImageData(img, 0, 0);
}

function wheelPicker(el, items, index, onPick) {
  const n = items.length;
  let cur = clamp(index, 0, Math.max(n - 1, 0));
  const sig = items.map((i) => i[0]).join("|");
  const fresh = el._sig !== sig;
  el._sig = sig;
  el._angle = el._angle || 0;
  el.innerHTML = `<button class="wheel-arrow" data-d="-1" aria-label="Anterior">▲</button>
    <div class="wheel-body" tabindex="0" role="listbox" aria-label="Elegir">
      <canvas class="wheel-art" aria-hidden="true"></canvas><div class="wheel-items"></div></div>
    <button class="wheel-arrow" data-d="1" aria-label="Siguiente">▼</button>`;
  const art = $(".wheel-art", el), list = $(".wheel-items", el), body = $(".wheel-body", el);
  drawWheelArt(art);
  const paint = (dir = 0) => {
    const at = (k) => items[(cur + k + n) % n];
    const slots = n > 1 ? [["prev", -1], ["cur", 0], ["next", 1]] : [["cur", 0]];
    list.innerHTML = slots
      .map(([cls, k]) => `<button class="wheel-item ${cls}" data-k="${k}" role="option" aria-selected="${k === 0}" title="${esc(at(k)[1])}">${esc(at(k)[1])}</button>`)
      .join("");
    $$(".wheel-item", list).forEach((b) => b.addEventListener("click", () => move(Number(b.dataset.k))));
    art.style.transform = `rotate(${el._angle}deg)`;
    if (dir && !REDUCED)
      list.animate([{ transform: `translateY(${dir * 34}px) rotate(${dir * 6}deg)`, opacity: 0.3 }, { transform: "none", opacity: 1 }], { duration: 380, easing: "cubic-bezier(.34,1.56,.64,1)" });
  };
  const move = (d) => {
    if (n < 2 || !d) return;
    cur = (cur + d + n) % n;
    el._angle += d * 40;
    paint(d);
    onPick(items[cur][0]);
  };
  $$(".wheel-arrow", el).forEach((b) => b.addEventListener("click", () => move(Number(b.dataset.d))));
  let lastWheel = 0;
  body.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (performance.now() - lastWheel < 180) return;
      lastWheel = performance.now();
      move(e.deltaY > 0 ? 1 : -1);
    },
    { passive: false }
  );
  body.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      move(e.key === "ArrowDown" ? 1 : -1);
    }
  });
  paint();
  // a new list rolls the wheel in from the left
  if (fresh && !REDUCED)
    art.animate([{ transform: `translateX(-140px) rotate(${el._angle - 220}deg)` }, { transform: `rotate(${el._angle}deg)` }], { duration: 750, easing: "cubic-bezier(.34,1.3,.64,1)" });
}

// ---------- simulation ----------
const simState = { mode: "policy", key: null, seed: null, playing: false, t: 0, speed: 1, tiles: [], maxT: 0, lastTs: 0 };

function simItems() {
  return simState.mode === "policy"
    ? Object.keys(D.replays).map((k) => [k, policyName(k)])
    : Object.keys(D.tracks).map((s) => [s, `Pista ${s}`]);
}

function renderSimulation() {
  if (!D.replays[simState.key]) simState.key = mainPolicy();
  if (!D.tracks[simState.seed]) simState.seed = Object.keys(D.tracks)[0];
  tabs($("#sim-mode"), [["policy", "Un piloto, varias pistas"], ["track", "Una pista, todos los pilotos"]], simState.mode, (m) => {
    simState.mode = m;
    renderSimulation();
  });
  const items = simItems();
  const value = simState.mode === "policy" ? simState.key : String(simState.seed);
  wheelPicker($("#sim-wheel"), items, items.findIndex((i) => String(i[0]) === value), (v) => {
    if (simState.mode === "policy") simState.key = v;
    else simState.seed = v;
    renderSimTiles();
  });
  $("#speed").innerHTML = [1, 2, 4, 8].map((s) => `<button data-speed="${s}" class="${s === simState.speed ? "active" : ""}">${s}x</button>`).join("");
  $$("#speed button").forEach((b) =>
    b.addEventListener("click", () => {
      simState.speed = Number(b.dataset.speed);
      $$("#speed button").forEach((x) => x.classList.toggle("active", x === b));
    })
  );
  renderSimTiles();
}

function renderSimTiles() {
  const tiles = simState.mode === "policy"
    ? D.replays[simState.key].map((ep) => ({ ep, key: simState.key, title: `Pista ${ep.seed}` }))
    : Object.entries(D.replays).map(([key, eps]) => ({ ep: eps.find((e) => String(e.seed) === String(simState.seed)), key, title: policyName(key) }));
  const grid = $("#sim-grid");
  grid.innerHTML = "";
  simState.tiles = tiles.filter((t) => t.ep).map((t, i) => {
    const dot = simState.mode === "track" ? `<span class="dot" style="background:${policyColor(t.key)}"></span> ` : "";
    grid.insertAdjacentHTML(
      "beforeend",
      `<div class="card tile"><canvas id="sim-${i}"></canvas><div class="tile-body">
        <span class="tile-title">${dot}${esc(t.title)}</span>
        <div class="tile-foot"><span id="sim-info-${i}"></span><span>${cap(episodeNote(t.ep))}</span></div></div></div>`
    );
    return { ...t, canvas: $(`#sim-${i}`), i };
  });
  simState.tiles.forEach((tile) => Object.assign(tile, sizeCanvas(tile.canvas)));
  simState.maxT = Math.max(0, ...simState.tiles.map((t) => t.ep.frames.x.length * t.ep.dt));
  simState.t = Math.min(simState.t, simState.maxT);
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
    const view = follow ? followView(f.x[i], f.y[i], w, h, 6) : fitView(track, w, h);
    drawTrack(ctx, track, view, w, h);
    if (trail) drawTrail(ctx, f, i, view);
    drawCar(ctx, f, i, view, lidarOn);
    const speed = Math.hypot(f.v[i], f.v_lat[i]) * 3.6;
    $(`#sim-info-${tile.i}`).textContent = `${speed.toFixed(0)} km/h, ${pct(Math.max(0, f.progress[i]))} de la vuelta`;
  });
  $("#clock").textContent = `${num(simState.t)} s`;
  $("#scrub").value = simState.maxT ? Math.round((1000 * simState.t) / simState.maxT) : 0;
}

function setPlaying(on) {
  simState.playing = on;
  if (on && simState.t >= simState.maxT) simState.t = 0;
  $("#play").innerHTML = `${icon(on ? "pause" : "play")}${on ? "Pausa" : "Reproducir"}`;
  simState.lastTs = performance.now();
  if (on) requestAnimationFrame(tick);
}

function tick(ts) {
  if (!simState.playing || state.view !== "simulation") return;
  const dt = Math.max(0, ts - simState.lastTs) / 1000;
  simState.t = Math.min(simState.t + dt * simState.speed, simState.maxT);
  simState.lastTs = ts;
  drawSim();
  if (simState.t >= simState.maxT) setPlaying(false);
  else requestAnimationFrame(tick);
}

// ---------- physics ----------
const physState = { mode: "policy", key: null, seed: null };

function renderPhysics() {
  if (!D.replays[physState.key]) physState.key = mainPolicy();
  if (!D.tracks[physState.seed]) physState.seed = Object.keys(D.tracks)[0];
  tabs($("#phys-mode"), [["policy", "Elegir piloto"], ["track", "Elegir pista"]], physState.mode, (m) => {
    physState.mode = m;
    renderPhysics();
  });
  const items = physState.mode === "policy" ? Object.keys(D.replays).map((k) => [k, policyName(k)]) : Object.keys(D.tracks).map((s) => [s, `Pista ${s}`]);
  const value = physState.mode === "policy" ? physState.key : String(physState.seed);
  wheelPicker($("#phys-wheel"), items, items.findIndex((i) => String(i[0]) === value), (v) => {
    if (physState.mode === "policy") physState.key = v;
    else physState.seed = v;
    renderPhysicsBody();
  });
  renderPhysicsBody();
}

function renderPhysicsBody() {
  const key = physState.key, seed = String(physState.seed);
  const ep = D.replays[key].find((e) => String(e.seed) === seed);
  const track = D.tracks[seed];
  const el = $("#phys-charts");
  destroyChartsIn(el);
  if (!ep || !track) {
    el.innerHTML = `<div class="card empty">No hay repetición de ${esc(policyName(key))} en la pista ${esc(seed)}.</div>`;
    return;
  }
  const f = ep.frames;
  const t = f.x.map((_, i) => +(i * ep.dt).toFixed(2));
  const color = policyColor(key);
  const series = (arr, label, c = color, extra = {}) => ({ label, data: arr.map((y, i) => ({ x: t[i], y })), borderColor: c, backgroundColor: c, borderWidth: 2, ...extra });
  const flat = (y, label, c) => ({ label, data: [{ x: t[0], y }, { x: t[t.length - 1], y }], borderColor: c, borderDash: [4, 4], borderWidth: 1, pointHoverRadius: 0 });
  const fast = (extra = {}) => lineOptions("tiempo [s]", { fast: true, ...extra });
  const half = track.width / 2, wall = half + track.width;

  makeChart(chartCard(el, "ph-speed", "Velocidad [km/h]"), {
    type: "line",
    data: { datasets: [series(f.v.map((v) => v * 3.6), "hacia delante"), series(f.v.map((v, i) => Math.hypot(v, f.v_lat[i]) * 3.6), "total", css("--c3"))] },
    options: fast(),
  });
  makeChart(chartCard(el, "ph-slip", "Derrape lateral [km/h]"), {
    type: "line",
    data: { datasets: [series(f.v_lat.map((v) => v * 3.6), "derrape", css("--accent"))] },
    options: fast({ legend: false }),
  });
  makeChart(chartCard(el, "ph-steer", "Dirección del volante [-1, 1] y ángulo de rueda"), {
    type: "line",
    data: { datasets: [series(f.steer_cmd, "volante [-1, 1]", css("--c3")), series(f.steer, "ángulo de rueda [rad]")] },
    options: fast(),
  });
  makeChart(chartCard(el, "ph-lat", "Desplazamiento lateral [m]", {
    what: "Distancia del coche a la línea central de la pista. Las líneas añil marcan el borde del asfalto, y las rojas, el muro.",
  }), {
    type: "line",
    data: { datasets: [series(f.lateral, "desplazamiento"), flat(half, "_g1", css("--good")), flat(-half, "_g2", css("--good")), flat(wall, "_w1", css("--accent")), flat(-wall, "_w2", css("--accent"))] },
    options: fast({ legend: false }),
  });
  makeChart(chartCard(el, "ph-throttle", "Acelerador y freno"), {
    type: "line",
    data: { datasets: [series(f.throttle, "acelerador", css("--c5"), { fill: true, backgroundColor: alpha(css("--c5"), 0.15) })] },
    options: fast({ legend: false, y: { min: -1, max: 1 } }),
  });

  const bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 2.01];
  const labels = bins.slice(0, -1).map((b, i) => `${num(b, 2)} a ${bins[i + 1] > 2 ? "2" : num(bins[i + 1], 2)}`);
  const datasets = Object.entries(D.replays).map(([k, eps]) => {
    const counts = new Array(labels.length).fill(0);
    let total = 0;
    eps.forEach((e) =>
      e.frames.steer_cmd.forEach((s, i) => {
        if (!i) return;
        const dd = Math.abs(s - e.frames.steer_cmd[i - 1]);
        const b = bins.findIndex((edge, j) => dd >= edge && dd < bins[j + 1]);
        if (b >= 0) counts[b]++;
        total++;
      })
    );
    return { label: policyName(k), data: counts.map((c) => c / Math.max(total, 1)), backgroundColor: alpha(policyColor(k), 0.85) };
  });
  makeChart(chartCard(el, "ph-jitter", "Cambio de volante por paso (jitter)", {
    what: "Cuánto se mueve el volante de un paso al siguiente, como parte de todos los pasos de cada piloto.",
    good: "Casi todo en las barras de la izquierda, que son giros suaves.",
  }), {
    type: "bar",
    data: { labels, datasets },
    options: { animation: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } }, scales: { y: { min: 0, ticks: { callback: (v) => pct(v) } } } },
  });

  const canvas = $("#phys-map");
  const { ctx, w, h } = sizeCanvas(canvas);
  const view = fitView(track, w, h);
  drawTrack(ctx, track, view, w, h);
  drawTrail(ctx, f, f.x.length - 1, view);
  drawCar(ctx, f, f.x.length - 1, view, false);
  Object.entries(D.replays).forEach(([k, eps]) => {
    const e = eps.find((x) => String(x.seed) === seed);
    if (!isPpo(k) || !e || e.outcome === "lap") return;
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
  $("#phys-legend").innerHTML = `<span style="color:${speedColor(0)}">lento</span><span>rastro coloreado por velocidad, la X marca dónde falló un PPO</span><span style="color:${speedColor(D.meta.max_speed)}">rápido</span>`;

  const fails = [];
  Object.entries(D.replays).forEach(([k, eps]) => isPpo(k) && eps.forEach((e) => e.outcome !== "lap" && fails.push([k, e])));
  $("#failures").innerHTML = fails.length
    ? fails
        .map(([k, e]) => `<div class="list-item" data-k="${k}" data-seed="${e.seed}"><div class="icon-box" style="color:${outcomeColor(e.outcome)}">${icon("alert")}</div>
          <div class="grow"><div>${esc(policyName(k))}, pista ${e.seed}</div><div class="sub">${OUTCOME[e.outcome][0]} al ${pct(e.progress)} de la vuelta, tras ${num(e.frames.x.length * e.dt)} s</div></div></div>`)
        .join("")
    : `<div class="empty">Ningún modelo PPO falló en estas pistas</div>`;
  $$("#failures .list-item").forEach((item) =>
    item.addEventListener("click", () => {
      physState.key = item.dataset.k;
      physState.seed = item.dataset.seed;
      renderPhysics();
    })
  );
}

// ---------- arcade race against the model ----------
// The player's car runs a line by line port of rust/src/car.rs with the constants from data.js,
// so it handles exactly like the car the model learned on. The assist only filters the keys.
const race = {
  phase: "title", sel: 0, raf: 0, last: 0, acc: 0, count: 0, keys: new Set(), assist: true, pix: 1,
  T: null, car: null, prev: null, idx: 0, frac: 0, progress: 0, lateral: 0, onGrass: false,
  grassSteps: 0, steps: 0, result: null, ctx: null, w: 0, h: 0, steerCmd: 0, thrCmd: 0, audio: null, busy: false,
};
const RACE_KEYS = {
  ArrowUp: "up", KeyW: "up", ArrowDown: "down", KeyS: "down",
  ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right",
};

const raceTracks = () => (D.race && D.race.tracks) || [];
const current = () => raceTracks()[race.sel];
const worldsOf = () => [...new Set(raceTracks().map((r) => r.difficulty))].sort((a, b) => a - b);
const themeOf = (r) => WORLD_THEMES[worldsOf().indexOf(r.difficulty) % WORLD_THEMES.length];

function stepCar(st, steerIn, throttleIn, onGrass) {
  const K = D.meta.car;
  let [x, y, yaw, v, steer, vLat] = st;
  const cmd = clamp(steerIn, -1, 1) * K.MAX_STEER;
  steer += clamp(cmd - steer, -K.STEER_RATE * K.DT, K.STEER_RATE * K.DT);
  const grip = onGrass ? K.GRASS_GRIP : K.GRIP;
  const drag = K.DRAG + (onGrass ? K.GRASS_DRAG : 0);
  const throttle = clamp(throttleIn, -1, 1);
  const braking = v * throttle < 0;
  const force = braking ? K.BRAKE : K.ACCEL;
  let vNew = v + (force * throttle - drag * v) * K.DT;
  if (braking && vNew * v < 0) vNew = 0; // brakes stop the car, reverse starts on the next step
  vNew = clamp(vNew, -K.REVERSE_SPEED, K.MAX_SPEED);
  let tanSteer = Math.tan(steer);
  if (Math.abs(vNew) > 1e-3) {
    const limit = (((K.YAW_GRIP * grip) / K.GRIP) * K.WHEELBASE) / (vNew * vNew);
    tanSteer = clamp(tanSteer, -limit, limit);
  }
  const dyaw = ((vNew * tanSteer) / K.WHEELBASE) * K.DT;
  const vLong = clamp(vNew + dyaw * vLat, -K.REVERSE_SPEED, K.MAX_SPEED);
  vLat -= dyaw * vNew;
  vLat -= clamp(vLat, -grip * K.DT, grip * K.DT);
  yaw += dyaw;
  x += (vLong * Math.cos(yaw) - vLat * Math.sin(yaw)) * K.DT;
  y += (vLong * Math.sin(yaw) + vLat * Math.cos(yaw)) * K.DT;
  return [x, y, yaw, vLong, steer, vLat];
}

// keyboard to car command. Without the assist the keys go straight to the car like the model's actions
function playerInput() {
  const K = D.meta.car, k = race.keys;
  const wantSteer = (k.has("left") ? 1 : 0) - (k.has("right") ? 1 : 0);
  const wantThr = (k.has("up") ? 1 : 0) - (k.has("down") ? 1 : 0);
  if (!race.assist) return [wantSteer, wantThr];
  const rate = K.DT / 0.15; // full lock in 0.15 s, back to center as fast
  race.steerCmd += clamp(wantSteer - race.steerCmd, -rate, rate);
  const v = Math.abs(race.car[3]);
  let limit = 1;
  if (v > 1) limit = Math.min(1, Math.atan((0.9 * K.GRIP * K.WHEELBASE) / (v * v)) / K.MAX_STEER);
  let steer = clamp(race.steerCmd, -limit, limit);
  steer = clamp(steer + clamp(0.12 * race.car[5], -0.35, 0.35), -1, 1); // steer into a slide
  const trate = K.DT / 0.25;
  race.thrCmd += clamp(wantThr - race.thrCmd, -trate, trate);
  if (wantThr < 0 && race.car[3] > 0.5) race.thrCmd = Math.min(race.thrCmd, -0.6);
  return [steer, race.thrCmd];
}

function prepTrack(g) {
  const c = g.centerline, n = c.length;
  const heading = c.map((_, i) => {
    const a = c[(i - 1 + n) % n], b = c[(i + 1) % n];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  });
  const s = [0];
  for (let i = 1; i < n; i++) s.push(s[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
  const length = s[n - 1] + Math.hypot(c[0][0] - c[n - 1][0], c[0][1] - c[n - 1][1]);
  return { g, c, n, heading, s, length };
}

function locate(T, x, y, lastIdx) {
  let best = lastIdx, bestD = Infinity;
  for (let k = -12; k <= 12; k++) {
    const i = (((lastIdx + k) % T.n) + T.n) % T.n;
    const d = (T.c[i][0] - x) ** 2 + (T.c[i][1] - y) ** 2;
    if (d < bestD) [best, bestD] = [i, d];
  }
  return best;
}

function frenet(T, x, y, i) {
  const h = T.heading[i], dx = x - T.c[i][0], dy = y - T.c[i][1];
  const along = dx * Math.cos(h) + dy * Math.sin(h);
  const lateral = -dx * Math.sin(h) + dy * Math.cos(h);
  return [mod1((T.s[i] + along) / T.length), lateral];
}

const bestKey = (r) => `race-best-${r.seed}-${r.difficulty}`;
const bestOf = (r) => Number(store.get(bestKey(r))) || null;

function stars(r) {
  const best = bestOf(r);
  if (!best) return 0;
  return best < r.lap_time ? 3 : best < r.lap_time + 1 ? 2 : 1;
}

function beep(freq, dur) {
  try {
    race.audio ||= new (window.AudioContext || window.webkitAudioContext)();
    const a = race.audio, o = a.createOscillator(), g = a.createGain();
    o.type = "square";
    o.frequency.value = freq;
    g.gain.setValueAtTime(0.06, a.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + dur);
    o.connect(g).connect(a.destination);
    o.start();
    o.stop(a.currentTime + dur);
  } catch (e) {
    /* sound is optional */
  }
}

function resetRace() {
  const r = current();
  if (!r) return;
  race.T = prepTrack(r.track);
  const [x, y] = race.T.c[0];
  race.car = [x, y, race.T.heading[0], 0, 0, 0];
  race.prev = race.car;
  Object.assign(race, { idx: 0, frac: 0, progress: 0, lateral: 0, onGrass: false, grassSteps: 0, steps: 0, acc: 0, result: null, steerCmd: 0, thrCmd: 0 });
}

function openLevel(i) {
  race.sel = i;
  race.phase = "idle";
  resetRace();
  raceOverlay();
}

function startRace() {
  resetRace();
  race.phase = "count";
  race.count = 0;
  race.keys.clear();
  $("#race-canvas").focus({ preventScroll: true });
  beep(440, 0.12);
  // the controls appear big with the first light and shrink into the corner on the second beep
  const keysBox = $("#race-keys");
  keysBox.hidden = false;
  keysBox.classList.remove("mini");
  raceOverlay();
}

function raceStep() {
  const K = D.meta.car, T = race.T;
  const [steer, throttle] = playerInput();
  race.prev = race.car;
  race.car = stepCar(race.car, steer, throttle, race.onGrass);
  race.steps += 1;
  race.idx = locate(T, race.car[0], race.car[1], race.idx);
  const [frac, lateral] = frenet(T, race.car[0], race.car[1], race.idx);
  race.progress += mod1(frac - race.frac + 0.5) - 0.5; // unwrap across the start line
  race.frac = frac;
  race.lateral = lateral;
  race.onGrass = Math.abs(lateral) > T.g.width / 2;
  race.grassSteps = race.onGrass ? race.grassSteps + 1 : 0;
  const time = race.steps * K.DT;
  if (race.progress >= 1) finishRace({ kind: "lap", time });
  else if (Math.abs(lateral) > K.WALL_OFFSET) finishRace({ kind: "wall", time });
  else if (race.grassSteps * K.DT > K.GRASS_TIME) finishRace({ kind: "grass", time });
  else if (time >= K.MAX_TIME) finishRace({ kind: "timeout", time });
}

function finishRace(result) {
  const r = current();
  race.phase = "done";
  race.result = result;
  if (result.kind === "lap") {
    const best = bestOf(r);
    if (!best || result.time < best) store.set(bestKey(r), result.time.toFixed(2));
    beep(result.time < r.lap_time ? 988 : 330, 0.35);
  } else beep(150, 0.4);
  raceOverlay();
}

function stopRace() {
  cancelAnimationFrame(race.raf);
  race.raf = 0;
  race.keys.clear();
  if (race.phase === "count" || race.phase === "run") {
    race.phase = "idle";
    resetRace();
  }
}

function ghostTimeAt(p) {
  const g = current().ghost.progress;
  let lo = 0, hi = g.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (g[mid] >= p) hi = mid;
    else lo = mid + 1;
  }
  return lo * D.meta.car.DT;
}

function ghostPose(t) {
  const g = current().ghost, n = g.x.length;
  const f = clamp(t / D.meta.car.DT, 0, n - 1), i = Math.floor(f), j = Math.min(i + 1, n - 1), a = f - i;
  const lerp = (arr) => arr[i] + (arr[j] - arr[i]) * a;
  const dyaw = Math.atan2(Math.sin(g.yaw[j] - g.yaw[i]), Math.cos(g.yaw[j] - g.yaw[i]));
  return [lerp(g.x), lerp(g.y), g.yaw[i] + dyaw * a];
}

// big pixels, the frame is shrunk to blocks and blown back up without smoothing
const pixCanvas = document.createElement("canvas");
function pixelate(ctx, block) {
  const c = ctx.canvas;
  const sw = Math.max(1, Math.round(c.width / block)), sh = Math.max(1, Math.round(c.height / block));
  [pixCanvas.width, pixCanvas.height] = [sw, sh];
  const p = pixCanvas.getContext("2d");
  p.imageSmoothingEnabled = true;
  p.drawImage(c, 0, 0, sw, sh);
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(pixCanvas, 0, 0, sw, sh, 0, 0, c.width, c.height);
  ctx.restore();
}

function drawRace(ts) {
  const r = current();
  const { ctx, w, h } = race;
  if (!r || !ctx) return;
  const DT = D.meta.car.DT;
  const attract = ["title", "menu", "idle"].includes(race.phase);
  let t, x, y, yaw;
  if (attract) {
    // attract mode, the model's lap loops behind the screens
    t = (ts / 1000) % (r.lap_time + 1.5);
    [x, y, yaw] = ghostPose(t);
  } else {
    const a = race.phase === "run" ? race.acc / DT : 1;
    const P = race.prev, C = race.car;
    x = P[0] + (C[0] - P[0]) * a;
    y = P[1] + (C[1] - P[1]) * a;
    yaw = P[2] + Math.atan2(Math.sin(C[2] - P[2]), Math.cos(C[2] - P[2])) * a;
    t = race.phase === "count" ? 0 : Math.max(0, (race.steps - 1 + a) * DT);
  }
  const scale = clamp(w / 105, 3.5, 7.5);
  const view = followView(x + Math.cos(yaw) * 10, y + Math.sin(yaw) * 10, w, h, scale);
  drawTrack(ctx, r.track, view, w, h, ts, themeOf(r));
  const [gx, gy, gyaw] = ghostPose(t);
  drawSprite(ctx, gx, gy, gyaw, view, carColors("#d9483f"));
  if (!attract) drawSprite(ctx, x, y, yaw, view, carColors(PLAYER_BLUE, "#f7f0e1"));

  const m = Math.min(130, w * 0.24), ox = 12, oy = h - m - 12;
  const mini = fitView(r.track, m, m, 8);
  const map = (px, py) => {
    const [u, v] = mini.map(px, py);
    return [ox + u, oy + v];
  };
  ctx.fillStyle = "rgba(15, 13, 16, 0.78)";
  ctx.fillRect(ox, oy, m, m);
  ctx.strokeStyle = "#f3e9d2";
  ctx.lineWidth = 2;
  ctx.beginPath();
  pathOf(ctx, r.track.centerline, { map });
  ctx.stroke();
  const dots = [[gx, gy, "#d9483f"]];
  if (!attract) dots.push([x, y, PLAYER_BLUE]);
  dots.forEach(([px, py, color]) => {
    const [u, v] = map(px, py);
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(u) - 3, Math.round(v) - 3, 6, 6);
  });
  if (race.pix > 1) pixelate(ctx, race.pix * (window.devicePixelRatio || 1));
  if (!attract) raceHud(t);
}

function raceHud(t) {
  const r = current();
  const v = race.car ? Math.hypot(race.car[3], race.car[5]) * 3.6 : 0;
  const p = clamp(race.progress, 0, 1);
  let gap = "-", cls = "";
  if (race.phase === "run" && p > 0.01) {
    const d = t - ghostTimeAt(p);
    gap = `${d > 0 ? "+" : "-"}${Math.abs(d).toFixed(1)}`;
    cls = d > 0 ? "behind" : "ahead";
  }
  const time = race.phase === "done" ? race.result.time : t;
  const cells = [
    ["tiempo", time.toFixed(2)],
    ["km/h", v.toFixed(0)],
    ["vuelta", `${Math.round(100 * p)}%`],
    ["vs IA", gap, cls],
    ["IA", r.lap_time.toFixed(2)],
  ];
  const hud = $("#race-hud");
  hud.hidden = false;
  hud.innerHTML = cells.map(([label, value, c = ""]) => `<div>${label}<b class="${c}">${value}</b></div>`).join("");
}

// a pixel dissolve, blocks cover the screen in random order, the next screen is swapped in
// underneath while it is covered, then the blocks clear in the same random order
function pixelTransition(mid) {
  const fx = $("#race-fx");
  if (REDUCED || race.busy) {
    mid();
    return;
  }
  race.busy = true;
  const cols = 32, rows = 20;
  [fx.width, fx.height] = [cols, rows];
  const order = Array.from({ length: cols * rows }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const o = fx.getContext("2d");
  fx.hidden = false;
  const t0 = performance.now(), total = 900;
  let swapped = false;
  const step = (ts) => {
    const k = (ts - t0) / total;
    if (k >= 1) {
      fx.hidden = true;
      race.pix = 1;
      race.busy = false;
      return;
    }
    const cover = k < 0.5 ? k * 2 : (1 - k) * 2;
    o.clearRect(0, 0, cols, rows);
    o.fillStyle = "#0b0a0c";
    const count = Math.floor(cover * order.length);
    for (let i = 0; i < count; i++) o.fillRect(order[i] % cols, Math.floor(order[i] / cols), 1, 1);
    race.pix = 1 + Math.round(cover * 14);
    if (!swapped && k >= 0.5) {
      swapped = true;
      mid();
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function toMenu() {
  pixelTransition(() => {
    race.phase = "menu";
    raceOverlay();
  });
}

function levelGrid() {
  const all = raceTracks();
  return worldsOf()
    .map((d, wi) => {
      const cards = all
        .map((r, i) => [r, i])
        .filter(([r]) => r.difficulty === d)
        .map(([r, i], li) => {
          const s = stars(r);
          return `<button class="level ${i === race.sel ? "sel" : ""}" data-i="${i}"><canvas data-thumb="${i}"></canvas>
            <span>nivel ${wi + 1}-${li + 1}</span><span>IA ${r.lap_time.toFixed(1)}s</span>
            <span class="stars">${"★".repeat(s)}${"☆".repeat(3 - s)}</span></button>`;
        })
        .join("");
      return `<div class="world"><div class="world-name">mundo ${wi + 1} · <b>${WORLD_NAMES[wi % WORLD_NAMES.length]}</b> · ${levelName(d).toLowerCase()}</div><div class="level-row">${cards}</div></div>`;
    })
    .join("");
}

function raceOverlay() {
  const el = $("#race-overlay"), r = current();
  const keysBox = $("#race-keys");
  if (race.phase !== "count" && race.phase !== "run") keysBox.hidden = true;
  if (["title", "menu", "idle"].includes(race.phase)) $("#race-hud").hidden = true;
  if (!r) {
    el.innerHTML = `<div class="panel8"><h4>sin pistas</h4><p>Genera el dashboard con las runs finales, make dashboard.</p></div>`;
    return;
  }
  if (race.phase === "title") {
    el.innerHTML = `<div class="title8"><h4>Learn for Speed</h4><p>arcade · ${raceTracks().length} pistas · ${worldsOf().length} mundos</p><div class="blink">pulsa enter</div></div>`;
    $(".title8", el).addEventListener("click", toMenu);
    return;
  }
  if (race.phase === "menu") {
    el.innerHTML = `<div class="levels"><h4>elige nivel</h4>${levelGrid()}<div class="blink">enter para elegir · esc para salir</div></div>`;
    $$("#race-overlay .level").forEach((b) => {
      b.addEventListener("click", () => openLevel(Number(b.dataset.i)));
      b.addEventListener("mouseenter", () => {
        race.sel = Number(b.dataset.i);
        $$("#race-overlay .level").forEach((x) => x.classList.toggle("sel", x === b));
      });
    });
    $$("#race-overlay canvas[data-thumb]").forEach((c) => {
      const tr = raceTracks()[Number(c.dataset.thumb)];
      const { ctx, w, h } = sizeCanvas(c);
      drawTrack(ctx, tr.track, fitView(tr.track, w, h, 4), w, h, performance.now(), themeOf(tr));
    });
    const sel = $("#race-overlay .level.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
    return;
  }
  const name = `pista ${r.seed} · ${WORLD_NAMES[worldsOf().indexOf(r.difficulty) % WORLD_NAMES.length]}`;
  if (race.phase === "count") {
    const lit = Math.min(3, Math.floor(race.count) + 1);
    el.innerHTML = `<div class="semaphore">${[0, 1, 2].map((i) => `<i class="${i < lit ? "red" : ""}"></i>`).join("")}</div>`;
    return;
  }
  if (race.phase === "run") {
    el.innerHTML = race.count < 3.6 ? `<div class="semaphore go"><i></i><i></i><i></i></div><div class="go-text">¡ya!</div>` : "";
    return;
  }
  if (race.phase === "idle") {
    const best = bestOf(r);
    el.innerHTML = `<div class="panel8"><h4>${name}</h4>
      <p>El coche rojo es la IA, que conduce su propia vuelta solo con el lidar. Tu coche azul tiene la misma física.</p>
      <p>IA ${r.lap_time.toFixed(2)} s${best ? ` · tu récord ${best.toFixed(2)} s` : ""}</p>
      <div class="row"><button class="pbtn" id="race-go">empezar · enter</button><button class="pbtn alt" id="race-back">niveles · esc</button></div></div>`;
  } else {
    const res = race.result;
    const margin = res.time - r.lap_time;
    const lap = res.kind === "lap";
    const title = !lap ? (res.kind === "timeout" ? "tiempo agotado" : "fuera de pista") : margin < 0 ? "¡ganaste a la IA!" : "gana la IA";
    const text = lap
      ? `${res.time.toFixed(2)} s contra ${r.lap_time.toFixed(2)} s, ${margin < 0 ? "por delante" : "por detrás"} ${Math.abs(margin).toFixed(2)} s.`
      : res.kind === "wall"
        ? `Tocaste el muro al ${Math.round(100 * clamp(race.progress, 0, 1))} % de la vuelta.`
        : res.kind === "grass"
          ? `Más de ${D.meta.car.GRASS_TIME} s fuera del asfalto acaba el intento, igual que para la IA.`
          : `El límite por vuelta es ${D.meta.car.MAX_TIME} s.`;
    const s = stars(r);
    el.innerHTML = `<div class="panel8"><h4>${title}</h4><p>${text}</p><p class="stars">${"★".repeat(s)}${"☆".repeat(3 - s)}</p>
      <div class="row"><button class="pbtn" id="race-go">otra vez · r</button><button class="pbtn alt" id="race-back">niveles · esc</button></div></div>`;
  }
  $("#race-go").addEventListener("click", startRace);
  $("#race-back").addEventListener("click", toMenu);
}

function raceFrame(ts) {
  if (state.view !== "race") return;
  const dt = Math.min(0.1, Math.max(0, (ts - race.last) / 1000));
  race.last = ts;
  if (race.phase === "count") {
    const before = Math.floor(race.count);
    race.count += dt;
    const now = Math.floor(race.count);
    if (race.count >= 3) {
      beep(880, 0.3);
      race.phase = "run";
      race.acc = 0;
      raceOverlay();
    } else if (now !== before) {
      beep(440, 0.12);
      if (now === 1) $("#race-keys").classList.add("mini");
      raceOverlay();
    }
  } else if (race.phase === "run") {
    if (race.count < 3.6) {
      race.count += dt;
      if (race.count >= 3.6) raceOverlay();
    }
    race.acc += dt;
    while (race.acc >= D.meta.car.DT && race.phase === "run") {
      race.acc -= D.meta.car.DT;
      raceStep();
    }
  }
  drawRace(ts);
  race.raf = requestAnimationFrame(raceFrame);
}

function renderRace() {
  const all = raceTracks();
  $("#race-note").textContent = all.length
    ? `${all.length} pistas de validación que ${policyName(D.race.policy)} completó. Termina la vuelta antes que su fantasma.`
    : "";
  race.assist = (store.get("race-assist") || "1") === "1";
  $("#race-assist").checked = race.assist;
  Object.assign(race, sizeCanvas($("#race-canvas")));
  if (race.sel >= all.length) race.sel = 0;
  if (!race.car) resetRace();
  raceOverlay();
  cancelAnimationFrame(race.raf);
  race.last = performance.now();
  if (all.length) race.raf = requestAnimationFrame(raceFrame);
}

function moveLevel(dx) {
  const n = raceTracks().length;
  if (!n) return;
  race.sel = (race.sel + dx + n) % n;
  raceOverlay();
}

// up and down jump to the same slot in the previous or next world
function moveWorld(dy) {
  const all = raceTracks(), worlds = worldsOf();
  const r = current();
  const wi = worlds.indexOf(r.difficulty);
  const slot = all.filter((x) => x.difficulty === r.difficulty).indexOf(r);
  const target = worlds[clamp(wi + dy, 0, worlds.length - 1)];
  const row = all.map((x, i) => [x, i]).filter(([x]) => x.difficulty === target);
  race.sel = row[Math.min(slot, row.length - 1)][1];
  raceOverlay();
}

function bindRace() {
  document.addEventListener("keydown", (e) => {
    if (state.view !== "race" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (!raceTracks().length || race.busy) return;
    if (race.phase === "title") {
      if (e.code === "Enter" || e.code === "Space") {
        e.preventDefault();
        beep(660, 0.1);
        toMenu();
      }
      return;
    }
    if (race.phase === "menu") {
      const across = { ArrowRight: 1, ArrowLeft: -1 }, down = { ArrowDown: 1, ArrowUp: -1 };
      if (across[e.code] != null) {
        e.preventDefault();
        moveLevel(across[e.code]);
      }
      if (down[e.code] != null) {
        e.preventDefault();
        moveWorld(down[e.code]);
      }
      if (e.code === "Enter") {
        e.preventDefault();
        openLevel(race.sel);
      }
      if (e.code === "Escape") {
        race.phase = "title";
        raceOverlay();
      }
      return;
    }
    const key = RACE_KEYS[e.code];
    if (key) {
      e.preventDefault();
      race.keys.add(key);
    }
    if (e.code === "Escape") {
      stopRace();
      race.last = performance.now();
      race.raf = requestAnimationFrame(raceFrame);
      toMenu();
    }
    if (e.code === "Enter" && (race.phase === "idle" || race.phase === "done")) {
      e.preventDefault();
      startRace();
    }
    if (e.code === "KeyR" && race.phase !== "idle") startRace();
  });
  document.addEventListener("keyup", (e) => {
    const key = RACE_KEYS[e.code];
    if (key) race.keys.delete(key);
  });
  window.addEventListener("blur", () => race.keys.clear());
  $("#race-assist").addEventListener("change", (e) => {
    race.assist = e.target.checked;
    store.set("race-assist", race.assist ? "1" : "0");
  });
  $$("#race-pad button").forEach((b) => {
    const on = (e) => {
      e.preventDefault();
      race.keys.add(b.dataset.key);
      b.classList.add("on");
    };
    const off = () => {
      race.keys.delete(b.dataset.key);
      b.classList.remove("on");
    };
    b.addEventListener("pointerdown", on);
    ["pointerup", "pointerleave", "pointercancel"].forEach((t) => b.addEventListener(t, off));
  });
}

// ---------- shell, arrows, the kart line and the channel number ----------
const nav = { kartX: null, osdTimer: 0 };

function renderNav() {
  $("#stops").innerHTML = VIEWS.map(([id, label]) => `<button data-view="${id}" class="${state.view === id ? "active" : ""}">${label}</button>`).join("");
  $$("#stops button").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
}

// the pixel arrows step through the sections and wrap around at the ends
function stepView(d) {
  const i = VIEWS.findIndex((v) => v[0] === state.view);
  go(VIEWS[(i + d + VIEWS.length) % VIEWS.length][0]);
}

// the kart recoils away from the new stop, then sprints to it and brakes with a small overshoot
function moveKart(animate) {
  const stop = $(`#stops button[data-view="${state.view}"]`);
  const line = $(".track");
  if (!stop || !line) return;
  const target = stop.getBoundingClientRect().left + stop.offsetWidth / 2 - line.getBoundingClientRect().left - 24;
  const from = nav.kartX == null ? target : nav.kartX;
  const dir = target >= from ? 1 : -1;
  drawKart(dir);
  const kart = $("#kart");
  const tf = (x) => `translateX(${x}px)`;
  kart.getAnimations().forEach((a) => a.cancel());
  kart.style.transform = tf(target);
  if (animate && !REDUCED && Math.abs(target - from) > 1) {
    kart.animate(
      [
        { transform: tf(from) },
        { transform: tf(from - dir * 12), offset: 0.18 },
        { transform: tf(target + dir * 9), offset: 0.78 },
        { transform: tf(target - dir * 3), offset: 0.9 },
        { transform: tf(target) },
      ],
      { duration: 680, easing: "cubic-bezier(.4,0,.2,1)" }
    );
  }
  nav.kartX = target;
}

// the channel number in green, as the TV shows it for a moment after zapping
function showOsd(view) {
  const i = VIEWS.findIndex((v) => v[0] === view);
  const osd = $("#osd");
  osd.innerHTML = `${i + 1}<small>${VIEWS[i][1]}</small>`;
  osd.classList.add("on");
  clearTimeout(nav.osdTimer);
  nav.osdTimer = setTimeout(() => osd.classList.remove("on"), 1600);
}

const RENDER = {
  overview: () => {
    renderCluster();
    renderCounters();
    startHero();
  },
  race: renderRace,
  training: renderTraining,
  validation: renderValidation,
  statistics: renderStatistics,
  test: renderTest,
  simulation: renderSimulation,
  physics: renderPhysics,
};

// a broken view shows its error in a card instead of staying blank
function safeRender(view) {
  pruneCharts();
  try {
    RENDER[view]();
  } catch (err) {
    console.error(err);
    const section = $(`.view[data-view="${view}"]`);
    section.querySelectorAll(".render-error").forEach((e) => e.remove());
    section.insertAdjacentHTML("beforeend", `<div class="card render-error">No se pudo dibujar esta sección. ${esc(err.message)}</div>`);
  }
  requestAnimationFrame(resizeCharts);
}

function go(view, animate = true) {
  const changed = view !== state.view;
  state.view = view;
  $$(".view").forEach((v) => (v.hidden = v.dataset.view !== view));
  if (view !== "simulation") setPlaying(false);
  if (view !== "race") stopRace();
  try {
    history.replaceState(null, "", `#${view}`);
  } catch (e) {
    /* the hash is a convenience */
  }
  renderNav();
  moveKart(animate && changed);
  if (animate && changed) showOsd(view);
  safeRender(view);
  if (changed) $("#screen").scrollTo({ top: 0, behavior: "auto" });
}

function renderAll() {
  chartDefaults();
  spriteCache.clear();
  go(state.view, false);
}

function applyTheme(choice) {
  store.set("theme", choice);
  if (choice === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", choice);
  $$("#theme button").forEach((btn) => btn.classList.toggle("active", btn.dataset.themeSet === choice));
  renderAll();
}

function bindShell() {
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => go(b.dataset.goto)));
  $$("#theme button").forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeSet)));
  $("#nav-prev").addEventListener("click", () => stepView(-1));
  $("#nav-next").addEventListener("click", () => stepView(1));
  $("#search").addEventListener("input", (e) => {
    state.search = e.target.value;
    if (state.view === "training") renderTraining();
  });
  $("#runs-all").addEventListener("click", () => setRuns(D.runs.map((r) => r.name), true));
  $("#runs-none").addEventListener("click", () => setRuns(D.runs.map((r) => r.name), false));
  bindRace();
  $("#play").addEventListener("click", () => setPlaying(!simState.playing));
  $("#scrub").addEventListener("input", (e) => {
    simState.t = (Number(e.target.value) / 1000) * simState.maxT;
    drawSim();
  });
  ["#opt-lidar", "#opt-follow", "#opt-trail"].forEach((id) => $(id).addEventListener("change", drawSim));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;
    if (e.key === "[" || e.key === "]") stepView(e.key === "]" ? 1 : -1);
    if (e.key === "/" && state.view === "training") {
      e.preventDefault();
      $("#search").focus();
    }
    if (e.key === " " && state.view === "simulation") {
      e.preventDefault();
      setPlaying(!simState.playing);
    }
  });
  let resizeTimer, lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      moveKart(false);
      if (window.innerWidth === lastWidth) return;
      lastWidth = window.innerWidth;
      if (state.view === "simulation") renderSimTiles();
      else if (state.view === "race") Object.assign(race, sizeCanvas($("#race-canvas")));
      else if (state.view !== "physics" && state.view !== "training") safeRender(state.view);
    }, 150);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if ((store.get("theme") || "system") === "system") applyTheme("system");
  });
  if (!REDUCED) setInterval(() => drawFlag(performance.now()), 160);
}

if (!D) {
  document.body.innerHTML = '<div class="empty">No hay data.js. Ejecuta <code>make dashboard</code> primero.</div>';
} else {
  fillIcons();
  const start = location.hash.slice(1);
  if (VIEWS.some((v) => v[0] === start)) state.view = start;
  bindShell();
  drawFlag(0);
  applyTheme(store.get("theme") || "system");
}
