"use strict";
(async function main() {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const META = JSON.parse($("#meta").textContent);
  const CLASSES = META.classes;
  const NAMES = ["Crazing", "Inclusion", "Patches", "Pitted surface", "Rolled-in scale", "Scratches"];
  const SHORT = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled-in", "Scratches"];
  const DESCRIPTIONS = [
    "A network of fine cracks in the surface. This is our crack class.",
    "Bits of non-metal material pressed into the steel.",
    "Irregular areas that are darker or lighter than the steel around them.",
    "Small pits and craters scattered over the surface.",
    "Oxide scale that was pressed into the strip during rolling.",
    "Long thin scratch lines on the surface.",
  ];
  const MODELS = {
    m1: { short: "M1", long: "M1 baseline CNN", color: "#eb6834" },
    m2: { short: "M2", long: "M2 improved CNN", color: "#1baf7a" },
    m3: { short: "M3", long: "M3 MobileNetV2", color: "#2a78d6" },
  };
  const N = 200;
  const status = $("#status");

  const num = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const pct = (v, d = 1) => (v * 100).toFixed(d) + " %";
  const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---------- colours
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const NAVY = rgb("#191e6b"), BLUE = rgb("#2a78d6"), RED = rgb("#e34948"), MID = rgb("#f0efec"), HEAT = rgb("#f5a300");
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const navyRamp = (t) => mix([255, 255, 255], NAVY, clamp01(t));
  const diverging = (t) => (t >= 0 ? mix(MID, BLUE, Math.min(t, 1)) : mix(MID, RED, Math.min(-t, 1)));
  const inkOn = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 140 ? "#fff" : "#14162b");
  const BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"];

  // ---------- weights
  function bytesOf(id) {
    const bin = atob($(id).textContent.trim());
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function halfToFloat(h) {
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 31, f = x & 1023;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return out;
  }
  function unpack(manifest, flat) {
    const w = {};
    for (const m of manifest) w[m.name] = { shape: m.shape, data: flat.subarray(m.offset, m.offset + m.size) };
    return w;
  }

  // Shape uniforms let MobileNetV2's many layer shapes share compiled GPU programs, so the first run starts faster.
  tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);
  let backend = "cpu";
  try {
    if (await tf.setBackend("webgl")) backend = "webgl";
  } catch (err) {
    backend = "cpu";
  }
  if (backend === "webgl" && !tf.env().getBool("WEBGL_RENDER_FLOAT32_ENABLED")) backend = "cpu";
  if (backend === "cpu") await tf.setBackend("cpu");
  await tf.ready();

  const M1W = unpack(META.m1.manifest, new Float32Array(bytesOf("#m1-weights").buffer));
  const M3W = unpack(META.m3.manifest, halfToFloat(new Uint16Array(bytesOf("#m3-weights").buffer)));
  const M1T = {}, M3T = {};
  function makeTensors() {
    for (const [w, t] of [[M1W, M1T], [M3W, M3T]]) {
      for (const [k, v] of Object.entries(w)) {
        if (t[k]) t[k].dispose();
        t[k] = tf.tensor(v.data, v.shape);
      }
    }
  }
  makeTensors();
  const GRAPH = META.m3.graph;
  const LAST = GRAPH[GRAPH.length - 1].name;
  const CONSUMERS = {};
  for (const n of GRAPH) for (const i of n.inputs) CONSUMERS[i] = (CONSUMERS[i] || 0) + 1;
  const M3_STAGE_KEYS = ["Conv1_relu", "expanded_conv_project", "block_2_add", "block_5_add", "block_12_add", "block_16_project", "out_relu"]
    .filter((k) => GRAPH.some((n) => n.name === k));

  // ---------- models
  const m3Head = (f) => tf.matMul(tf.mean(f, [1, 2]), M3T["dense/kernel"]).add(M3T["dense/bias"]);

  function m1Forward(x) {
    return tf.tidy(() => {
      const conv = (t, name) => tf.relu(tf.conv2d(t, M1T[name + "/kernel"], 1, "same").add(M1T[name + "/bias"]));
      const a1 = conv(x, "conv2d"), p1 = tf.maxPool(a1, 2, 2, "valid");
      const a2 = conv(p1, "conv2d_1"), p2 = tf.maxPool(a2, 2, 2, "valid");
      const a3 = conv(p2, "conv2d_2"), p3 = tf.maxPool(a3, 2, 2, "valid");
      const g = tf.mean(p3, [1, 2]);
      const d = tf.relu(tf.matMul(g, M1T["dense/kernel"]).add(M1T["dense/bias"]));
      const z = tf.matMul(d, M1T["dense_1/kernel"]).add(M1T["dense_1/bias"]);
      return { a1, p1, a2, p2, a3, p3, g, d, z, p: tf.softmax(z) };
    });
  }

  function m3Forward(x, keep = []) {
    return tf.tidy(() => {
      const t = {}, left = { ...CONSUMERS }, kept = {};
      for (const n of GRAPH) {
        const a = n.inputs.map((i) => t[i]);
        let y;
        if (n.op === "input") y = x;
        else if (n.op === "conv") y = tf.conv2d(a[0], M3T[n.name + "/kernel"], n.strides, n.padding).add(M3T[n.name + "/bias"]);
        else if (n.op === "dw") y = tf.depthwiseConv2d(a[0], M3T[n.name + "/kernel"], n.strides, n.padding).add(M3T[n.name + "/bias"]);
        else if (n.op === "relu6") y = tf.relu6(a[0]);
        else if (n.op === "pad") y = tf.pad(a[0], [[0, 0], n.pad[0], n.pad[1], [0, 0]]);
        else y = tf.add(a[0], a[1]);
        t[n.name] = y;
        if (keep.includes(n.name)) kept[n.name] = y;
        for (const i of n.inputs) {
          left[i] -= 1;
          if (left[i] === 0 && !(i in kept) && t[i] !== x) t[i].dispose();
        }
      }
      const feats = t[LAST];
      const z = m3Head(feats);
      return { ...kept, feats, z, p: tf.softmax(z) };
    });
  }

  // Grad-CAM by the chain rule. After the last feature maps both models only pool and apply dense layers, so the
  // average gradient of each map is the gradient of the class probability with respect to that map's average
  // (through max pooling each pooled cell passes its gradient to one input cell, which keeps the sum the same).
  const softmaxGrad = (p, c) => p.map((v, m) => p[c] * ((m === c ? 1 : 0) - v));
  function camFrom(maps, channels, weights) {
    const n = maps.length / channels, cam = new Float32Array(n);
    let mx = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < channels; k++) s += maps[i * channels + k] * weights[k];
      cam[i] = s > 0 ? s : 0;
      if (cam[i] > mx) mx = cam[i];
    }
    if (mx > 0) for (let i = 0; i < n; i++) cam[i] /= mx;
    return cam;
  }
  function gradCamM3(feats, p) {
    const dz = softmaxGrad(p, argmax(p)), W = M3W["dense/kernel"].data, C = M3W["dense/kernel"].shape[0], w = new Float32Array(C);
    for (let k = 0; k < C; k++) for (let m = 0; m < 6; m++) w[k] += W[k * 6 + m] * dz[m];
    return camFrom(feats, C, w);
  }
  function gradCamM1(a3, d, p) {
    const dz = softmaxGrad(p, argmax(p)), W2 = M1W["dense_1/kernel"].data, W1 = M1W["dense/kernel"].data;
    const dd = new Float32Array(128), w = new Float32Array(128);
    for (let j = 0; j < 128; j++) if (d[j] > 0) for (let m = 0; m < 6; m++) dd[j] += W2[j * 6 + m] * dz[m];
    for (let k = 0; k < 128; k++) for (let j = 0; j < 128; j++) w[k] += W1[k * 128 + j] * dd[j];
    return camFrom(a3, 128, w);
  }

  const to224 = (x200) => tf.image.resizeBilinear(x200, [224, 224], false, true).tile([1, 1, 1, 3]).mul(2).sub(1);

  // ---------- images
  const IMAGES = META.images;
  const scratch = document.createElement("canvas");
  scratch.width = scratch.height = N;
  const sctx = scratch.getContext("2d", { willReadFrequently: true });
  function bitmapToGray(bmp) {
    sctx.clearRect(0, 0, N, N);
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(bmp, 0, 0, N, N);
    const d = sctx.getImageData(0, 0, N, N).data;
    const g = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const r = d[4 * i], gr = d[4 * i + 1], b = d[4 * i + 2];
      g[i] = (r === gr && gr === b ? r : Math.round(0.299 * r + 0.587 * gr + 0.114 * b)) / 255;
    }
    return g;
  }
  async function decodeTestImages() {
    await Promise.all(IMAGES.map(async (im) => {
      const blob = new Blob([Uint8Array.from(atob(im.img), (c) => c.charCodeAt(0))], { type: META.imageType });
      const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      im.gray = bitmapToGray(bmp);
      im.src = `data:${META.imageType};base64,${im.img}`;
      bmp.close();
    }));
  }

  // ---------- stress test
  const SLIDERS = [
    { key: "shift", label: "Brightness", min: -0.5, max: 0.5, step: 0.05, value: 0, fmt: (v) => (v ? (v > 0 ? "+" : "") + num(v) : "unchanged") },
    { key: "contrast", label: "Contrast", min: 0.2, max: 1.6, step: 0.05, value: 1, fmt: (v) => (v === 1 ? "unchanged" : "× " + v.toFixed(2)) },
    { key: "noise", label: "Random noise", min: 0, max: 0.15, step: 0.01, value: 0, fmt: (v) => (v ? "σ = " + v.toFixed(2) : "off") },
    { key: "blur", label: "Blur", min: 0, max: 4, step: 1, value: 0, fmt: (v) => (v ? `${2 * v + 1} × ${2 * v + 1} px` : "off") },
    { key: "rotate", label: "Rotation", min: -45, max: 45, step: 5, value: 0, fmt: (v) => (v ? v + "°" : "none") },
  ];
  const DEFAULTS = Object.fromEntries(SLIDERS.map((s) => [s.key, s.value]));

  function reflectIndex(i) {
    const p = 2 * N;
    i = ((i % p) + p) % p;
    return i < N ? i : p - 1 - i;
  }
  function rotate(src, deg) {
    const out = new Float32Array(N * N), a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), m = (N - 1) / 2;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const dx = x - m, dy = y - m, sx = c * dx - s * dy + m, sy = s * dx + c * dy + m;
        const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
        const xa = reflectIndex(x0), xb = reflectIndex(x0 + 1), ya = reflectIndex(y0), yb = reflectIndex(y0 + 1);
        out[y * N + x] = (src[ya * N + xa] * (1 - fx) + src[ya * N + xb] * fx) * (1 - fy) + (src[yb * N + xa] * (1 - fx) + src[yb * N + xb] * fx) * fy;
      }
    }
    return out;
  }
  // Same kernel as the robustness test in notebook 07, with zero padding like tf.nn.conv2d "SAME".
  function blur(src, k) {
    const sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8, r = (k - 1) / 2, w = [];
    for (let i = -r; i <= r; i++) w.push(Math.exp(-(i * i) / (2 * sigma * sigma)));
    const total = w.reduce((a, b) => a + b, 0);
    for (let i = 0; i < w.length; i++) w[i] /= total;
    const tmp = new Float32Array(N * N), out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < N) acc += src[y * N + xx] * w[i + r]; }
      tmp[y * N + x] = acc;
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < N) acc += tmp[yy * N + x] * w[i + r]; }
      out[y * N + x] = acc;
    }
    return out;
  }
  function seededNormal(seed) {
    let a = seed >>> 0;
    const uniform = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  }
  function applyStress(src, s, seed) {
    let a = src;
    const clip = (arr) => arr.map(clamp01);
    if (s.rotate) a = rotate(a, s.rotate);
    if (s.blur) a = clip(blur(a, 2 * s.blur + 1));
    if (s.contrast !== 1) {
      const mean = a.reduce((p, v) => p + v, 0) / a.length;
      a = a.map((v) => clamp01(mean + (v - mean) * s.contrast));
    }
    if (s.shift) a = a.map((v) => clamp01(v + s.shift));
    if (s.noise) {
      const rand = seededNormal(seed * 7919 + 17);
      a = a.map((v) => clamp01(v + s.noise * rand()));
    }
    return a === src ? src.slice() : a;
  }
  const isStressed = (s) => SLIDERS.some((sl) => s[sl.key] !== DEFAULTS[sl.key]);

  // ---------- state
  const state = {
    idx: 0, cls: 0, upload: null, stress: { ...DEFAULTS }, heatOn: false, heatModel: "m3",
    step: 0, pxScale: "255", px: [100, 100], conv: [100, 100], filter: null, stageModel: "m1", stage: { m1: 1, m3: 1 },
    softModel: "m3", camAlpha: 0.7, cmModel: "m3", cmCell: null, run: null,
  };
  let R = null, busy = false, dirty = false, scanTimer = null;

  // ---------- drawing helpers
  const offs = {};
  function offscreen(w, h, tag) {
    const key = `${tag}:${w}x${h}`;
    if (!offs[key]) { offs[key] = document.createElement("canvas"); offs[key].width = w; offs[key].height = h; }
    return offs[key];
  }
  function paint(canvas, w, h, fill, smooth, tag = "img", clear = true) {
    const off = offscreen(w, h, tag), octx = off.getContext("2d"), img = octx.createImageData(w, h);
    fill(img.data);
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext("2d");
    if (clear) ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }
  function drawGray(canvas, gray, smooth = true, n = N, dim = 1) {
    paint(canvas, n, n, (d) => {
      for (let i = 0; i < n * n; i++) {
        const v = Math.round(clamp01(gray[i]) * 255 * dim);
        d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = v;
        d[4 * i + 3] = 255;
      }
    }, smooth);
  }
  function drawHeat(canvas, cam, n, alpha, clear) {
    paint(canvas, n, n, (d) => {
      for (let i = 0; i < n * n; i++) {
        d[4 * i] = HEAT[0]; d[4 * i + 1] = HEAT[1]; d[4 * i + 2] = HEAT[2];
        d[4 * i + 3] = Math.round(255 * alpha * cam[i]);
      }
    }, true, "heat", clear);
  }
  function drawMap(canvas, data, h, w, c, ch, smooth, limit) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < h * w; i++) { const v = data[i * c + ch]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = hi - lo || 1;
    paint(canvas, w, h, (d) => {
      for (let i = 0; i < h * w; i++) {
        const col = limit !== undefined && i > limit ? [255, 255, 255] : navyRamp((data[i * c + ch] - lo) / span);
        d[4 * i] = col[0]; d[4 * i + 1] = col[1]; d[4 * i + 2] = col[2]; d[4 * i + 3] = 255;
      }
    }, smooth, "map");
    return [lo, hi];
  }

  // ---------- running the models on the current image
  const currentSource = () => (state.upload ? state.upload.gray : IMAGES[state.idx].gray);
  const truthIndex = () => (state.upload ? null : IMAGES[state.idx].label);

  function disposeResult(r) {
    if (!r) return;
    tf.dispose([r.x200, r.x224, ...Object.values(r.m1), ...Object.values(r.m3), ...Object.values(r.extra)]);
  }

  async function update() {
    if (busy) { dirty = true; return; }
    busy = true;
    try {
      do {
        dirty = false;
        const t0 = performance.now();
        const gray = applyStress(currentSource(), state.stress, state.upload ? 1 : state.idx + 1);
        const x200 = tf.tensor4d(gray, [1, N, N, 1]);
        const x224 = tf.tidy(() => to224(x200));
        const m1 = m1Forward(x200);
        const m3 = m3Forward(x224, M3_STAGE_KEYS);
        const [p1, p3, z1, z3, a3, d1, f3] = (await Promise.all([m1.p, m3.p, m1.z, m3.z, m1.a3, m1.d, m3.feats].map((t) => t.data())));
        const next = {
          gray, x200, x224, m1, m3, p1: Array.from(p1), p3: Array.from(p3), z1: Array.from(z1), z3: Array.from(z3), extra: {}, cache: {},
          cam1: gradCamM1(a3, d1, Array.from(p1)), cam1n: m1.a3.shape[1], cam3: gradCamM3(f3, Array.from(p3)), cam3n: m3.feats.shape[1],
          ms: performance.now() - t0,
        };
        disposeResult(R);
        R = next;
        await renderAll();
      } while (dirty);
    } catch (err) {
      status.textContent = "Something went wrong while running the models: " + err.message;
      console.error(err);
    }
    busy = false;
  }

  async function data(key, make) {
    if (!R.cache[key]) {
      const t = make();
      R.cache[key] = { data: await t.data(), shape: t.shape };
      if (!Object.values(R.m1).includes(t) && !Object.values(R.m3).includes(t)) R.extra[key] = t;
    }
    return R.cache[key];
  }

  async function renderAll() {
    renderView();
    renderCards();
    await renderStep();
    const where = backend === "webgl" ? "graphics card (WebGL)" : "processor";
    status.textContent = `Running in this browser on the ${where}. Self-check: same probabilities as our Keras models to within ${num(state.check, 3)}. Last image: ${Math.round(R.ms)} ms.${state.gpuNote || ""}`;
  }

  // ---------- workspace
  function renderClassList() {
    $("#class-list").innerHTML = CLASSES.map((c, i) =>
      `<button type="button" role="tab" data-cls="${i}" aria-selected="${i === state.cls}">${NAMES[i]} <span>45</span></button>`).join("");
    const rows = IMAGES.map((im, i) => [im, i]).filter(([im]) => im.label === state.cls);
    $("#thumbs").innerHTML = rows.map(([im, i]) => {
      const miss = im.m3[0] !== im.label;
      return `<button type="button" data-idx="${i}" aria-current="${!state.upload && i === state.idx}" title="${esc(im.file)}${miss ? ": M3 got this one wrong in our test run" : ""}">` +
        `<img src="${im.src}" alt="${esc(NAMES[im.label])} test image ${esc(im.file)}" loading="lazy">${miss ? '<span class="miss"></span>' : ""}</button>`;
    }).join("");
    const misses = rows.filter(([im]) => im.m3[0] !== im.label).length;
    $("#class-desc").textContent = DESCRIPTIONS[state.cls] + (misses ? ` A red dot marks the ${misses === 1 ? "image" : misses + " images"} M3 got wrong in our test run.` : "");
  }

  function renderView() {
    drawGray($("#view"), R.gray, true, N, state.heatOn ? 0.6 : 1);
    const heat = $("#view-heat"), ctx = heat.getContext("2d");
    ctx.clearRect(0, 0, heat.width, heat.height);
    if (state.heatOn) {
      const m = state.heatModel;
      drawHeat(heat, m === "m3" ? R.cam3 : R.cam1, m === "m3" ? R.cam3n : R.cam1n, 0.75, true);
    }
    const t = truthIndex();
    if (state.upload) {
      $("#image-title").textContent = "Your image";
      $("#truth").innerHTML = `${esc(state.upload.name)}. Its true class is unknown, so the models cannot be marked right or wrong.`;
    } else {
      const im = IMAGES[state.idx];
      $("#image-title").textContent = "Test image " + im.file;
      $("#truth").innerHTML = `True class: <b>${NAMES[t]}</b>${isStressed(state.stress) ? ". Changed by the stress test." : ""}`;
    }
  }

  function card(key, probs, final) {
    const k = argmax(probs), t = truthIndex(), m = MODELS[key];
    const tag = t === null ? "" : k === t ? '<span class="verdict-tag ok">✓ correct</span>' : '<span class="verdict-tag no">✗ wrong</span>';
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    return `<div class="card-head"><span class="card-name"><i class="sw" style="background:${m.color}"></i>${m.long}${final ? ", final model" : ""}</span>${tag}</div>
      <div class="card-pred">${NAMES[k]}</div>
      <div class="card-conf">${pct(probs[k])} probability</div>
      <div class="bars">${order.map(([p, i]) => `<div class="bar-row${i === k ? " top" : ""}"><span class="lab">${NAMES[i]}</span>
        <span class="bar-track"><span class="bar-fill" style="display:block;width:${(p * 100).toFixed(1)}%;background:${m.color}"></span></span>
        <span class="val">${pct(p)}</span></div>`).join("")}</div>`;
  }
  function renderCards() {
    $("#card-m3").innerHTML = card("m3", R.p3, true);
    $("#card-m1").innerHTML = card("m1", R.p1, false);
  }

  function buildSliders() {
    $("#sliders").innerHTML = SLIDERS.map((s) => `<div class="slider">
      <label for="s-${s.key}">${s.label}<output id="o-${s.key}">${s.fmt(state.stress[s.key])}</output></label>
      <input type="range" id="s-${s.key}" data-key="${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${state.stress[s.key]}"></div>`).join("");
  }
  function syncSliders() {
    for (const s of SLIDERS) {
      $(`#s-${s.key}`).value = state.stress[s.key];
      $(`#o-${s.key}`).textContent = s.fmt(state.stress[s.key]);
    }
  }

  function selectImage(i) {
    stopScan();
    state.idx = i;
    state.upload = null;
    state.cls = IMAGES[i].label;
    renderClassList();
    update();
  }

  // ---------- step 1: pixels
  function canvasPos(canvas, ev) {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * N), y = Math.floor(((ev.clientY - r.top) / r.height) * N);
    return [Math.max(0, Math.min(N - 1, y)), Math.max(0, Math.min(N - 1, x))];
  }
  function box(ctx, canvas, r, c, half, color, width) {
    const s = canvas.width / N;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.strokeRect((c - half) * s, (r - half) * s, (2 * half + 1) * s, (2 * half + 1) * s);
  }
  function renderPixels() {
    const cv = $("#px-canvas");
    drawGray(cv, R.gray, false);
    const [r, c] = state.px, ctx = cv.getContext("2d");
    box(ctx, cv, r, c, 4, "#ffffff", 3);
    box(ctx, cv, r, c, 4, "#0b84cf", 1.5);
    const scale = state.pxScale, cells = [];
    for (let i = -4; i <= 4; i++) for (let j = -4; j <= 4; j++) {
      const y = r + i, x = c + j;
      if (y < 0 || x < 0 || y >= N || x >= N) { cells.push('<div style="background:#f6f7f9;color:#9a9db0">·</div>'); continue; }
      const v = R.gray[y * N + x], g = Math.round(v * 255);
      const text = scale === "255" ? String(g) : scale === "1" ? v.toFixed(2) : num(v * 2 - 1);
      cells.push(`<div class="${i === 0 && j === 0 ? "centre" : ""}" style="background:rgb(${g},${g},${g});color:${g < 128 ? "#fff" : "#14162b"}">${text}</div>`);
    }
    $("#px-grid").innerHTML = cells.join("");
    $("#px-pos").textContent = `Pixels around row ${r + 1}, column ${c + 1} of 200. The outlined one is the centre.`;
  }

  // ---------- step 2: one filter
  function renderFilters() {
    const k = M1W["conv2d/kernel"].data;
    $("#filters").innerHTML = Array.from({ length: 32 }, (_, f) => {
      const w = Array.from({ length: 9 }, (_, i) => k[i * 32 + f]);
      const m = Math.max(...w.map(Math.abs)) || 1;
      return `<button type="button" data-filter="${f}" aria-pressed="${f === state.filter}" aria-label="Filter ${f + 1}">${w.map((v) => `<i style="background:${css(diverging(v / m))}"></i>`).join("")}</button>`;
    }).join("");
  }
  async function renderConv(limit) {
    const a1 = await data("m1:a1", () => R.m1.a1);
    if (state.filter === null) {
      // Start on the filter whose map varies most on this image, so the first map shown is not blank.
      let best = 0, bestSpread = -1;
      for (let ch = 0; ch < 32; ch++) {
        let s1 = 0, s2 = 0;
        for (let i = 0; i < N * N; i++) { const v = a1.data[i * 32 + ch]; s1 += v; s2 += v * v; }
        const spread = s2 / (N * N) - (s1 / (N * N)) ** 2;
        if (spread > bestSpread) { bestSpread = spread; best = ch; }
      }
      state.filter = best;
      pressed($("#filters"), "data-filter", String(best));
    }
    const f = state.filter, [r, c] = state.conv, k = M1W["conv2d/kernel"].data, bias = M1W["conv2d/bias"].data[f];
    const cv = $("#conv-canvas"), ctx = cv.getContext("2d");
    drawGray(cv, R.gray, true);
    box(ctx, cv, r, c, 1, "#ffffff", 3);
    box(ctx, cv, r, c, 1, "#0b84cf", 1.5);
    ctx.strokeStyle = "rgba(11,132,207,.55)";
    ctx.lineWidth = 1;
    const s = cv.width / N;
    ctx.beginPath();
    ctx.moveTo(0, (r + 0.5) * s); ctx.lineTo((c - 2) * s, (r + 0.5) * s); ctx.moveTo((c + 3) * s, (r + 0.5) * s); ctx.lineTo(cv.width, (r + 0.5) * s);
    ctx.moveTo((c + 0.5) * s, 0); ctx.lineTo((c + 0.5) * s, (r - 2) * s); ctx.moveTo((c + 0.5) * s, (r + 3) * s); ctx.lineTo((c + 0.5) * s, cv.height);
    ctx.stroke();

    const patch = [], weights = [], prods = [];
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
      const y = r + i, x = c + j;
      const v = y < 0 || x < 0 || y >= N || x >= N ? 0 : R.gray[y * N + x];
      const w = k[((i + 1) * 3 + (j + 1)) * 32 + f];
      patch.push(v); weights.push(w); prods.push(v * w);
    }
    const sum = prods.reduce((a, b) => a + b, 0), z = sum + bias, out = Math.max(0, z);
    const net = a1.data[(r * N + c) * 32 + f];
    const wMax = Math.max(...weights.map(Math.abs)) || 1, pMax = Math.max(...prods.map(Math.abs)) || 1;
    const matrix = (vals, color, d, cap) => `<div><div class="mat">${vals.map((v, i) => {
      const col = color(v, i);
      return `<div style="background:${css(col)};color:${inkOn(col)}">${num(v, d)}</div>`;
    }).join("")}</div><div class="mat-cap">${cap}</div></div>`;
    $("#conv-calc").innerHTML = `
      <div class="calc-row">
        ${matrix(patch, (v) => [Math.round(v * 255), Math.round(v * 255), Math.round(v * 255)], 2, "pixels under the filter")}
        <span class="op">×</span>
        ${matrix(weights, (v) => diverging(v / wMax), 2, `weights of filter ${f + 1}`)}
        <span class="op">=</span>
        ${matrix(prods, (v) => diverging(v / pMax), 3, "products")}
      </div>
      <div class="calc-lines">
        <div>Add the 9 products: <b>${num(sum, 3)}</b></div>
        <div>Add the bias ${num(bias, 3)}: <b>${num(z, 3)}</b></div>
        <div class="big">ReLU, max(0, ${num(z, 3)}) = ${num(out, 3)}</div>
        <div class="check-val">TensorFlow.js computed ${num(net, 3)} at this point of the feature map, the same number.</div>
      </div>`;
    const fm = $("#fmap-canvas");
    const [lo, hi] = drawMap(fm, a1.data, N, N, 32, f, false, limit);
    if (limit === undefined) {
      const fctx = fm.getContext("2d");
      box(fctx, fm, r, c, 1, "#f5a300", 2);
    }
    $("#fmap-note").textContent = `Feature map of filter ${f + 1} after ReLU: 200 × 200 values from ${num(lo)} (white) to ${num(hi)} (dark).`;
  }
  function stopScan() {
    if (scanTimer) { clearInterval(scanTimer); scanTimer = null; $("#scan-btn").textContent = "Slide the filter across the image"; }
  }
  function startScan() {
    if (scanTimer) { stopScan(); return; }
    const stride = 8;
    let pos = 0;
    const total = Math.ceil(N / stride) * Math.ceil(N / stride);
    $("#scan-btn").textContent = "Stop";
    scanTimer = setInterval(() => {
      if (pos >= total || !R) { stopScan(); renderConv(); return; }
      const r = Math.floor(pos / Math.ceil(N / stride)) * stride, c = (pos % Math.ceil(N / stride)) * stride;
      state.conv = [r, c];
      renderConv(r * N + c);
      pos += 1;
    }, 20);
  }

  // ---------- step 3: layer by layer
  const M1_STAGES = [
    { key: "in", name: "Input", params: 0, text: "The 200 × 200 grayscale image as 40,000 numbers between 0 and 1." },
    { key: "a1", name: "Conv 1 + ReLU", params: 320, text: "32 filters of 3 × 3 slide over the image, as in step 2. Each filter makes one feature map, so there are 32 maps of 200 × 200. The first filters react to simple things such as edges, bright spots and dark lines." },
    { key: "p1", name: "Max pool", params: 0, text: "Max pooling keeps the largest value in every 2 × 2 square. The maps shrink to half the width and height, so the next layer covers a wider area for the same work, and a small shift of the defect changes little. Nothing here is learned." },
    { key: "a2", name: "Conv 2 + ReLU", params: 18496, text: "64 filters, each 3 × 3 × 32. Every filter looks at all 32 maps of the layer before at once, so it can combine simple patterns into bigger ones, such as a network of cracks or a cluster of pits." },
    { key: "p2", name: "Max pool", params: 0, text: "Halves the maps again, to 50 × 50." },
    { key: "a3", name: "Conv 3 + ReLU", params: 73856, text: "128 filters of 3 × 3 × 64. Each map now responds to a larger part of the steel surface. Grad-CAM in step 5 uses this layer." },
    { key: "p3", name: "Max pool", params: 0, text: "Halves the maps to 25 × 25." },
    { key: "g", name: "Average", params: 0, text: "Global average pooling turns each of the 128 maps into one number, its average. The image is now described by 128 numbers, each saying how strongly one pattern appears anywhere in it." },
    { key: "d", name: "Dense + ReLU", params: 16512, text: "A fully connected layer: each of its 128 outputs is a weighted sum of all 128 inputs plus a bias, then ReLU. In training a dropout layer switched off half of these outputs at random, so the model could not lean on a few of them. When predicting, dropout does nothing." },
    { key: "p", name: "Dense + softmax", params: 774, text: "The last layer turns the 128 numbers into 6 scores, one per class, and softmax turns the scores into probabilities (step 4)." },
  ];
  const M3_STAGES = [
    { key: "in", name: "Input", text: "224 × 224 with three identical channels, scaled to −1 to 1, because MobileNetV2 was pretrained on colour photos of that size. One channel is shown." },
    { key: "Conv1_relu", name: "First conv", text: "32 filters with a stride of 2, so the image is halved to 112 × 112 straight away. ReLU6 keeps values between 0 and 6." },
    { key: "expanded_conv_project", name: "Block 1", text: "MobileNetV2 is built from 17 blocks. Each block widens the maps with a 1 × 1 convolution, filters each map on its own with a cheap 3 × 3 depthwise convolution and narrows them again with another 1 × 1 convolution. That is why it is small and fast." },
    { key: "block_2_add", name: "Block 3", text: "When a block keeps the same shape, its input is added to its output. These shortcuts let the signal pass through a deep network, which makes it easier to train." },
    { key: "block_5_add", name: "Block 6", text: "The maps keep shrinking while their number grows: the network trades position detail for more kinds of pattern." },
    { key: "block_12_add", name: "Block 13", text: "Batch normalisation follows every convolution. For this page it was folded into the convolution weights, which gives the same result with fewer steps." },
    { key: "block_16_project", name: "Block 17", text: "320 maps of 7 × 7. Each position now stands for a 32 × 32 pixel area of the input." },
    { key: "out_relu", name: "Last conv", text: "A final 1 × 1 convolution makes 1,280 maps of 7 × 7. Grad-CAM in step 5 uses these maps." },
    { key: "g", name: "Average", text: "Global average pooling: one number per map, so 1,280 numbers describe the image." },
    { key: "p", name: "Dense + softmax", text: "The layer we added: 1,280 × 6 weights and 6 biases (7,686 learned numbers) turn the 1,280 numbers into 6 class scores. The layers before it started from the ImageNet weights, and part of them were then fine-tuned on steel." },
  ];

  async function stageTensor(model, key) {
    if (model === "m1") {
      if (key === "in") return data("m1:in", () => R.x200);
      return data("m1:" + key, () => R.m1[key]);
    }
    if (key === "in") return data("m3:in", () => tf.tidy(() => R.x224.slice([0, 0, 0, 0], [1, 224, 224, 1])));
    if (key === "g") return data("m3:g", () => tf.mean(R.m3.feats, [1, 2]));
    if (key === "p") return data("m3:p", () => R.m3.p);
    return data("m3:" + key, () => R.m3[key]);
  }
  function stageShape(model, key) {
    if (model === "m1") return (key === "in" ? R.x200 : R.m1[key]).shape;
    if (key === "in") return [1, 224, 224, 3];
    if (key === "g") return [1, R.m3.feats.shape[3]];
    return R.m3[key].shape;
  }
  const shapeText = (shape) => shape.slice(1).join(" × ");

  async function renderStages() {
    const model = state.stageModel, stages = model === "m1" ? M1_STAGES : M3_STAGES;
    const shapes = stages.map((s) => stageShape(model, s.key));
    const sel = state.stage[model];
    $("#pipeline").innerHTML = stages.map((s, i) => {
      const sh = shapes[i], spatial = sh.length === 4 ? sh[1] : 0, chans = sh[sh.length - 1];
      const hgt = spatial ? Math.round(10 + 64 * (spatial / 224)) : 6, wid = Math.round(6 + 5 * Math.log2(chans + 1));
      return (i ? '<span class="pipe-arrow" aria-hidden="true">→</span>' : "") +
        `<button type="button" role="tab" class="pipe-step" data-stage="${i}" aria-selected="${i === sel}">
          <span class="pipe-box" style="width:${wid}px;height:${hgt}px"></span>
          <span class="pipe-name">${s.name}</span><span class="pipe-shape">${shapeText(sh)}</span></button>`;
    }).join("");
    const st = stages[sel], t = await stageTensor(model, st.key), maps = $("#stage-maps");
    const sh = t.shape;
    maps.className = "stage-maps";
    if (st.key === "in") {
      maps.classList.add("single");
      maps.innerHTML = '<canvas width="300" height="300"></canvas>';
      const n = sh[1], arr = model === "m1" ? t.data : t.data.map((v) => (v + 1) / 2);
      drawGray(maps.firstChild, arr, false, n);
    } else if (sh.length === 4) {
      const [, h, w, c] = sh, means = [];
      for (let ch = 0; ch < c; ch++) {
        let s = 0;
        for (let i = 0; i < h * w; i++) s += t.data[i * c + ch];
        means.push([s / (h * w), ch]);
      }
      means.sort((a, b) => b[0] - a[0]);
      const show = means.slice(0, 32);
      maps.innerHTML = show.map(([, ch]) => `<canvas width="${Math.min(w, 100)}" height="${Math.min(h, 100)}" title="map ${ch + 1}"></canvas>`).join("");
      $$("canvas", maps).forEach((cv, i) => drawMap(cv, t.data, h, w, c, show[i][1], w > 20));
    } else if (st.key === "p") {
      maps.classList.add("vector");
      const probs = Array.from(t.data);
      maps.innerHTML = card(model, probs, model === "m3");
    } else {
      maps.classList.add("vector");
      const vals = Array.from(t.data), mx = Math.max(...vals) || 1;
      maps.innerHTML = `<canvas class="vec" width="640" height="160" aria-label="${vals.length} numbers"></canvas><p class="note">${vals.length} numbers, one bar each, in their natural order. Taller means larger.</p>`;
      const cv = maps.firstChild, ctx = cv.getContext("2d"), bw = cv.width / vals.length;
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.fillStyle = "#191e6b";
      vals.forEach((v, i) => { const hh = Math.max(0, v / mx) * (cv.height - 4); ctx.fillRect(i * bw, cv.height - hh, Math.max(bw - (bw > 3 ? 1 : 0), 0.6), hh); });
      ctx.fillStyle = "#d6d9e1"; ctx.fillRect(0, cv.height - 1, cv.width, 1);
    }
    const params = model === "m1" ? `<dt>Learned numbers</dt><dd>${st.params.toLocaleString("en-US")}</dd>` : "";
    const shown = sh.length === 4 && st.key !== "in" ? `<dt>Shown</dt><dd>${sh[3] <= 32 ? `all ${sh[3]} maps, most active first` : `the 32 most active of ${sh[3]} maps`}</dd>` : "";
    $("#stage-explain").innerHTML = `<h3>${st.name}</h3><p>${st.text}</p><dl class="facts"><dt>Output shape</dt><dd>${shapeText(sh)}</dd>${params}${shown}</dl>` +
      (model === "m1" ? '<p class="note">M1 has 109,958 learned numbers in total.</p>' : '<p class="note">M3 has 2,265,670 learned numbers in total.</p>');
  }

  // ---------- step 4: softmax
  function renderSoftmax() {
    const model = state.softModel, z = model === "m3" ? R.z3 : R.z1, p = model === "m3" ? R.p3 : R.p1;
    const zmax = Math.max(...z), raw = z.every((v) => v < 30);
    const show = (v) => {
      if (v > 0 && (v >= 1e7 || v < 1e-3)) {
        const [m, e] = v.toExponential(2).split("e");
        return `${m} × 10<sup>${Number(e) < 0 ? "−" + Math.abs(Number(e)) : Number(e)}</sup>`;
      }
      return v >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 1 }) : v.toFixed(3);
    };
    const ezShown = z.map((v) => Math.exp(raw ? v : v - zmax));
    const sumShown = ezShown.reduce((a, b) => a + b, 0), k = argmax(p);
    $("#soft-table").innerHTML = `<thead><tr><th>Class</th><th>Score z</th><th>${raw ? "e<sup>z</sup>" : "e<sup>z − max</sup>"}</th><th>Probability</th></tr></thead>
      <tbody>${z.map((v, i) => `<tr class="${i === k ? "top" : ""}"><td>${NAMES[i]}</td><td>${num(v, 2)}</td><td>${show(ezShown[i])}</td>
        <td><span class="pbar" style="width:${Math.round(p[i] * 90)}px"></span>${pct(p[i])}</td></tr>`).join("")}</tbody>
      <tfoot><tr><td>Sum</td><td></td><td>${show(sumShown)}</td><td>100.0 %</td></tr></tfoot>`;
    const t = truthIndex();
    const lossLine = t === null ? "" : `<p>During training the model was scored with the <strong>cross-entropy loss</strong>: minus the log of the probability it gave the true class. For this image that is −log(${p[t].toFixed(3)}) = <strong>${(-Math.log(Math.max(p[t], 1e-12))).toFixed(3)}</strong>. Training nudged every learned number, a little at a time, to make this loss small on the training images.</p>`;
    $("#soft-explain").innerHTML = `<p>The last layer gives one <strong>score</strong> per class (also called a logit). Scores can be any number, positive or negative, and only their differences matter.</p>
      <p><strong>Softmax</strong> turns them into probabilities that are all positive and add up to 1: raise e to the power of each score, then divide by the sum.</p>
      <div class="formula">p(${NAMES[k]}) = ${show(ezShown[k])} / ${show(sumShown)} = <strong>${p[k].toFixed(3)}</strong></div>
      ${raw ? "" : "<p class=\"note\">The scores are large, so the largest score was subtracted first. That gives the same probabilities and keeps the numbers readable.</p>"}
      ${lossLine}`;
  }

  // ---------- step 5: Grad-CAM
  function renderCams() {
    for (const key of ["m3", "m1"]) {
      const cv = $("#cam-" + key), probs = key === "m3" ? R.p3 : R.p1, k = argmax(probs);
      drawGray(cv, R.gray, true, N, 0.6);
      drawHeat(cv, key === "m3" ? R.cam3 : R.cam1, key === "m3" ? R.cam3n : R.cam1n, state.camAlpha, false);
      const t = truthIndex();
      $(`#cam-${key}-cap`).innerHTML = `<b>${MODELS[key].long}</b>: ${NAMES[k]}, ${pct(probs[k])}${t === null ? "" : k === t ? ', <span class="verdict-tag ok">✓ correct</span>' : ', <span class="verdict-tag no">✗ wrong</span>'}<br><span class="note">Heat map from ${key === "m3" ? "1,280 maps of 7 × 7" : "128 maps of 50 × 50"}</span>`;
    }
  }

  async function renderStep() {
    if (!R) return;
    if (state.step === 0) renderPixels();
    else if (state.step === 1) await renderConv();
    else if (state.step === 2) await renderStages();
    else if (state.step === 3) renderSoftmax();
    else renderCams();
  }

  // ---------- results
  function renderAccuracy() {
    const rows = ["m1", "m2", "m3"].map((k) => META.metrics.find((m) => m.model.startsWith(k)));
    $("#acc-chart").innerHTML = rows.map((m) => {
      const key = m.model.slice(0, 2), col = MODELS[key].color;
      return `<div class="acc-row"><span class="who"><i class="sw" style="background:${col}"></i>${MODELS[key].long}</span>
        <div class="track" title="${m.errors} of 270 test images wrong"><div class="fill" style="width:${(m.accuracy * 100).toFixed(2)}%;background:${col}"></div>
        <span class="num">${(m.accuracy * 100).toFixed(2)} %, ${m.errors} wrong</span></div></div>`;
    }).join("") + '<div class="acc-axis"><span></span><div><span>0 %</span><span>50 %</span><span>100 %</span></div></div>';
  }

  function renderCM() {
    const model = state.cmModel, preds = state.run ? state.run.preds[model] : IMAGES.map((im) => im[model][0]);
    const cm = CLASSES.map(() => CLASSES.map(() => []));
    IMAGES.forEach((im, i) => cm[im.label][preds[i]].push(i));
    const wrong = IMAGES.filter((im, i) => preds[i] !== im.label).length;
    let html = `<div class="cm" style="grid-template-columns: 96px repeat(6, minmax(34px, 1fr))"><div class="h cm-axis">true ↓ / predicted →</div>` +
      SHORT.map((s) => `<div class="h">${s}</div>`).join("");
    cm.forEach((row, t) => {
      html += `<div class="h rh">${SHORT[t]}</div>`;
      row.forEach((cell, p) => {
        const n = cell.length, diag = t === p;
        const bg = n === 0 ? "#f6f7f9" : diag ? BLUES[Math.min(BLUES.length - 1, Math.round((n / 45) * (BLUES.length - 1)))] : "#f9dcdc";
        const fg = n === 0 ? "#9a9db0" : diag ? inkOn(rgb(bg)) : "#a52323";
        const pressed = state.cmCell && state.cmCell[0] === t && state.cmCell[1] === p;
        html += `<button type="button" data-cell="${t},${p}" aria-pressed="${!!pressed}" style="background:${bg};color:${fg}"
          aria-label="${n} ${NAMES[t]} images predicted as ${NAMES[p]}">${n}</button>`;
      });
    });
    $("#cm").innerHTML = html + "</div>";
    $("#cm-title").textContent = `Confusion matrix: ${MODELS[model].long}`;
    const source = state.run ? `the run in this browser just now${state.run.stressed ? ", with the stress-test changes" : ""}` : "our saved Keras test run";
    $("#cm-note").textContent = `${270 - wrong} of 270 right, ${wrong} wrong, from ${source}. Rows are the true class and columns the predicted class, so the diagonal holds the right answers. Click a square to see its images.`;
    renderPicks(cm);
  }
  function renderPicks(cm) {
    const box = $("#cm-picks");
    if (!state.cmCell) { box.innerHTML = ""; return; }
    const [t, p] = state.cmCell, ids = cm[t][p];
    box.innerHTML = ids.slice(0, 24).map((i) => `<button type="button" data-idx="${i}" title="${esc(IMAGES[i].file)}"><img src="${IMAGES[i].src}" alt="${esc(IMAGES[i].file)}"></button>`).join("") +
      (ids.length > 24 ? `<span class="note">and ${ids.length - 24} more</span>` : "") +
      (ids.length ? "" : '<span class="note">No images in this square.</span>');
  }

  async function runAll() {
    const btn = $("#run-btn"), out = $("#run-out"), bar = $("#run-bar");
    const stressed = $("#run-stress").checked && isStressed(state.stress);
    btn.disabled = true;
    out.textContent = "Running…";
    const preds = { m1: [], m3: [] }, conf = { m1: [], m3: [] }, batch = 10, t0 = performance.now();
    for (let s = 0; s < IMAGES.length; s += batch) {
      const part = IMAGES.slice(s, s + batch);
      const flat = new Float32Array(part.length * N * N);
      part.forEach((im, j) => flat.set(stressed ? applyStress(im.gray, state.stress, s + j + 1) : im.gray, j * N * N));
      const [p1, p3] = tf.tidy(() => {
        const x = tf.tensor4d(flat, [part.length, N, N, 1]);
        return [m1Forward(x).p, m3Forward(to224(x)).p];
      });
      const [a1, a3] = await Promise.all([p1.array(), p3.array()]);
      tf.dispose([p1, p3]);
      for (const [key, arr] of [["m1", a1], ["m3", a3]]) for (const row of arr) { const k = argmax(row); preds[key].push(k); conf[key].push(row[k]); }
      bar.style.width = `${(Math.min(s + batch, IMAGES.length) / IMAGES.length) * 100}%`;
      out.textContent = `Running… ${Math.min(s + batch, IMAGES.length)} of 270 images`;
      await tf.nextFrame();
    }
    const secs = (performance.now() - t0) / 1000;
    state.run = { preds, stressed };
    const rows = ["m3", "m1"].map((key) => {
      const right = preds[key].filter((k, i) => k === IMAGES[i].label).length;
      const same = preds[key].filter((k, i) => k === IMAGES[i][key][0]).length;
      const diff = Math.max(...conf[key].map((c, i) => (preds[key][i] === IMAGES[i][key][0] ? Math.abs(c - IMAGES[i][key][1]) : 0)));
      return `<tr><td><i class="sw" style="background:${MODELS[key].color}"></i> ${MODELS[key].long}</td><td>${right} / 270</td><td>${pct(right / 270, 2)}</td><td>${same} / 270${stressed ? "" : `, probabilities within ${num(diff, 3)}`}</td></tr>`;
    }).join("");
    out.innerHTML = `<table class="run-table"><thead><tr><th>Model</th><th>Right</th><th>Accuracy</th><th>Same answer as saved Keras run</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="note">${stressed ? "With the stress-test changes applied to every image. " : ""}Took ${secs.toFixed(1)} s for 540 predictions.${stressed ? "" : " The small probability differences come from rounding M3's weights to 16-bit numbers to make this page smaller."}</p>`;
    btn.disabled = false;
    renderCM();
  }

  const ROB_AXIS = {
    darker: { title: "Brightness lowered by", fmt: (v) => (v ? num(v, 2) : "0") },
    noise: { title: "Noise strength (σ)", fmt: (v) => (v ? v.toFixed(2) : "0") },
    blur: { title: "Blur kernel", fmt: (v) => (v ? `${v} × ${v}` : "none") },
    low_contrast: { title: "Contrast kept", fmt: (v) => (v ? "× " + v.toFixed(1) : "× 1.0") },
  };
  function renderRobustness() {
    const kind = $("#rob-kind").value, axis = ROB_AXIS[kind];
    const W = 560, H = 290, L = 46, Rm = 92, T = 12, B = 44, pw = W - L - Rm, ph = H - T - B;
    const rows = META.robustness.filter((r) => r.corruption === kind);
    const levels = [0, 1, 2, 3].map((s) => rows.find((r) => r.severity === s).level);
    const x = (s) => L + (s / 3) * pw, y = (a) => T + (1 - a) * ph;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation accuracy of M1, M2 and M3 as the image gets ${kind.replace("_", " ")}">`;
    svg += '<g class="grid">' + [0, 0.25, 0.5, 0.75, 1].map((a) => `<line x1="${L}" x2="${L + pw}" y1="${y(a)}" y2="${y(a)}"></line>`).join("") + "</g>";
    svg += '<g class="axis">' + [0, 0.25, 0.5, 0.75, 1].map((a) => `<text x="${L - 8}" y="${y(a) + 4}" text-anchor="end">${a * 100} %</text>`).join("");
    svg += levels.map((lv, s) => `<text x="${x(s)}" y="${T + ph + 18}" text-anchor="middle">${axis.fmt(lv)}</text>`).join("") + "</g>";
    svg += `<text class="axis-t" x="${L + pw / 2}" y="${H - 4}" text-anchor="middle">${axis.title}</text>`;
    svg += `<line x1="${L}" x2="${L + pw}" y1="${y(1 / 6)}" y2="${y(1 / 6)}" stroke="#9a9db0" stroke-dasharray="4 4"></line><text class="axis-t" x="${L + 4}" y="${y(1 / 6) - 5}">random guessing, 16.7 %</text>`;
    const ends = [];
    for (const key of ["m1", "m2", "m3"]) {
      const pts = [0, 1, 2, 3].map((s) => rows.find((r) => r.severity === s && r.model.startsWith(key)).accuracy);
      const col = MODELS[key].color;
      svg += `<polyline fill="none" stroke="${col}" stroke-width="2" points="${pts.map((a, s) => `${x(s)},${y(a)}`).join(" ")}"></polyline>`;
      pts.forEach((a, s) => {
        svg += `<g class="pt" data-kind="${kind}" data-level="${levels[s]}" data-tip="${MODELS[key].long}, ${axis.title.toLowerCase()} ${axis.fmt(levels[s])}: ${pct(a)}" tabindex="0" role="button" aria-label="${MODELS[key].long} ${axis.fmt(levels[s])} ${pct(a)}. Apply this change">
          <circle cx="${x(s)}" cy="${y(a)}" r="12" fill="transparent"></circle><circle cx="${x(s)}" cy="${y(a)}" r="5" fill="${col}" stroke="#fff" stroke-width="2"></circle></g>`;
      });
      ends.push([key, y(pts[3])]);
    }
    ends.sort((a, b) => a[1] - b[1]);
    for (let i = 1; i < ends.length; i++) if (ends[i][1] - ends[i - 1][1] < 15) ends[i][1] = ends[i - 1][1] + 15;
    for (const [key, yy] of ends) svg += `<text class="series-label" x="${L + pw + 10}" y="${yy + 4}">${MODELS[key].short} ${pct(rows.find((r) => r.severity === 3 && r.model.startsWith(key)).accuracy, 0)}</text>`;
    svg += "</svg>";
    $("#rob-chart").innerHTML = `<div class="chart">${svg}</div><div class="legend">${["m1", "m2", "m3"].map((k) => `<span><i class="sw" style="background:${MODELS[k].color}"></i>${MODELS[k].long}</span>`).join("")}</div>`;
    $("#rob-title").textContent = "Robustness: accuracy as the image gets worse";
  }
  function applyRobustness(kind, level) {
    state.stress = { ...DEFAULTS };
    if (kind === "darker") state.stress.shift = -level;
    else if (kind === "noise") state.stress.noise = level;
    else if (kind === "blur") state.stress.blur = level ? (level - 1) / 2 : 0;
    else state.stress.contrast = level || 1;
    syncSliders();
    $("#stress").open = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
    update();
  }

  // ---------- tooltip
  const tip = $("#tip");
  document.addEventListener("pointermove", (ev) => {
    const el = ev.target.closest && ev.target.closest("[data-tip]");
    if (!el) { tip.hidden = true; return; }
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 270) + "px";
    tip.style.top = ev.clientY + 14 + "px";
  });

  // ---------- events
  function pressed(group, attr, value) {
    $$(`[${attr}]`, group).forEach((b) => b.setAttribute("aria-pressed", String(b.getAttribute(attr) === value)));
  }
  $("#class-list").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-cls]");
    if (!b) return;
    state.cls = Number(b.dataset.cls);
    renderClassList();
  });
  $("#thumbs").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-idx]");
    if (b) selectImage(Number(b.dataset.idx));
  });
  $("#random-btn").addEventListener("click", () => selectImage(Math.floor(Math.random() * IMAGES.length)));
  $("#upload").addEventListener("change", async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const bmp = await createImageBitmap(file);
      state.upload = { gray: bitmapToGray(bmp), name: file.name };
      bmp.close();
      renderClassList();
      update();
    } catch (err) {
      status.textContent = "That file could not be read as an image. Try a JPG or PNG.";
    }
    ev.target.value = "";
  });
  $("#sliders").addEventListener("input", (ev) => {
    const key = ev.target.dataset.key;
    if (!key) return;
    state.stress[key] = Number(ev.target.value);
    syncSliders();
    update();
  });
  $("#reset-btn").addEventListener("click", () => { state.stress = { ...DEFAULTS }; syncSliders(); update(); });
  $("#heat-on").addEventListener("change", (ev) => { state.heatOn = ev.target.checked; renderView(); });
  $(".heat-row .seg").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-heat]");
    if (!b) return;
    state.heatModel = b.dataset.heat;
    pressed(b.parentElement, "data-heat", state.heatModel);
    state.heatOn = true;
    $("#heat-on").checked = true;
    renderView();
  });

  function showStep(i) {
    stopScan();
    state.step = i;
    $$("#step-tabs [data-step]").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.step) === i)));
    $$(".step-panel").forEach((p) => { p.hidden = Number(p.dataset.panel) !== i; });
    renderStep();
  }
  $("#step-tabs").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-step]");
    if (b) showStep(Number(b.dataset.step));
  });
  $("#step-tabs").addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
    const i = (state.step + (ev.key === "ArrowRight" ? 1 : 4)) % 5;
    showStep(i);
    $(`#step-tabs [data-step="${i}"]`).focus();
  });

  function hoverCanvas(canvas, key, render) {
    canvas.tabIndex = 0;
    canvas.addEventListener("pointermove", (ev) => {
      if (key === "conv") stopScan();
      state[key] = canvasPos(canvas, ev);
      if (R) render();
    });
    canvas.addEventListener("keydown", (ev) => {
      const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
      const m = moves[ev.key];
      if (!m) return;
      ev.preventDefault();
      const step = ev.shiftKey ? 10 : 1;
      state[key] = [Math.max(0, Math.min(N - 1, state[key][0] + m[0] * step)), Math.max(0, Math.min(N - 1, state[key][1] + m[1] * step))];
      if (R) render();
    });
  }
  hoverCanvas($("#px-canvas"), "px", renderPixels);
  hoverCanvas($("#conv-canvas"), "conv", () => renderConv());
  $("#px-scale").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-scale]");
    if (!b) return;
    state.pxScale = b.dataset.scale;
    pressed(b.parentElement, "data-scale", state.pxScale);
    renderPixels();
  });
  $("#filters").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-filter]");
    if (!b) return;
    state.filter = Number(b.dataset.filter);
    pressed($("#filters"), "data-filter", String(state.filter));
    if (!scanTimer) renderConv();
  });
  $("#scan-btn").addEventListener("click", startScan);
  $("#stage-model").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-model]");
    if (!b) return;
    state.stageModel = b.dataset.model;
    pressed(b.parentElement, "data-model", state.stageModel);
    renderStages();
  });
  $("#pipeline").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-stage]");
    if (!b) return;
    state.stage[state.stageModel] = Number(b.dataset.stage);
    renderStages();
  });
  $("#soft-model").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-model]");
    if (!b) return;
    state.softModel = b.dataset.model;
    pressed(b.parentElement, "data-model", state.softModel);
    renderSoftmax();
  });
  $("#cam-alpha").addEventListener("input", (ev) => { state.camAlpha = Number(ev.target.value); renderCams(); });
  $("#cm-model").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-model]");
    if (!b) return;
    state.cmModel = b.dataset.model;
    pressed(b.parentElement, "data-model", state.cmModel);
    renderCM();
  });
  $("#cm").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-cell]");
    if (!b) return;
    const cell = b.dataset.cell.split(",").map(Number);
    state.cmCell = state.cmCell && state.cmCell[0] === cell[0] && state.cmCell[1] === cell[1] ? null : cell;
    renderCM();
  });
  $("#cm-picks").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-idx]");
    if (!b) return;
    selectImage(Number(b.dataset.idx));
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  $("#run-btn").addEventListener("click", runAll);
  $("#rob-kind").addEventListener("change", renderRobustness);
  const robClick = (ev) => {
    const g = ev.target.closest(".pt");
    if (g) applyRobustness(g.dataset.kind, Number(g.dataset.level));
  };
  $("#rob-chart").addEventListener("click", robClick);
  $("#rob-chart").addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); robClick(ev); } });

  // Runs both models on reference test images and compares with the probabilities Keras gave in Python.
  async function selfCheck() {
    const ref = META.check, flat = new Float32Array(ref.idx.length * N * N);
    ref.idx.forEach((i, j) => flat.set(IMAGES[i].gray, j * N * N));
    const [a, b] = tf.tidy(() => {
      const x = tf.tensor4d(flat, [ref.idx.length, N, N, 1]);
      return [m1Forward(x).p, m3Forward(to224(x)).p];
    });
    const p1 = await a.array(), p3 = await b.array();
    tf.dispose([a, b]);
    const diff = (p, q) => Math.max(...p.flatMap((row, i) => row.map((v, j) => Math.abs(v - q[i][j]))));
    return { m1: diff(p1, ref.m1), m3: diff(p3, ref.m3) };
  }

  // ---------- start
  $("#defect-list").innerHTML = NAMES.map((n, i) => `<li><b>${n}.</b> ${DESCRIPTIONS[i]}</li>`).join("");
  buildSliders();
  renderFilters();
  renderAccuracy();
  renderRobustness();
  status.textContent = "Preparing the 270 test images…";
  await decodeTestImages();
  const first = IMAGES.findIndex((im) => im.file === "patches_110.jpg");
  state.idx = first >= 0 ? first : 0;
  state.cls = IMAGES[state.idx].label;
  renderClassList();
  renderCM();
  showStep(0);
  status.textContent = "Checking that this browser computes the same numbers as our Python models…";
  const off = (c) => !(c.m1 <= 0.01 && c.m3 <= 0.05);
  let check = await selfCheck();
  if (backend === "webgl" && off(check)) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    check = await selfCheck();
  }
  if (backend === "webgl" && off(check)) {
    state.gpuNote = ` The graphics card was off by ${num(Math.max(check.m1, check.m3), 3)}, so the processor is used instead.`;
    await tf.setBackend("cpu");
    backend = "cpu";
    makeTensors();
    check = await selfCheck();
  }
  state.check = Math.max(check.m1, check.m3);
  await update();
})().catch((err) => {
  const s = document.getElementById("status");
  if (s) s.textContent = "The demo could not start: " + err.message;
  console.error(err);
});
