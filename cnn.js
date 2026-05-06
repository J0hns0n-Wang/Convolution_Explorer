(() => {
  // ─── Config ────────────────────────────────────────────────
  const MODEL_URL  = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
  const LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt';
  const INPUT_SIZE = 224;
  let numTiles = 12;

  // Keras layer names verified against the model.json.
  const LAYER_NAMES = ['conv_pw_1_relu', 'conv_pw_5_relu', 'conv_pw_11_relu'];
  const STORAGE_PRESET = 'cnn:preset';

  // Receptive field params: stride from 224×224 input, RF size in input pixels.
  // Computed by tracing conv+depthwise kernel sizes and strides through MobileNet v1.
  const RF_PARAMS = [
    { stride: 2,  rf: 7   }, // Layer 1: 112×112 feature map
    { stride: 8,  rf: 43  }, // Layer 2: 28×28  feature map
    { stride: 32, rf: 219 }, // Layer 3: 7×7    feature map
  ];

  const LAYER_INFO = [
    {
      title: 'Layer 1 · Edges & Orientations',
      desc: 'The first learned layer responds to simple local structure: edges at various angles, color transitions, and corners. Each filter\'s receptive field covers only ~7 px of the original image — it is nearly blind to context.',
      rf: 7,
    },
    {
      title: 'Layer 2 · Textures & Patterns',
      desc: 'Mid-network filters combine many edge detectors into textures — grids, curves, stripes, spots. Each now "sees" a ~43 px window, so it can reason about local geometry.',
      rf: 43,
    },
    {
      title: 'Layer 3 · Parts & Shapes',
      desc: 'Deep filters respond to object parts and abstract shapes assembled from many textures. Each filter\'s receptive field spans ~219 px — nearly the full image — letting it recognise whole structures.',
      rf: 219,
    },
  ];

  const PRESETS = {
    portrait: 'https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=400&h=400&fit=crop&auto=format',
    dog:      'https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=400&h=400&fit=crop&auto=format',
    building: 'https://images.unsplash.com/photo-1486325212027-8081e485255e?w=400&h=400&fit=crop&auto=format',
    texture:  'https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&h=400&fit=crop&auto=format',
  };

  // ─── Viridis colormap (11 sampled stops, lerped) ───────────
  const VIRIDIS = [
    [68,1,84],[72,35,116],[64,67,135],[52,94,141],
    [41,120,142],[32,144,140],[34,167,132],[68,190,112],
    [121,209,81],[189,222,38],[253,231,37],
  ];
  function viridis(t) {
    if (!isFinite(t)) return [0, 0, 0];
    t = Math.max(0, Math.min(1, t));
    const seg = t * (VIRIDIS.length - 1);
    const i = Math.floor(seg);
    if (i >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];
    const f = seg - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }

  // ─── State ─────────────────────────────────────────────────
  let activationModel = null;
  let inited = false;
  let inferenceInFlight = false;
  let pendingInference = false;
  let imagenetLabels = [];
  let lastPredData = null; // re-render predictions when labels finally load

  const layerData     = [null, null, null]; // Float32Array per layer
  const layerShape    = [null, null, null]; // [1, H, W, C] per layer
  const layerChannels = [null, null, null]; // selected channel indices per layer

  let revealedLayers = 0;

  // Draw mode
  let cnnDrawMode = false;
  let isDrawingCNN = false;
  let cnnBaseImageData = null;
  let cnnDrawDebounce = null;
  const CNN_BRUSH_RADIUS = 14;

  // Modal nav
  let modalLayerIdx = -1;
  let modalChannelPos = 0;
  let modalInputSnapshot = null;

  // ─── DOM refs ──────────────────────────────────────────────
  const $loading     = () => document.getElementById('cnn-loading');
  const $error       = () => document.getElementById('cnn-error');
  const $app         = () => document.getElementById('cnn-app');
  const $inputCanvas = () => document.getElementById('cnn-input-canvas');
  const $rfCanvas    = () => document.getElementById('cnn-rf-canvas');

  // ─── ImageNet labels ───────────────────────────────────────
  async function fetchLabels() {
    try {
      const res = await fetch(LABELS_URL);
      const text = await res.text();
      const lines = text.trim().split('\n');
      // First line is "background"; the next 1000 are the ImageNet classes.
      imagenetLabels = lines.slice(1, 1001).map(l => l.trim());
      // If predictions already arrived before labels, render them now.
      if (lastPredData) renderPredictions(lastPredData);
    } catch (err) {
      console.warn('Could not load ImageNet labels:', err);
    }
  }

  // ─── Init ──────────────────────────────────────────────────
  async function initCNN() {
    if (inited) return;
    inited = true;

    fetchLabels(); // non-blocking; labels arrive whenever the fetch completes

    try {
      const baseModel = await tf.loadLayersModel(MODEL_URL);
      const layerOutputs = LAYER_NAMES.map(name => baseModel.getLayer(name).output);
      // Include the model's own softmax output so we can show top-5 predictions.
      const predOutput = baseModel.outputs[0]; // shape [1, 1000]
      activationModel = tf.model({ inputs: baseModel.inputs, outputs: [...layerOutputs, predOutput] });

      // Warmup so first real call is fast.
      tf.tidy(() => {
        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        const outs = activationModel.predict(dummy);
        (Array.isArray(outs) ? outs : [outs]).forEach(t => t.dispose());
      });

      $loading().hidden = true;
      $app().hidden = false;

      buildLayerLocks();
      addTilesControl();
      wireControls();

      const savedPreset = localStorage.getItem(STORAGE_PRESET) || 'dog';
      await loadPreset(savedPreset);
    } catch (err) {
      console.error('CNN init failed:', err);
      const el = document.getElementById('cnn-error-msg');
      if (el) el.textContent = err && err.message ? err.message : String(err);
      $loading().hidden = true;
      $error().hidden = false;
    }
  }

  // ─── Step-by-step layer reveals ────────────────────────────
  function buildLayerLocks() {
    for (let i = 1; i < 3; i++) {
      const section = document.querySelector(`.layer-section[data-layer="${i}"]`);
      if (!section) continue;
      section.style.position = 'relative';

      const overlay = document.createElement('div');
      overlay.className = 'layer-lock';
      overlay.id = `layer-lock-${i}`;
      overlay.innerHTML =
        '<div class="layer-lock-inner">' +
        '<div class="layer-lock-icon">🔒</div>' +
        `<p class="layer-lock-msg">Study Layer ${i} above — what patterns do the active channels respond to? Then unlock the next depth.</p>` +
        `<button class="layer-unlock-btn">Reveal Layer ${i + 1} →</button>` +
        '</div>';
      overlay.querySelector('.layer-unlock-btn').addEventListener('click', () => revealLayer(i));
      section.appendChild(overlay);
    }
  }

  function revealLayer(idx) {
    if (idx > revealedLayers) revealedLayers = idx;
    const overlay = document.getElementById(`layer-lock-${idx}`);
    if (overlay) {
      overlay.classList.add('unlocking');
      setTimeout(() => overlay.remove(), 350);
    }
  }

  // ─── Tiles per layer control ───────────────────────────────
  function addTilesControl() {
    const flow = $app() && $app().querySelector('.layers-flow');
    if (!flow) return;

    const ctrl = document.createElement('div');
    ctrl.className = 'tiles-control';
    ctrl.innerHTML =
      '<span class="tiles-label">Channels shown per layer:</span>' +
      [8, 12, 16, 24].map(n =>
        `<button class="tiles-btn${n === numTiles ? ' active' : ''}" data-count="${n}">${n}</button>`
      ).join('');

    ctrl.querySelectorAll('.tiles-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        numTiles = parseInt(btn.dataset.count, 10);
        ctrl.querySelectorAll('.tiles-btn').forEach(b =>
          b.classList.toggle('active', b === btn)
        );
        for (let i = 0; i < 3; i++) renderLayer(i);
      });
    });

    flow.before(ctrl);
  }

  // ─── Controls ──────────────────────────────────────────────
  function wireControls() {
    document.getElementById('cnn-upload').addEventListener('change', onUpload);
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => loadPreset(btn.dataset.preset));
    });
    wireCNNDrawing();
    wireOverlaySlider();
  }

  // ─── Draw mode ─────────────────────────────────────────────
  function setCNNDrawMode(active) {
    cnnDrawMode = active;
    const canvas   = $inputCanvas();
    const drawBtn  = document.getElementById('cnn-draw-btn');
    const clearBtn = document.getElementById('cnn-clear-draw-btn');
    if (canvas)   canvas.classList.toggle('cnn-draw-active', active);
    if (drawBtn) {
      drawBtn.textContent = active ? '✏ Stop Drawing' : '✏ Draw';
      drawBtn.classList.toggle('active', active);
    }
    if (clearBtn) clearBtn.hidden = !active;
  }

  function paintCNNAt(clientX, clientY, erase) {
    const canvas = $inputCanvas();
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) * (canvas.width  / rect.width);
    const y = (clientY - rect.top)  * (canvas.height / rect.height);
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.arc(x, y, CNN_BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = erase ? '#000' : '#fff';
    ctx.fill();
    clearTimeout(cnnDrawDebounce);
    cnnDrawDebounce = setTimeout(runInference, 400);
  }

  function wireCNNDrawing() {
    const canvas = $inputCanvas();
    if (!canvas) return;

    canvas.addEventListener('mousedown', e => {
      if (!cnnDrawMode) return;
      e.preventDefault();
      isDrawingCNN = true;
      paintCNNAt(e.clientX, e.clientY, e.button === 2 || e.ctrlKey);
    });
    canvas.addEventListener('mousemove', e => {
      if (!isDrawingCNN || !cnnDrawMode) return;
      e.preventDefault();
      paintCNNAt(e.clientX, e.clientY, e.button === 2 || e.ctrlKey);
    });
    window.addEventListener('mouseup', () => { isDrawingCNN = false; });
    canvas.addEventListener('contextmenu', e => { if (cnnDrawMode) e.preventDefault(); });

    canvas.addEventListener('touchstart', e => {
      if (!cnnDrawMode) return;
      e.preventDefault();
      isDrawingCNN = true;
      paintCNNAt(e.touches[0].clientX, e.touches[0].clientY, false);
    }, { passive: false });
    canvas.addEventListener('touchmove', e => {
      if (!isDrawingCNN || !cnnDrawMode) return;
      e.preventDefault();
      paintCNNAt(e.touches[0].clientX, e.touches[0].clientY, false);
    }, { passive: false });
    window.addEventListener('touchend', () => { isDrawingCNN = false; });

    const drawBtn = document.getElementById('cnn-draw-btn');
    if (drawBtn) drawBtn.addEventListener('click', () => setCNNDrawMode(!cnnDrawMode));

    const clearBtn = document.getElementById('cnn-clear-draw-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        if (cnnBaseImageData) {
          $inputCanvas().getContext('2d').putImageData(cnnBaseImageData, 0, 0);
          runInference();
        }
      });
    }
  }

  // ─── Receptive field overlay ───────────────────────────────
  // Shows a dim vignette over the input image with a bright cutout showing
  // exactly which input region a hovered activation tile can "see".
  function showRFHighlight(layerIdx, ax, ay) {
    const rfCanvas = $rfCanvas();
    if (!rfCanvas) return;
    const { stride, rf } = RF_PARAMS[layerIdx];
    const cx   = (ax + 0.5) * stride;
    const cy   = (ay + 0.5) * stride;
    const half = rf / 2;
    const x1 = Math.max(0, cx - half);
    const y1 = Math.max(0, cy - half);
    const x2 = Math.min(INPUT_SIZE, cx + half);
    const y2 = Math.min(INPUT_SIZE, cy + half);

    const ctx = rfCanvas.getContext('2d');
    ctx.clearRect(0, 0, rfCanvas.width, rfCanvas.height);

    // Dim the whole image.
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, 0, rfCanvas.width, rfCanvas.height);

    // Punch a transparent window over the receptive field region.
    ctx.clearRect(x1, y1, x2 - x1, y2 - y1);

    // Highlight border around the RF window.
    ctx.strokeStyle = 'rgba(255, 220, 50, 0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  function clearRFHighlight() {
    const rfCanvas = $rfCanvas();
    if (!rfCanvas) return;
    rfCanvas.getContext('2d').clearRect(0, 0, rfCanvas.width, rfCanvas.height);
  }

  // ─── Wire modal (runs immediately so it's always dismissable) ─
  (function wireModal() {
    const close = document.getElementById('cnn-modal-close');
    const bg    = document.getElementById('cnn-modal-bg');
    const prev  = document.getElementById('cnn-modal-prev');
    const next  = document.getElementById('cnn-modal-next');

    if (close) close.addEventListener('click', closeModal);
    if (bg)    bg.addEventListener('click', closeModal);
    if (prev)  prev.addEventListener('click', () => modalNavigate(-1));
    if (next)  next.addEventListener('click', () => modalNavigate(1));

    document.addEventListener('keydown', e => {
      const modal = document.getElementById('cnn-modal');
      if (!modal || modal.hidden) return;
      if      (e.key === 'Escape')     closeModal();
      else if (e.key === 'ArrowLeft')  { e.preventDefault(); modalNavigate(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); modalNavigate(1);  }
    });
  })();

  // ─── Overlay slider (Image ↔ Activation blend in modal) ────
  function wireOverlaySlider() {
    const slider  = document.getElementById('cnn-overlay-opacity');
    const overlay = document.getElementById('cnn-modal-canvas');
    if (slider && overlay) {
      slider.addEventListener('input', () => { overlay.style.opacity = slider.value; });
    }
  }

  // ─── Preset / upload ───────────────────────────────────────
  function setActivePreset(name) {
    document.querySelectorAll('.preset-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.preset === name);
    });
  }

  async function loadPreset(name) {
    const url = PRESETS[name];
    if (!url) return;
    setActivePreset(name);
    try { localStorage.setItem(STORAGE_PRESET, name); } catch (_) {}
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = async () => { drawInputImage(img); await runInference(); };
    img.onerror = () => console.error('Failed to load preset:', url);
    img.src = url;
  }

  function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setActivePreset(null);
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      drawInputImage(img);
      URL.revokeObjectURL(url);
      e.target.value = '';
      await runInference();
    };
    img.src = url;
  }

  function drawInputImage(img) {
    const canvas = $inputCanvas();
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(canvas.width / img.width, canvas.height / img.height);
    const w = img.width * scale, h = img.height * scale;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    cnnBaseImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  // ─── Inference ─────────────────────────────────────────────
  async function runInference() {
    if (!activationModel) return;
    if (inferenceInFlight) { pendingInference = true; return; }
    inferenceInFlight = true;
    pendingInference = false;

    const canvas = $inputCanvas();
    let outs;
    try {
      const input = tf.tidy(() => {
        const t = tf.browser.fromPixels(canvas);
        const resized = tf.image.resizeBilinear(t, [INPUT_SIZE, INPUT_SIZE]);
        return resized.toFloat().div(127.5).sub(1).expandDims(0);
      });
      outs = activationModel.predict(input);
      if (!Array.isArray(outs)) outs = [outs];
      input.dispose();
    } catch (err) {
      console.error('Inference failed:', err);
      inferenceInFlight = false;
      return;
    }

    try {
      // outs[0..2] = layer activation maps; outs[3] = softmax predictions
      for (let i = 0; i < 3; i++) {
        layerData[i]  = await outs[i].data();
        layerShape[i] = outs[i].shape.slice();
      }
      const predData = await outs[3].data();

      for (let i = 0; i < 3; i++) renderLayer(i);
      renderPredictions(predData);
    } finally {
      outs.forEach(t => t.dispose());
      inferenceInFlight = false;
      if (pendingInference) runInference();
    }
  }

  // ─── Predictions panel ─────────────────────────────────────
  function renderPredictions(predData) {
    lastPredData = predData;
    const el = document.getElementById('cnn-pred-list');
    if (!el) return;

    if (!imagenetLabels.length) {
      el.innerHTML = '<p class="pred-placeholder">Labels loading…</p>';
      return;
    }

    const indexed = Array.from(predData).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const top5 = indexed.slice(0, 5);

    el.innerHTML = top5.map(({ v, i }) => {
      const label    = imagenetLabels[i] || `class ${i}`;
      const pct      = (v * 100).toFixed(1);
      const barWidth = Math.min(100, v * 100).toFixed(1);
      return `<div class="pred-row">
        <div class="pred-row-top">
          <span class="pred-label" title="${label}">${label}</span>
          <span class="pred-pct">${pct}%</span>
        </div>
        <div class="pred-bar-track">
          <div class="pred-bar-fill" style="width:${barWidth}%"></div>
        </div>
      </div>`;
    }).join('');
  }

  // ─── Channel variance selection ────────────────────────────
  function pickChannelsByVariance(data, shape, count) {
    const [, H, W, C] = shape;
    const HW = H * W;
    const stats = [];
    for (let ch = 0; ch < C; ch++) {
      let sum = 0, sum2 = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = data[y * W * C + x * C + ch];
        sum += v; sum2 += v * v;
      }
      const mean = sum / HW;
      stats.push({ ch, variance: sum2 / HW - mean * mean });
    }
    stats.sort((a, b) => b.variance - a.variance);
    return stats.slice(0, Math.min(count, C)).map(s => s.ch).sort((a, b) => a - b);
  }

  // ─── Render one activation channel → canvas ────────────────
  function renderActivationToCanvas(canvas, data, shape, ch) {
    const [, H, W, C] = shape;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');

    let mn = Infinity, mx = -Infinity;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = data[y * W * C + x * C + ch];
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const range = (mx - mn) || 1;

    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const [r, g, b] = viridis((data[y * W * C + x * C + ch] - mn) / range);
      const idx = (y * W + x) * 4;
      img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b; img.data[idx+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return { min: mn, max: mx };
  }

  // ─── Layer-level stats ─────────────────────────────────────
  function computeActiveChannels(data, shape) {
    const [, H, W, C] = shape;
    let active = 0;
    outer: for (let ch = 0; ch < C; ch++) {
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (data[y * W * C + x * C + ch] > 0) { active++; continue outer; }
      }
    }
    return active;
  }

  function updateLayerStats(idx, data, shape) {
    const [, H, W, C] = shape;
    const info = LAYER_INFO[idx];
    const active = computeActiveChannels(data, shape);
    const pct = Math.round(100 * active / C);

    let el = document.getElementById(`layer-${idx}-stats`);
    if (!el) {
      el = document.createElement('div');
      el.id = `layer-${idx}-stats`;
      el.className = 'layer-stats';
      const grid = document.getElementById(`layer-${idx}-grid`);
      if (grid) grid.before(el);
    }
    el.innerHTML =
      `<span class="stat-chip" title="Output spatial resolution of this feature map">📐 ${H}×${W}</span>` +
      `<span class="stat-chip" title="Number of learned filters (channels) at this depth">🧠 ${C} channels</span>` +
      `<span class="stat-chip" title="How large a region of the original 224×224 image each filter can see">👁 RF ≈ ${info.rf}px</span>` +
      `<span class="stat-chip${pct < 50 ? ' stat-warn' : ''}" ` +
        `title="${active}/${C} channels have at least one non-zero pixel (ReLU keeps only positive activations)">` +
        `⚡ ${pct}% active</span>`;
  }

  // ─── Render layer tile grid ─────────────────────────────────
  function renderLayer(idx) {
    const data  = layerData[idx];
    const shape = layerShape[idx];
    if (!data || !shape) return;

    const grid = document.getElementById(`layer-${idx}-grid`);
    grid.innerHTML = '';

    const channels = pickChannelsByVariance(data, shape, numTiles);
    layerChannels[idx] = channels;

    updateLayerStats(idx, data, shape);

    for (const ch of channels) {
      const wrapper = document.createElement('div');
      wrapper.className = 'tile-wrapper';

      const canvas = document.createElement('canvas');
      canvas.className = 'activation-map';
      const { min, max } = renderActivationToCanvas(canvas, data, shape, ch);
      canvas.title = `Ch. ${ch}  ·  range ${min.toFixed(2)} → ${max.toFixed(2)}  ·  hover to see receptive field · click to enlarge`;

      // RF hover: dim the input image and show a cutout for the RF region.
      canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        const ax = Math.floor((e.clientX - rect.left) / rect.width  * canvas.width);
        const ay = Math.floor((e.clientY - rect.top)  / rect.height * canvas.height);
        showRFHighlight(idx, ax, ay);
      });
      canvas.addEventListener('mouseleave', clearRFHighlight);

      canvas.addEventListener('click', () => openModal(idx, ch));

      const label = document.createElement('div');
      label.className = 'tile-label';
      label.textContent = `Ch. ${ch}`;

      wrapper.appendChild(canvas);
      wrapper.appendChild(label);
      grid.appendChild(wrapper);
    }
  }

  // ─── Modal ─────────────────────────────────────────────────
  function openModal(layerIdx, ch) {
    const data  = layerData[layerIdx];
    const shape = layerShape[layerIdx];
    if (!data || !shape) return;

    modalLayerIdx = layerIdx;
    const channels = layerChannels[layerIdx] || [];
    const pos = channels.indexOf(ch);
    modalChannelPos = pos >= 0 ? pos : 0;

    const inputC = $inputCanvas();
    modalInputSnapshot = inputC
      ? inputC.getContext('2d').getImageData(0, 0, inputC.width, inputC.height)
      : null;

    renderModalContent();
    document.getElementById('cnn-modal').hidden = false;
  }

  function renderModalContent() {
    if (modalLayerIdx < 0) return;
    const channels = layerChannels[modalLayerIdx] || [];
    const ch = channels[modalChannelPos];
    if (ch === undefined) return;

    const data  = layerData[modalLayerIdx];
    const shape = layerShape[modalLayerIdx];
    if (!data || !shape) return;

    const overlayCanvas = document.getElementById('cnn-modal-canvas');
    const { min, max } = renderActivationToCanvas(overlayCanvas, data, shape, ch);
    const slider = document.getElementById('cnn-overlay-opacity');
    if (slider) overlayCanvas.style.opacity = slider.value;

    const underlayCanvas = document.getElementById('cnn-modal-input');
    if (underlayCanvas && modalInputSnapshot) {
      underlayCanvas.width  = modalInputSnapshot.width;
      underlayCanvas.height = modalInputSnapshot.height;
      underlayCanvas.getContext('2d').putImageData(modalInputSnapshot, 0, 0);
    }

    const [, H, W, C] = shape;
    let activePx = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      if (data[y * W * C + x * C + ch] > 0) activePx++;
    const activePct = Math.round(100 * activePx / (H * W));

    const info = LAYER_INFO[modalLayerIdx];
    document.getElementById('cnn-modal-title').textContent =
      `${info.title} — Channel ${ch}`;
    document.getElementById('cnn-modal-desc').innerHTML =
      `${info.desc}` +
      `<span class="modal-stats">` +
      `Activation range: ${min.toFixed(3)} → ${max.toFixed(3)}&nbsp;&nbsp;·&nbsp;&nbsp;` +
      `${activePct}% of spatial positions active (non-zero after ReLU)` +
      `</span>`;

    const prevBtn = document.getElementById('cnn-modal-prev');
    const nextBtn = document.getElementById('cnn-modal-next');
    const counter = document.getElementById('cnn-modal-counter');
    if (prevBtn) prevBtn.disabled = modalChannelPos === 0;
    if (nextBtn) nextBtn.disabled = modalChannelPos >= channels.length - 1;
    if (counter) counter.textContent = `${modalChannelPos + 1} / ${channels.length}`;
  }

  function modalNavigate(dir) {
    const channels = layerChannels[modalLayerIdx] || [];
    const newPos = modalChannelPos + dir;
    if (newPos < 0 || newPos >= channels.length) return;
    modalChannelPos = newPos;
    renderModalContent();
  }

  function closeModal() {
    const modal = document.getElementById('cnn-modal');
    if (modal) modal.hidden = true;
    clearRFHighlight();
  }

  // ─── Expose to app.js for lazy init on tab switch ──────────
  window.initCNN = initCNN;
})();
