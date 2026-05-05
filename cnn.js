(() => {
  // ─── Config ────────────────────────────────────────────────
  const MODEL_URL =
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json";
  const INPUT_SIZE = 224;
  const NUM_TILES_PER_LAYER = 12;

  const LAYER_NAMES = ["conv_pw_1_relu", "conv_pw_5_relu", "conv_pw_11_relu"];
  const STORAGE_PRESET = "cnn:preset";

  const LAYER_INFO = [
    {
      title: "Layer 1 · Edges & Orientations",
      desc: "Earliest filters fire on simple structure: edges, corners, and color gradients.",
    },
    {
      title: "Layer 2 · Textures & Patterns",
      desc: "Mid-level filters combine edges into textures and repeating patterns.",
    },
    {
      title: "Layer 3 · Parts & Shapes",
      desc: "Deeper filters respond to object parts and abstract shapes.",
    },
  ];

  const PRESETS = {
    portrait:
      "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=400&h=400&fit=crop&auto=format",
    dog: "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=400&h=400&fit=crop&auto=format",
    building:
      "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=400&h=400&fit=crop&auto=format",
    texture:
      "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&h=400&fit=crop&auto=format",
  };

  const VIRIDIS = [
    [68, 1, 84],
    [72, 35, 116],
    [64, 67, 135],
    [52, 94, 141],
    [41, 120, 142],
    [32, 144, 140],
    [34, 167, 132],
    [68, 190, 112],
    [121, 209, 81],
    [189, 222, 38],
    [253, 231, 37],
  ];

  function viridis(t) {
    if (!isFinite(t)) return [0, 0, 0];
    t = Math.max(0, Math.min(1, t));
    const seg = t * (VIRIDIS.length - 1);
    const i = Math.floor(seg);
    if (i >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];
    const f = seg - i;
    const a = VIRIDIS[i],
      b = VIRIDIS[i + 1];
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

  const layerData = [null, null, null];
  const layerShape = [null, null, null];
  const layerChannels = [null, null, null];

  // Drawing state
  let isCnnDrawing = false;
  let cnnDrawMode = false;
  let drawTimer = null;

  // ─── DOM refs ──────────────────────────────────────────────
  const $loading = () => document.getElementById("cnn-loading");
  const $error = () => document.getElementById("cnn-error");
  const $app = () => document.getElementById("cnn-app");
  const $inputCanvas = () => document.getElementById("cnn-input-canvas");

  // ─── Init ──────────────────────────────────────────────────
  async function initCNN() {
    if (inited) return;
    inited = true;

    try {
      const baseModel = await tf.loadLayersModel(MODEL_URL);
      const outputs = LAYER_NAMES.map(
        (name) => baseModel.getLayer(name).output,
      );
      activationModel = tf.model({ inputs: baseModel.inputs, outputs });

      tf.tidy(() => {
        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        const outs = activationModel.predict(dummy);
        const arr = Array.isArray(outs) ? outs : [outs];
        arr.forEach((t) => t.dispose());
      });

      $loading().hidden = true;
      $app().hidden = false;

      wireControls();

      const savedPreset = localStorage.getItem(STORAGE_PRESET) || "dog";
      await loadPreset(savedPreset);
    } catch (err) {
      console.error("CNN init failed:", err);
      const detailEl = document.getElementById("cnn-error-msg");
      if (detailEl)
        detailEl.textContent = err && err.message ? err.message : String(err);
      $loading().hidden = true;
      $error().hidden = false;
    }
  }

  // ─── Controls ──────────────────────────────────────────────
  function wireControls() {
    document.getElementById("cnn-upload").addEventListener("change", onUpload);
    document.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        cnnDrawMode = false;
        document.getElementById("cnn-draw-btn").classList.remove("active");
        document.getElementById("cnn-clear-draw-btn").hidden = true;
        loadPreset(btn.dataset.preset);
      });
    });

    // Drawing Controls
    const drawBtn = document.getElementById("cnn-draw-btn");
    const clearDrawBtn = document.getElementById("cnn-clear-draw-btn");
    const cnnCanvas = $inputCanvas();
    const cnnCtx = cnnCanvas.getContext("2d");

    drawBtn.addEventListener("click", () => {
      cnnDrawMode = !cnnDrawMode;
      drawBtn.classList.toggle("active", cnnDrawMode);
      clearDrawBtn.hidden = !cnnDrawMode;

      if (cnnDrawMode) {
        cnnCtx.fillStyle = "#000";
        cnnCtx.fillRect(0, 0, cnnCanvas.width, cnnCanvas.height);
        setActivePreset(null);
        runInference();
      }
    });

    clearDrawBtn.addEventListener("click", () => {
      cnnCtx.fillStyle = "#000";
      cnnCtx.fillRect(0, 0, cnnCanvas.width, cnnCanvas.height);
      runInference();
    });

    function getCnnPos(e) {
      const rect = cnnCanvas.getBoundingClientRect();
      const t = (e.touches && e.touches[0]) || e;
      const x = (t.clientX - rect.left) * (cnnCanvas.width / rect.width);
      const y = (t.clientY - rect.top) * (cnnCanvas.height / rect.height);
      return { x, y };
    }

    function startDraw(e) {
      if (!cnnDrawMode) return;
      e.preventDefault();
      isCnnDrawing = true;
      const pos = getCnnPos(e);
      cnnCtx.beginPath();
      cnnCtx.moveTo(pos.x, pos.y);
    }

    function moveDraw(e) {
      if (!isCnnDrawing || !cnnDrawMode) return;
      e.preventDefault();
      const pos = getCnnPos(e);
      cnnCtx.lineTo(pos.x, pos.y);
      cnnCtx.strokeStyle = "#fff";
      cnnCtx.lineWidth = 12;
      cnnCtx.lineCap = "round";
      cnnCtx.lineJoin = "round";
      cnnCtx.stroke();

      clearTimeout(drawTimer);
      drawTimer = setTimeout(() => runInference(), 100);
    }

    function endDraw() {
      if (isCnnDrawing) {
        isCnnDrawing = false;
        runInference();
      }
    }

    cnnCanvas.addEventListener("mousedown", startDraw);
    cnnCanvas.addEventListener("mousemove", moveDraw);
    window.addEventListener("mouseup", endDraw);
    cnnCanvas.addEventListener("touchstart", startDraw, { passive: false });
    cnnCanvas.addEventListener("touchmove", moveDraw, { passive: false });
    window.addEventListener("touchend", endDraw);

    // Modal opacity
    document
      .getElementById("cnn-overlay-opacity")
      .addEventListener("input", (e) => {
        document.getElementById("cnn-modal-canvas").style.opacity =
          e.target.value;
      });
  }

  (function wireModal() {
    const close = document.getElementById("cnn-modal-close");
    const bg = document.getElementById("cnn-modal-bg");
    if (close) close.addEventListener("click", closeModal);
    if (bg) bg.addEventListener("click", closeModal);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeModal();
    });
  })();

  function setActivePreset(name) {
    document.querySelectorAll(".preset-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.preset === name);
    });
  }

  async function loadPreset(name) {
    const url = PRESETS[name];
    if (!url) return;
    setActivePreset(name);
    try {
      localStorage.setItem(STORAGE_PRESET, name);
    } catch (_) {}

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = async () => {
      drawInputImage(img);
      await runInference();
    };
    img.src = url;
  }

  function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setActivePreset(null);
    cnnDrawMode = false;
    document.getElementById("cnn-draw-btn").classList.remove("active");
    document.getElementById("cnn-clear-draw-btn").hidden = true;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      drawInputImage(img);
      URL.revokeObjectURL(url);
      e.target.value = "";
      await runInference();
    };
    img.src = url;
  }

  function drawInputImage(img) {
    const canvas = $inputCanvas();
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(
      canvas.width / img.width,
      canvas.height / img.height,
    );
    const w = img.width * scale;
    const h = img.height * scale;
    const x = (canvas.width - w) / 2;
    const y = (canvas.height - h) / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, x, y, w, h);
  }

  // ─── Inference ─────────────────────────────────────────────
  async function runInference() {
    if (!activationModel || inferenceInFlight) return;
    inferenceInFlight = true;
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
      console.error("Inference failed:", err);
      inferenceInFlight = false;
      return;
    }

    try {
      for (let i = 0; i < outs.length; i++) {
        const data = await outs[i].data();
        layerData[i] = data;
        layerShape[i] = outs[i].shape.slice();
      }
      for (let i = 0; i < outs.length; i++) {
        renderLayer(i);
      }
    } finally {
      outs.forEach((t) => t.dispose());
      inferenceInFlight = false;
    }
  }

  // ─── Rendering ─────────────────────────────────────────────
  function pickChannelsByVariance(data, shape, count) {
    const [, H, W, C] = shape;
    const HW = H * W;
    const stats = new Array(C);
    for (let ch = 0; ch < C; ch++) {
      let sum = 0,
        sum2 = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = data[y * W * C + x * C + ch];
          sum += v;
          sum2 += v * v;
        }
      }
      const mean = sum / HW;
      const variance = sum2 / HW - mean * mean;
      stats[ch] = { ch, variance };
    }
    stats.sort((a, b) => b.variance - a.variance);
    const picked = stats.slice(0, Math.min(count, C)).map((s) => s.ch);
    picked.sort((a, b) => a - b);
    return picked;
  }

  function renderActivationToCanvas(canvas, data, shape, ch) {
    const [, H, W, C] = shape;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");

    let mn = Infinity,
      mx = -Infinity;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = data[y * W * C + x * C + ch];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    const range = mx - mn || 1;

    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = data[y * W * C + x * C + ch];
        const t = (v - mn) / range;
        const [r, g, b] = viridis(t);
        const i = (y * W + x) * 4;
        img.data[i] = r;
        img.data[i + 1] = g;
        img.data[i + 2] = b;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function renderLayer(idx) {
    const data = layerData[idx];
    const shape = layerShape[idx];
    if (!data || !shape) return;
    const grid = document.getElementById(`layer-${idx}-grid`);
    grid.innerHTML = "";

    const channels = pickChannelsByVariance(data, shape, NUM_TILES_PER_LAYER);
    layerChannels[idx] = channels;

    for (const ch of channels) {
      const canvas = document.createElement("canvas");
      canvas.className = "activation-map";
      canvas.title = `Channel ${ch} — click to enlarge`;
      renderActivationToCanvas(canvas, data, shape, ch);
      canvas.addEventListener("click", () => openModal(idx, ch));
      grid.appendChild(canvas);
    }
  }

  // ─── Modal ─────────────────────────────────────────────────
  function openModal(layerIdx, ch) {
    const data = layerData[layerIdx];
    const shape = layerShape[layerIdx];
    if (!data || !shape) return;

    const canvas = document.getElementById("cnn-modal-canvas");
    renderActivationToCanvas(canvas, data, shape, ch);

    const inputCanvas = $inputCanvas();
    const modalInput = document.getElementById("cnn-modal-input");
    modalInput.width = inputCanvas.width;
    modalInput.height = inputCanvas.height;
    const ctx = modalInput.getContext("2d");
    ctx.drawImage(inputCanvas, 0, 0);

    const info = LAYER_INFO[layerIdx];
    document.getElementById("cnn-modal-title").textContent =
      `${info.title} — Channel ${ch}`;
    document.getElementById("cnn-modal-desc").textContent = info.desc;

    document.getElementById("cnn-modal").hidden = false;
  }

  function closeModal() {
    const modal = document.getElementById("cnn-modal");
    if (modal) modal.hidden = true;
  }

  window.initCNN = initCNN;
})();
