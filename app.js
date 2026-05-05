// ─── Tabs ────────────────────────────────────────────────────
(() => {
  const STORAGE_TAB = 'explorer:tab';
  const buttons  = document.querySelectorAll('.tab-btn');
  const contents = document.querySelectorAll('.tab-content');

  function activateTab(target) {
    buttons.forEach(b => {
      const on = b.dataset.tab === target;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    contents.forEach(c => {
      const on = c.id === `tab-${target}`;
      c.classList.toggle('active', on);
      c.hidden = !on;
    });
    try { localStorage.setItem(STORAGE_TAB, target); } catch (_) {}
    if (target === 'cnn' && typeof window.initCNN === 'function') {
      window.initCNN();
    }
  }

  buttons.forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });

  const saved = (() => { try { return localStorage.getItem(STORAGE_TAB); } catch (_) { return null; } })();
  if (saved === 'cnn' || saved === 'conv') activateTab(saved);
})();

// ─── Convolution Explorer (Tab 1) ────────────────────────────
(() => {
  // ─── Constants ─────────────────────────────────────────────
  const N     = 16;   // drawable grid size
  const KSIZE = 3;    // kernel size (fixed 3×3)
  const CSIZE = 320;  // canvas CSS pixel size

  // ─── State ─────────────────────────────────────────────────
  const input  = new Float32Array(N * N);
  let   output  = new Float32Array(N * N);  // resized when outputSize changes
  let   computed = new Uint8Array(N * N);

  let strideS  = 1;  // stride S
  let paddingP = 0;  // zero-padding P

  let currentFilter  = 'edge';
  let isDrawing      = false;
  let drawValue      = 1;
  let brushIntensity = 1;

  let animPlaying = false;
  let animIndex   = 0;
  let animTimer   = null;
  let animSpeedMs = 80;

  // ─── Persistence ───────────────────────────────────────────
  const STORAGE = {
    FILTER:  'conv:filter',
    BRUSH:   'conv:brush',
    INPUT:   'conv:input',
    STRIDE:  'conv:stride',
    PADDING: 'conv:padding',
  };

  let saveScheduled = false;
  function saveInputSoon() {
    if (saveScheduled) return;
    saveScheduled = true;
    requestAnimationFrame(() => {
      saveScheduled = false;
      try { localStorage.setItem(STORAGE.INPUT, JSON.stringify(Array.from(input))); } catch (_) {}
    });
  }
  function loadInput() {
    try {
      const raw = localStorage.getItem(STORAGE.INPUT);
      if (!raw) return false;
      const arr = JSON.parse(raw);
      if (!Array.isArray(arr) || arr.length !== N * N) return false;
      for (let i = 0; i < N * N; i++) input[i] = arr[i];
      return true;
    } catch (_) { return false; }
  }

  // ─── Dimension helpers ─────────────────────────────────────
  // Cell size of the input canvas (which shows a padded grid).
  function inputCellSize() { return CSIZE / (N + 2 * paddingP); }
  // Number of output pixels along each axis.
  function outputSize()    { return Math.max(1, Math.floor((N + 2 * paddingP - KSIZE) / strideS) + 1); }
  // Cell size of the output canvas.
  function outputCellSize(){ return CSIZE / outputSize(); }

  // Reallocate output / computed arrays when outputSize changes.
  function syncOutputArrays() {
    const need = outputSize() * outputSize();
    if (output.length !== need) {
      output   = new Float32Array(need);
      computed = new Uint8Array(need);
    }
  }

  // ─── Filter definitions ────────────────────────────────────
  const FILTERS = {
    edge: {
      label: 'Edge Detection',
      kernel: [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]],
      bias: 0, post: 'abs',
    },
    blur: {
      label: 'Blur',
      kernel: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
      bias: 0, post: 'clamp',
    },
    sharpen: {
      label: 'Sharpen',
      kernel: [[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]],
      bias: 0, post: 'clamp',
    },
    emboss: {
      label: 'Emboss',
      kernel: [[-2,-1, 0],[-1, 1, 1],[ 0, 1, 2]],
      bias: 0.5, post: 'clamp',
    },
  };

  function fmtNum(v) {
    if (Math.abs(v) < 1e-9) return '0';
    if (Math.abs(v - Math.round(v)) < 1e-6) return String(Math.round(v));
    const inv = 1 / v;
    if (Math.abs(inv - Math.round(inv)) < 1e-3) return `1/${Math.round(inv)}`;
    return v.toFixed(2);
  }

  const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;

  // ─── Convolution ───────────────────────────────────────────
  // outR / outC are OUTPUT pixel coordinates (0-based).
  // The kernel top-left lands at padded input position (outR·S, outC·S),
  // which corresponds to actual input position (outR·S − P, outC·S − P).
  function convolveAt(outR, outC) {
    const f  = FILTERS[currentFilter];
    const OS = outputSize();
    let sum = 0;
    for (let dr = 0; dr < KSIZE; dr++) {
      for (let dc = 0; dc < KSIZE; dc++) {
        const inR = outR * strideS - paddingP + dr;
        const inC = outC * strideS - paddingP + dc;
        const v = (inR >= 0 && inR < N && inC >= 0 && inC < N) ? input[inR * N + inC] : 0;
        sum += f.kernel[dr][dc] * v;
      }
    }
    sum += f.bias;
    if (f.post === 'abs') sum = Math.abs(sum);
    output[outR * OS + outC] = sum;
  }

  function computeAll() {
    const OS = outputSize();
    syncOutputArrays();
    for (let r = 0; r < OS; r++)
      for (let c = 0; c < OS; c++) {
        convolveAt(r, c);
        computed[r * OS + c] = 1;
      }
  }

  // ─── Canvas refs ───────────────────────────────────────────
  const inputCanvas  = document.getElementById('input-canvas');
  const outputCanvas = document.getElementById('output-canvas');
  const ictx = inputCanvas.getContext('2d');
  const octx = outputCanvas.getContext('2d');

  // ─── Input canvas drawing ──────────────────────────────────
  // Draws the (N + 2P) × (N + 2P) padded grid. Padding cells are shown as a
  // dim overlay so students can see them but cannot draw on them.
  function drawInput(highlight) {
    const PN = N + 2 * paddingP;
    const CS = inputCellSize();

    ictx.fillStyle = '#000';
    ictx.fillRect(0, 0, CSIZE, CSIZE);

    // Padding cell background.
    if (paddingP > 0) {
      ictx.fillStyle = 'rgba(255,255,255,0.05)';
      for (let r = 0; r < PN; r++) {
        for (let c = 0; c < PN; c++) {
          const inner = r >= paddingP && r < paddingP + N && c >= paddingP && c < paddingP + N;
          if (!inner) ictx.fillRect(c * CS, r * CS, CS, CS);
        }
      }
    }

    // Actual drawable cells.
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = clamp01(input[r * N + c]);
        if (v <= 0) continue;
        const g = Math.round(v * 255);
        ictx.fillStyle = `rgb(${g},${g},${g})`;
        ictx.fillRect((c + paddingP) * CS, (r + paddingP) * CS, CS, CS);
      }
    }

    // Grid lines.
    ictx.strokeStyle = 'rgba(255,255,255,0.06)';
    ictx.lineWidth = 1;
    ictx.beginPath();
    for (let i = 1; i < PN; i++) {
      ictx.moveTo(i * CS + 0.5, 0);       ictx.lineTo(i * CS + 0.5, CSIZE);
      ictx.moveTo(0, i * CS + 0.5);       ictx.lineTo(CSIZE, i * CS + 0.5);
    }
    ictx.stroke();

    // Kernel highlight: in padded coordinates the kernel for output cell
    // (outR, outC) starts at padded row outR·S, col outC·S.
    if (highlight) {
      const [outR, outC] = highlight;
      const x = outC * strideS * CS;
      const y = outR * strideS * CS;
      const w = KSIZE * CS;
      ictx.fillStyle = 'rgba(179, 27, 27, 0.20)';
      ictx.fillRect(x, y, w, w);
      ictx.strokeStyle = '#B31B1B';
      ictx.lineWidth = 2;
      ictx.strokeRect(x + 1, y + 1, w - 2, w - 2);
    }
  }

  // ─── Output canvas drawing ─────────────────────────────────
  function drawOutput(highlight, useMask) {
    const OS = outputSize();
    const CS = outputCellSize();

    octx.fillStyle = '#000';
    octx.fillRect(0, 0, CSIZE, CSIZE);

    for (let r = 0; r < OS; r++) {
      for (let c = 0; c < OS; c++) {
        const idx = r * OS + c;
        if (useMask && !computed[idx]) continue;
        const v = clamp01(output[idx]);
        if (v <= 0) continue;
        const g = Math.round(v * 255);
        octx.fillStyle = `rgb(${g},${g},${g})`;
        octx.fillRect(c * CS, r * CS, CS, CS);
      }
    }

    octx.strokeStyle = 'rgba(255,255,255,0.06)';
    octx.lineWidth = 1;
    octx.beginPath();
    for (let i = 1; i < OS; i++) {
      octx.moveTo(i * CS + 0.5, 0);     octx.lineTo(i * CS + 0.5, CSIZE);
      octx.moveTo(0, i * CS + 0.5);     octx.lineTo(CSIZE, i * CS + 0.5);
    }
    octx.stroke();

    if (highlight) {
      const [r, c] = highlight;
      octx.strokeStyle = '#B31B1B';
      octx.lineWidth = 2;
      octx.strokeRect(c * CS + 1, r * CS + 1, CS - 2, CS - 2);
    }
  }

  // ─── Drawing on input ──────────────────────────────────────
  // Maps pointer position to actual (non-padded) grid cell.
  function pixelFromEvent(e) {
    const CS   = inputCellSize();
    const rect = inputCanvas.getBoundingClientRect();
    const t    = (e.touches && e.touches[0]) || e;
    const x    = (t.clientX - rect.left) * (CSIZE / rect.width);
    const y    = (t.clientY - rect.top)  * (CSIZE / rect.height);
    const c = Math.floor(x / CS) - paddingP;
    const r = Math.floor(y / CS) - paddingP;
    if (r < 0 || r >= N || c < 0 || c >= N) return null;
    return [r, c];
  }

  function paintAt(r, c, v) {
    if (input[r * N + c] === v) return false;
    input[r * N + c] = v;
    saveInputSoon();
    return true;
  }

  function pixelOn(r, c) { return input[r * N + c] > 0.01; }

  function onPaintStart(e) {
    e.preventDefault();
    stopAnimation();
    const px = pixelFromEvent(e);
    if (!px) return;
    isDrawing = true;
    drawValue = (e.button === 2 || e.ctrlKey || e.metaKey)
      ? 0
      : pixelOn(px[0], px[1]) ? 0 : brushIntensity;
    if (paintAt(px[0], px[1], drawValue)) refreshLive();
  }

  function onPaintMove(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const px = pixelFromEvent(e);
    if (!px) return;
    if (paintAt(px[0], px[1], drawValue)) refreshLive();
  }

  function onPaintEnd() { isDrawing = false; }

  inputCanvas.addEventListener('mousedown',  onPaintStart);
  inputCanvas.addEventListener('mousemove',  onPaintMove);
  window.addEventListener('mouseup',         onPaintEnd);
  inputCanvas.addEventListener('contextmenu', e => e.preventDefault());
  inputCanvas.addEventListener('touchstart',  onPaintStart, { passive: false });
  inputCanvas.addEventListener('touchmove',   onPaintMove,  { passive: false });
  window.addEventListener('touchend',         onPaintEnd);

  // ─── Filter UI ─────────────────────────────────────────────
  const filtersEl = document.getElementById('filters');
  for (const [key, f] of Object.entries(FILTERS)) {
    const btn = document.createElement('button');
    btn.className = 'filter-btn';
    btn.textContent = f.label;
    btn.dataset.filter = key;
    btn.addEventListener('click', () => setFilter(key));
    filtersEl.appendChild(btn);
  }

  function setFilter(key) {
    stopAnimation();
    currentFilter = key;
    filtersEl.querySelectorAll('.filter-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.filter === key);
    });
    try { localStorage.setItem(STORAGE.FILTER, key); } catch (_) {}
    drawKernel();
    refreshLive();
  }

  // ─── Kernel display ────────────────────────────────────────
  const kernelEl = document.getElementById('kernel');
  const calcEl   = document.getElementById('calc-content');
  const PLACEHOLDER_HTML = '<div class="calc-placeholder">Press <strong>▶ Animate</strong> or <strong>Step</strong> to walk through each output pixel. Values shown as 0–255.</div>';

  function drawKernel() {
    kernelEl.innerHTML = '';
    const f = FILTERS[currentFilter];
    for (let i = 0; i < KSIZE; i++) {
      for (let j = 0; j < KSIZE; j++) {
        const v = f.kernel[i][j];
        const cell = document.createElement('div');
        cell.className = 'kernel-cell';
        if (v > 0) cell.classList.add('pos');
        else if (v < 0) cell.classList.add('neg');
        cell.textContent = fmtNum(v);
        kernelEl.appendChild(cell);
      }
    }
  }

  function setComputation(html) {
    calcEl.innerHTML = html || PLACEHOLDER_HTML;
  }

  // ─── Output dimension formula ──────────────────────────────
  function updateDimFormula() {
    const OS        = outputSize();
    const numerator = N + 2 * paddingP - KSIZE;
    const el        = document.getElementById('dim-formula');
    if (!el) return;

    // Show step-by-step substitution matching the screenshot style.
    el.innerHTML =
      `<div class="dim-line">output = ⌊(<span class="c-N">N</span> + 2<span class="c-P">P</span> &minus; <span class="c-K">K</span>) / <span class="c-S">S</span>⌋ + 1</div>` +
      `<div class="dim-line">       = ⌊(<span class="c-N">${N}</span> + 2&middot;<span class="c-P">${paddingP}</span> &minus; <span class="c-K">${KSIZE}</span>) / <span class="c-S">${strideS}</span>⌋ + 1</div>` +
      `<div class="dim-line">       = ⌊${numerator}${strideS > 1 ? ` / ${strideS}` : ''}⌋ + 1 = <span class="dim-result">${OS}</span></div>`;
  }

  // ─── Conv params (stride + padding) ────────────────────────
  function setConvParams(p, s) {
    stopAnimation();
    paddingP = p;
    strideS  = s;

    const strideSlider  = document.getElementById('stride');
    const paddingSlider = document.getElementById('padding');
    if (strideSlider)  strideSlider.value  = s;
    if (paddingSlider) paddingSlider.value = p;

    const strideVal  = document.getElementById('stride-val');
    const paddingVal = document.getElementById('padding-val');
    if (strideVal)  strideVal.textContent  = s;
    if (paddingVal) paddingVal.textContent = p;

    document.querySelectorAll('.conv-preset-btn').forEach(btn => {
      btn.classList.toggle('active',
        parseInt(btn.dataset.p) === p && parseInt(btn.dataset.s) === s);
    });

    try {
      localStorage.setItem(STORAGE.STRIDE,  String(s));
      localStorage.setItem(STORAGE.PADDING, String(p));
    } catch (_) {}

    syncOutputArrays();
    updateDimFormula();
    refreshLive();
  }

  // ─── Live refresh ──────────────────────────────────────────
  function refreshLive() {
    if (animPlaying) return;
    syncOutputArrays();
    computeAll();
    drawInput();
    drawOutput();
    setComputation('');
  }

  // ─── Animation ─────────────────────────────────────────────
  const playBtn  = document.getElementById('play-btn');
  const stepBtn  = document.getElementById('step-btn');
  const speedInput = document.getElementById('speed');

  speedInput.addEventListener('input', () => {
    animSpeedMs = parseInt(speedInput.value, 10);
    if (animPlaying && animTimer) {
      clearInterval(animTimer);
      animTimer = setInterval(animStep, animSpeedMs);
    }
  });

  const brushInput = document.getElementById('brush');
  brushInput.addEventListener('input', () => {
    brushIntensity = parseFloat(brushInput.value);
    try { localStorage.setItem(STORAGE.BRUSH, String(brushIntensity)); } catch (_) {}
  });

  playBtn.addEventListener('click', () => {
    if (animPlaying) pauseAnimation(); else startAnimation();
  });

  stepBtn.addEventListener('click', () => {
    const OS = outputSize();
    if (animPlaying) pauseAnimation();
    // Reset to beginning if at end or not started.
    if (animIndex === 0 || animIndex >= OS * OS) {
      syncOutputArrays();
      output.fill(0);
      computed.fill(0);
      animIndex = 0;
    }
    animStep();
  });

  function startAnimation() {
    const OS = outputSize();
    animPlaying = true;
    if (animIndex >= OS * OS) {
      animIndex = 0;
      syncOutputArrays();
      computed.fill(0);
    }
    playBtn.textContent = '⏸ Pause';
    animStep();
    animTimer = setInterval(animStep, animSpeedMs);
  }

  function pauseAnimation() {
    animPlaying = false;
    playBtn.textContent = '▶ Animate';
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
  }

  function stopAnimation() {
    pauseAnimation();
    animIndex = 0;
  }

  function animStep() {
    const OS = outputSize();
    if (animIndex >= OS * OS) {
      pauseAnimation();
      drawInput();
      drawOutput();
      setComputation('<div class="calc-placeholder">Convolution complete. Press Animate to restart, or draw to refresh.</div>');
      return;
    }
    const outR = Math.floor(animIndex / OS);
    const outC = animIndex % OS;
    convolveAt(outR, outC);
    computed[outR * OS + outC] = 1;
    drawInput([outR, outC]);
    drawOutput([outR, outC], true);
    setComputationForCell(outR, outC);
    animIndex++;
  }

  // ─── Calculation panel ─────────────────────────────────────
  // Returns the luminance (0-255) of the actual input pixel at padded coords.
  function lumAt(paddedR, paddedC) {
    const r = paddedR - paddingP;
    const c = paddedC - paddingP;
    if (r < 0 || r >= N || c < 0 || c >= N) return 0;
    return Math.round(255 * input[r * N + c]);
  }

  function setComputationForCell(outR, outC) {
    const f  = FILTERS[currentFilter];
    const OS = outputSize();
    const lines = [];
    let sum = 0;

    for (let dr = 0; dr < KSIZE; dr++) {
      const parts = [];
      for (let dc = 0; dc < KSIZE; dc++) {
        const lum = lumAt(outR * strideS + dr, outC * strideS + dc);
        const k   = f.kernel[dr][dc];
        sum += lum * k;
        const kClass = k > 0 ? 'k-pos' : k < 0 ? 'k-neg' : 'k-zero';
        const vClass = lum === 0 ? 'v-zero' : '';
        parts.push(`<span class="${vClass}">${String(lum).padStart(3, ' ')}</span>×<span class="${kClass}">${fmtNum(k)}</span>`);
      }
      lines.push(parts.join(' + '));
    }

    const sumInt = Math.round(sum);
    let resultLine = `= ${sumInt}`;
    if (f.bias !== 0) {
      const biasInt = Math.round(f.bias * 255);
      resultLine = `= ${sumInt} + ${biasInt} <span class="post">(bias)</span> = ${sumInt + biasInt}`;
    }
    if (f.post === 'abs' && sumInt < 0) resultLine += `  → |·| = ${Math.abs(sumInt)}`;

    const step = outR * OS + outC + 1;
    setComputation(
      `<div class="calc-block">` +
      `<span class="label">Output[${outR}, ${outC}] — step ${step} of ${OS * OS}</span>` +
      `<div class="math">  ${lines[0]}\n+ ${lines[1]}\n+ ${lines[2]}</div>` +
      `<div class="result-line">${resultLine}</div>` +
      `</div>`
    );
  }

  // ─── Stride / Padding sliders ──────────────────────────────
  document.getElementById('stride').addEventListener('input', e => {
    setConvParams(paddingP, parseInt(e.target.value, 10));
  });
  document.getElementById('padding').addEventListener('input', e => {
    setConvParams(parseInt(e.target.value, 10), strideS);
  });

  // ─── Convolution presets ───────────────────────────────────
  document.querySelectorAll('.conv-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setConvParams(parseInt(btn.dataset.p, 10), parseInt(btn.dataset.s, 10));
    });
  });

  // ─── Clear / reset ─────────────────────────────────────────
  document.getElementById('clear-btn').addEventListener('click', () => {
    stopAnimation();
    input.fill(0);
    saveInputSoon();
    refreshLive();
  });

  // ─── Image upload ──────────────────────────────────────────
  document.getElementById('upload-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      stopAnimation();
      const tmp  = document.createElement('canvas');
      tmp.width  = N; tmp.height = N;
      const tctx = tmp.getContext('2d');
      const scale = Math.max(N / img.width, N / img.height);
      const w = img.width * scale, h = img.height * scale;
      tctx.imageSmoothingEnabled = true;
      tctx.imageSmoothingQuality = 'high';
      tctx.drawImage(img, (N - w) / 2, (N - h) / 2, w, h);
      const data = tctx.getImageData(0, 0, N, N).data;
      for (let i = 0; i < N * N; i++) {
        input[i] = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
      }
      URL.revokeObjectURL(url);
      saveInputSoon();
      refreshLive();
      e.target.value = '';
    };
    img.src = url;
  });

  // ─── Initial seed ──────────────────────────────────────────
  function seed() {
    const pattern = [
      "..####..",
      ".######.",
      ".##..##.",
      ".##.....",
      ".##.....",
      ".##.....",
      ".##..##.",
      ".######.",
      "..####..",
    ];
    const offR = 4, offC = 4;
    for (let i = 0; i < pattern.length; i++)
      for (let j = 0; j < pattern[i].length; j++)
        if (pattern[i][j] === '#') paintAt(offR + i, offC + j, 1);
  }

  // ─── Restore session state ─────────────────────────────────
  if (!loadInput()) seed();

  const savedBrush = (() => { try { return localStorage.getItem(STORAGE.BRUSH);   } catch (_) { return null; } })();
  const savedFilter= (() => { try { return localStorage.getItem(STORAGE.FILTER);  } catch (_) { return null; } })();
  const savedStride= (() => { try { return localStorage.getItem(STORAGE.STRIDE);  } catch (_) { return null; } })();
  const savedPad   = (() => { try { return localStorage.getItem(STORAGE.PADDING); } catch (_) { return null; } })();

  if (savedBrush  != null && !isNaN(parseFloat(savedBrush))) {
    brushIntensity = parseFloat(savedBrush);
    brushInput.value = brushIntensity;
  }
  if (savedStride != null && !isNaN(parseInt(savedStride))) strideS  = parseInt(savedStride);
  if (savedPad    != null && !isNaN(parseInt(savedPad)))    paddingP = parseInt(savedPad);

  setFilter(FILTERS[savedFilter] ? savedFilter : 'edge');
  setConvParams(paddingP, strideS); // sync sliders + formula + initial render
})();
