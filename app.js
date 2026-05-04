(() => {
  // ─── Constants & state ─────────────────────────────────────
  const N = 16;            // grid size
  const KSIZE = 3;
  const KHALF = 1;
  const CSIZE = 320;       // canvas pixel size
  const CELL = CSIZE / N;  // 20px per cell

  const input = new Float32Array(N * N);
  const output = new Float32Array(N * N);
  const computed = new Uint8Array(N * N); // 1 = pixel filled in animation

  let currentFilter = 'edge';
  let isDrawing = false;
  let drawValue = 1;
  let brushIntensity = 1;

  let animPlaying = false;
  let animIndex = 0;
  let animTimer = null;
  let animSpeedMs = 80;

  // ─── Filter definitions ────────────────────────────────────
  const FILTERS = {
    edge: {
      label: 'Edge Detection',
      kernel: [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]],
      bias: 0,
      post: 'abs',
    },
    blur: {
      label: 'Blur',
      kernel: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
      bias: 0,
      post: 'clamp',
    },
    sharpen: {
      label: 'Sharpen',
      kernel: [[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]],
      bias: 0,
      post: 'clamp',
    },
    emboss: {
      label: 'Emboss',
      kernel: [[-2,-1, 0],[-1, 1, 1],[ 0, 1, 2]],
      bias: 0.5,
      post: 'clamp',
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
  function convolveAt(r, c) {
    const f = FILTERS[currentFilter];
    let sum = 0;
    for (let dr = -KHALF; dr <= KHALF; dr++) {
      for (let dc = -KHALF; dc <= KHALF; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= N || cc < 0 || cc >= N) continue;
        sum += f.kernel[dr + KHALF][dc + KHALF] * input[rr * N + cc];
      }
    }
    sum += f.bias;
    if (f.post === 'abs') sum = Math.abs(sum);
    output[r * N + c] = sum;
  }

  function computeAll() {
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        convolveAt(r, c);
        computed[r * N + c] = 1;
      }
    }
  }

  // ─── Canvas rendering ──────────────────────────────────────
  const inputCanvas = document.getElementById('input-canvas');
  const outputCanvas = document.getElementById('output-canvas');
  const ictx = inputCanvas.getContext('2d');
  const octx = outputCanvas.getContext('2d');

  function drawGridBackground(ctx) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, CSIZE, CSIZE);
  }

  function drawGridLines(ctx) {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < N; i++) {
      ctx.moveTo(i * CELL + 0.5, 0);
      ctx.lineTo(i * CELL + 0.5, CSIZE);
      ctx.moveTo(0, i * CELL + 0.5);
      ctx.lineTo(CSIZE, i * CELL + 0.5);
    }
    ctx.stroke();
  }

  function drawCells(ctx, data, useMask) {
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;
        if (useMask && !computed[idx]) continue;
        const v = clamp01(data[idx]);
        if (v <= 0) continue;
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      }
    }
  }

  function drawInput(highlight) {
    drawGridBackground(ictx);
    drawCells(ictx, input, false);
    drawGridLines(ictx);
    if (highlight) {
      const [r, c] = highlight;
      const x = (c - KHALF) * CELL;
      const y = (r - KHALF) * CELL;
      const w = KSIZE * CELL;
      ictx.fillStyle = 'rgba(179, 27, 27, 0.20)';
      ictx.fillRect(x, y, w, w);
      ictx.strokeStyle = '#B31B1B';
      ictx.lineWidth = 2;
      ictx.strokeRect(x + 1, y + 1, w - 2, w - 2);
    }
  }

  function drawOutput(highlight, useMask) {
    drawGridBackground(octx);
    drawCells(octx, output, useMask);
    drawGridLines(octx);
    if (highlight) {
      const [r, c] = highlight;
      octx.strokeStyle = '#B31B1B';
      octx.lineWidth = 2;
      octx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
    }
  }

  // ─── Drawing on input ──────────────────────────────────────
  function pixelFromEvent(e) {
    const rect = inputCanvas.getBoundingClientRect();
    const t = (e.touches && e.touches[0]) || e;
    const x = (t.clientX - rect.left) * (CSIZE / rect.width);
    const y = (t.clientY - rect.top) * (CSIZE / rect.height);
    const c = Math.floor(x / CELL);
    const r = Math.floor(y / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) return null;
    return [r, c];
  }

  function paintAt(r, c, v) {
    if (input[r * N + c] === v) return false;
    input[r * N + c] = v;
    return true;
  }

  function pixelOn(r, c) {
    return input[r * N + c] > 0.01;
  }

  function onPaintStart(e) {
    e.preventDefault();
    stopAnimation();
    const px = pixelFromEvent(e);
    if (!px) return;
    isDrawing = true;
    if (e.button === 2 || e.ctrlKey || e.metaKey) {
      drawValue = 0;
    } else {
      drawValue = pixelOn(px[0], px[1]) ? 0 : brushIntensity;
    }
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

  inputCanvas.addEventListener('mousedown', onPaintStart);
  inputCanvas.addEventListener('mousemove', onPaintMove);
  window.addEventListener('mouseup', onPaintEnd);
  inputCanvas.addEventListener('contextmenu', e => e.preventDefault());
  inputCanvas.addEventListener('touchstart', onPaintStart, { passive: false });
  inputCanvas.addEventListener('touchmove', onPaintMove, { passive: false });
  window.addEventListener('touchend', onPaintEnd);

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
    for (const btn of filtersEl.querySelectorAll('.filter-btn')) {
      btn.classList.toggle('active', btn.dataset.filter === key);
    }
    drawKernel();
    refreshLive();
  }

  // ─── Kernel display ────────────────────────────────────────
  const kernelEl = document.getElementById('kernel');
  const calcEl = document.getElementById('calc-content');
  const PLACEHOLDER_HTML = '<div class="calc-placeholder">Press <strong>▶ Animate</strong> to step through each output pixel. Pixel values are shown as 0–255.</div>';

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

  // ─── Live refresh ──────────────────────────────────────────
  function refreshLive() {
    if (animPlaying) return;
    computeAll();
    drawInput();
    drawOutput();
    setComputation('');
  }

  // ─── Animation ─────────────────────────────────────────────
  const playBtn = document.getElementById('play-btn');
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
  });

  playBtn.addEventListener('click', () => {
    if (animPlaying) pauseAnimation();
    else startAnimation();
  });

  function startAnimation() {
    animPlaying = true;
    if (animIndex >= N * N) {
      animIndex = 0;
      computed.fill(0);
    }
    playBtn.textContent = '⏸ Pause';
    animStep();
    animTimer = setInterval(animStep, animSpeedMs);
  }

  function pauseAnimation() {
    animPlaying = false;
    playBtn.textContent = '▶ Animate';
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }
  }

  function stopAnimation() {
    pauseAnimation();
    animIndex = 0;
  }

  function animStep() {
    if (animIndex >= N * N) {
      pauseAnimation();
      drawInput();
      drawOutput();
      setComputation('<div class="calc-placeholder">Convolution complete. Press Animate to restart, or draw to refresh.</div>');
      return;
    }
    const r = Math.floor(animIndex / N);
    const c = animIndex % N;
    convolveAt(r, c);
    computed[r * N + c] = 1;
    drawInput([r, c]);
    drawOutput([r, c], true);
    setComputationForCell(r, c);
    animIndex++;
  }

  function lumAt(r, c) {
    if (r < 0 || r >= N || c < 0 || c >= N) return 0;
    return Math.round(255 * input[r * N + c]);
  }

  function setComputationForCell(r, c) {
    const f = FILTERS[currentFilter];
    const lines = [];
    let sum = 0;

    for (let dr = -KHALF; dr <= KHALF; dr++) {
      const parts = [];
      for (let dc = -KHALF; dc <= KHALF; dc++) {
        const lum = lumAt(r + dr, c + dc);
        const k = f.kernel[dr + KHALF][dc + KHALF];
        sum += lum * k;
        const lumStr = String(lum).padStart(3, ' ');
        const kStr = fmtNum(k);
        const kClass = k > 0 ? 'k-pos' : k < 0 ? 'k-neg' : 'k-zero';
        const vClass = lum === 0 ? 'v-zero' : '';
        parts.push(`<span class="${vClass}">${lumStr}</span>×<span class="${kClass}">${kStr}</span>`);
      }
      lines.push(parts.join(' + '));
    }

    const sumInt = Math.round(sum);
    let resultLine = `= ${sumInt}`;
    if (f.bias !== 0) {
      const biasInt = Math.round(f.bias * 255);
      resultLine = `= ${sumInt} + ${biasInt} <span class="post">(bias)</span> = ${sumInt + biasInt}`;
    }
    if (f.post === 'abs' && sumInt < 0) {
      resultLine += `  → |·| = ${Math.abs(sumInt)}`;
    }

    setComputation(
      `<div class="calc-block">` +
        `<span class="label">Output[${r}, ${c}] =</span>` +
        `<div class="math">  ${lines[0]}\n+ ${lines[1]}\n+ ${lines[2]}</div>` +
        `<div class="result-line">${resultLine}</div>` +
      `</div>`
    );
  }

  // ─── Clear / reset ─────────────────────────────────────────
  document.getElementById('clear-btn').addEventListener('click', () => {
    stopAnimation();
    input.fill(0);
    refreshLive();
  });

  // ─── Image upload ──────────────────────────────────────────
  document.getElementById('upload-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      stopAnimation();
      const tmp = document.createElement('canvas');
      tmp.width = N;
      tmp.height = N;
      const tctx = tmp.getContext('2d');
      // cover-fit so the image fills the grid (crops longer dimension)
      const scale = Math.max(N / img.width, N / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (N - w) / 2;
      const y = (N - h) / 2;
      tctx.imageSmoothingEnabled = true;
      tctx.imageSmoothingQuality = 'high';
      tctx.drawImage(img, x, y, w, h);
      const data = tctx.getImageData(0, 0, N, N).data;
      for (let i = 0; i < N * N; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        input[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      }
      URL.revokeObjectURL(url);
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
    for (let i = 0; i < pattern.length; i++) {
      for (let j = 0; j < pattern[i].length; j++) {
        if (pattern[i][j] === '#') paintAt(offR + i, offC + j, 1);
      }
    }
  }

  seed();
  setFilter('edge');
})();
