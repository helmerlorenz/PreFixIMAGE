(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const MODELS = [
    { key: 'restormer', name: 'Restormer' },
    { key: 'dncnn', name: 'DnCNN' },
    { key: 'nafnet', name: 'NAFNet' }
  ];
  let working = false;
  let developerMode = false;
  const devToggle = $('#devModeToggle');
  function updateMode() {
    devToggle.setAttribute('aria-pressed', String(developerMode));
    devToggle.textContent = `Developer Testing Mode: ${developerMode ? 'ON' : 'OFF'}`;
    $('#devInstruction').hidden = !developerMode;
    $('#devReference').hidden = !developerMode;
    $('#statisticsTitle').textContent = developerMode ? 'EVALUATION — PSNR & SSIM' : 'MODEL STATISTICS';
    $('#metricsNote').textContent = developerMode
      ? 'Ground-truth quality metrics, calculated only against the selected clean reference. Higher PSNR and SSIM indicate closer similarity to the reference.'
      : 'Values represent the actual output PNG file sizes and measured inference times.';
  }
  devToggle.addEventListener('click', () => {
    if (working) return;
    developerMode = !developerMode;
    $('#reference').value = '';
    updateMode();
    if ($('#results').classList.contains('active') || $('#processing').classList.contains('active')) show('upload');
  });
  updateMode();
  let animationTimer = null;
  let modelStates = {};

  const show = (id) => {
    document.querySelectorAll('.screen').forEach(el => el.classList.toggle('active', el.id === id));
    window.scrollTo({ top: 0, behavior: 'instant' });
  };
  $('#begin').addEventListener('click', () => show('upload'));
  $('#home').addEventListener('click', () => { if (!working) show('landing'); });
  $('#again').addEventListener('click', () => {
    if (working) return;
    $('#reference').value = '';
    $('#resultNotice').textContent = '';
    $('#models').replaceChildren();
    $('#metrics').replaceChildren();
    show('upload');
  });
  $('#uploadBtn').addEventListener('click', () => $('#file').click());
  $('#file').addEventListener('change', event => {
    const file = event.target.files[0]; if (file) processFile(file);
  });

  $('#sampleBtn').addEventListener('click', () => {
    const c = document.createElement('canvas'); c.width = 460; c.height = 320;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, c.width, c.height);
    grad.addColorStop(0, '#2e657c'); grad.addColorStop(1, '#e8b871');
    g.fillStyle = grad; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#153f48'; g.fillRect(20, 70, 140, 220);
    g.fillStyle = '#edecdf'; g.fillRect(70, 100, 70, 70);
    g.fillStyle = '#f5b86a'; g.beginPath(); g.arc(280, 105, 55, 0, Math.PI * 2); g.fill();
    const pixels = g.getImageData(0, 0, c.width, c.height);
    for (let p = 0; p < pixels.data.length; p += 4) {
      const z = (Math.random() - .5) * 76;
      for (let k = 0; k < 3; k++) pixels.data[p + k] = Math.max(0, Math.min(255, pixels.data[p + k] + z));
    }
    g.putImageData(pixels, 0, 0);
    c.toBlob(blob => { if (blob) processFile(new File([blob], 'sample.png', { type: 'image/png' })); }, 'image/png');
  });

  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => {
    event.preventDefault(); drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => {
    event.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', event => {
    const file = event.dataTransfer.files[0]; if (file) processFile(file);
  });
  window.addEventListener('paste', event => {
    if (!$('#upload').classList.contains('active')) return;
    const item = [...(event.clipboardData?.items || [])].find(p => p.type.startsWith('image/'));
    if (item) processFile(item.getAsFile());
  });

  async function refreshStatus() {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) throw new Error('Status unavailable');
      const data = await response.json();
      $('#modeNotice').textContent = data.mode === 'preview'
        ? 'PREVIEW MODE — demo filters only. Set IMAGE_MODE=real for neural network inference.'
        : `REAL MODE — using ${String(data.device).toUpperCase()} for neural network inference.`;
    } catch (_) {
      $('#modeNotice').textContent = 'Backend not connected. Open iMAGE through FastAPI.';
    }
  }

  function label(key) { return MODELS.find(m => m.key === key)?.name || key; }
  function formatBytes(size) {
    if (!Number.isFinite(size)) return 'Unavailable';
    return size >= 1048576 ? `${(size / 1048576).toFixed(2)} MB` : `${(size / 1024).toFixed(1)} KB`;
  }
  function formatMs(time) {
    if (!Number.isFinite(time)) return 'Unavailable';
    return time >= 1000 ? `${(time / 1000).toFixed(2)} s` : `${time.toFixed(2)} ms`;
  }
  function button(container, title, onClick) {
    const el = document.createElement('button');
    el.className = 'pill'; el.textContent = title;
    el.addEventListener('click', () => onClick(el));
    container.appendChild(el);
  }
  function save(dataURI, filename) {
    const a = document.createElement('a');
    a.href = dataURI; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
  }
  async function imageBlob(uri) {
    const response = await fetch(uri);
    if (!response.ok) throw new Error('Unable to obtain image');
    return await response.blob();
  }
  async function copyImage(uri, buttonElement) {
    try {
      const blob = await imageBlob(uri);
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      const text = buttonElement.textContent;
      buttonElement.textContent = 'Copied';
      setTimeout(() => { buttonElement.textContent = text; }, 1200);
    } catch (_) { alert('Image clipboard is not supported here. Use Download instead.'); }
  }
  async function shareImage(uri, filename) {
    try {
      const blob = await imageBlob(uri);
      const file = new File([blob], filename, { type: 'image/png' });
      if (navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: 'iMAGE denoised image' });
      } else {
        alert('Native file sharing is unavailable in this browser. Use Download instead.');
      }
    } catch (error) {
      if (error?.name !== 'AbortError') alert('Sharing unavailable. Use Download instead.');
    }
  }
  function exportActions(parent, image, filename) {
    const actions = document.createElement('div'); actions.className = 'actions';
    button(actions, 'Download', () => save(image, filename));
    button(actions, 'Copy', element => copyImage(image, element));
    button(actions, 'Share', () => shareImage(image, filename));
    parent.appendChild(actions);
  }

  function prepareProgress() {
    modelStates = Object.fromEntries(MODELS.map(m => [m.key, { progress: 0, status: 'Preparing', result: null }]));
    const parent = $('#processCards'); parent.replaceChildren();
    MODELS.forEach(m => {
      const card = document.createElement('div'); card.className = 'process-card';
      const top = document.createElement('div'); top.className = 'process-top';
      const title = document.createElement('span'); title.textContent = m.name;
      const state = document.createElement('span'); state.className = 'process-status'; state.id = `state-${m.key}`;
      state.textContent = 'Preparing'; top.append(title, state);
      const track = document.createElement('div'); track.className = 'process-bar';
      const fill = document.createElement('i'); fill.id = `fill-${m.key}`; track.appendChild(fill);
      const foot = document.createElement('div'); foot.className = 'process-foot';
      const desc = document.createElement('span'); desc.textContent = 'Awaiting result'; desc.id = `desc-${m.key}`;
      const pct = document.createElement('span'); pct.id = `pct-${m.key}`; pct.textContent = '0%';
      foot.append(desc, pct); card.append(top, track, foot); parent.appendChild(card);
    });
    show('processing');
    // Indeterminate visual animation until confirmed server completion; percentages are estimates.
    animationTimer = setInterval(() => {
      MODELS.forEach(m => {
        const state = modelStates[m.key];
        if (state.status === 'Complete' || state.status === 'Failed') return;
        const ceiling = state.status === 'Running' ? 95 : 82;
        state.progress = Math.min(ceiling, state.progress + Math.max(.35, (ceiling - state.progress) * .055));
        paintProgress(m.key);
      });
    }, 280);
  }
  function paintProgress(key) {
    const state = modelStates[key];
    $(`#fill-${key}`).style.width = `${Math.round(state.progress)}%`;
    $(`#pct-${key}`).textContent = `${Math.round(state.progress)}%`;
    $(`#state-${key}`).textContent = state.status;
    $(`#desc-${key}`).textContent = state.status === 'Complete' ? 'Output ready'
      : state.status === 'Failed' ? 'Processing failed'
      : 'Estimated progress · awaiting model output';
  }
  function readEvent(event) {
    if (event.event === 'model_started') {
      modelStates[event.key].status = 'Running';
      paintProgress(event.key);
    } else if (event.event === 'model_finished') {
      const state = modelStates[event.key];
      state.status = 'Complete'; state.progress = 100; state.result = event;
      paintProgress(event.key);
    } else if (event.event === 'model_failed') {
      const state = modelStates[event.key];
      state.status = 'Failed'; state.progress = 0;
      state.result = event;
      paintProgress(event.key);
    }
  }

  async function consumeLines(response, onEvent) {
    if (!response.body) throw new Error('Streaming responses are unsupported by this browser.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let partial = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      partial += decoder.decode(value, { stream: true });
      const lines = partial.split('\n'); partial = lines.pop();
      for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    }
    partial += decoder.decode();
    if (partial.trim()) onEvent(JSON.parse(partial));
  }

  function makeComparison(title, rows, key, formatter) {
    const section = document.createElement('section');
    const heading = document.createElement('div'); heading.className = 'metric-heading'; heading.textContent = title;
    section.appendChild(heading);
    const max = Math.max(1, ...rows.map(r => Number.isFinite(r[key]) ? r[key] : 0));
    rows.forEach(r => {
      const block = document.createElement('div'); block.className = 'metric-item';
      const top = document.createElement('div'); top.className = 'metric-row';
      const name = document.createElement('span'); name.textContent = label(r.key);
      const value = document.createElement('span'); value.textContent = r.error ? 'Unavailable' : formatter(r[key]);
      top.append(name, value);
      const track = document.createElement('div'); track.className = 'metric-track';
      const fill = document.createElement('i');
      fill.style.width = `${r.error ? 0 : r[key] === null && key === 'psnr' ? 100 : Math.min(100, (r[key] || 0) / max * 100)}%`;
      track.appendChild(fill); block.append(top, track); section.appendChild(block);
    });
    return section;
  }

  function display(input, size, mode, outputs) {
    const orig = $('#origCanvas'); const image = new Image();
    image.onload = () => {
      orig.width = image.naturalWidth; orig.height = image.naturalHeight;
      orig.getContext('2d').drawImage(image, 0, 0);
    };
    image.src = input;
    const originalCard = orig.closest('.card');
    originalCard.querySelector('.actions').remove();
    exportActions(originalCard, input, 'image_original.png');

    const metrics = $('#metrics'); metrics.replaceChildren();
    if (developerMode) {
      const graded = outputs.map(o => ({ ...o,
        psnr: o.metrics?.psnr ?? null, ssim: o.metrics?.ssim ?? null,
        error: o.error || (!o.metrics ? 'Metrics unavailable' : null)
      }));
      metrics.append(
        makeComparison('PSNR · dB (HIGHER IS BETTER)', graded, 'psnr', n => n === null ? '∞ dB' : `${n.toFixed(2)} dB`),
        makeComparison('SSIM · 0–1 (HIGHER IS BETTER)', graded, 'ssim', n => n.toFixed(4))
      );
    } else {
      metrics.append(
        makeComparison('NEW IMAGE FILE SIZE', outputs, 'file_size_bytes', formatBytes),
        makeComparison('DENOISING TIME', outputs, 'runtime_ms', formatMs)
      );
    }
    const models = $('#models'); models.replaceChildren();
    for (const def of MODELS) {
      const entry = outputs.find(o => o.key === def.key) || { key: def.key, error: 'No result received' };
      const card = document.createElement('article'); card.className = 'card model';
      const name = document.createElement('h3'); name.textContent = def.name;
      const area = document.createElement('div'); area.className = 'imgbox';
      card.append(name, area);
      const details = document.createElement('p'); details.className = 'status';
      if (entry.error) {
        details.textContent = entry.error; details.style.color = '#f5a6a6';
      } else {
        const img = document.createElement('img'); img.src = entry.image;
        img.alt = `${def.name} denoised output`; area.appendChild(img);
        details.textContent = developerMode && entry.metrics
          ? `PSNR: ${entry.metrics.psnr === null ? '∞' : entry.metrics.psnr.toFixed(2)} dB · SSIM: ${entry.metrics.ssim.toFixed(4)}`
          : `${formatMs(entry.runtime_ms)} · ${formatBytes(entry.file_size_bytes)}`;
        exportActions(card, entry.image, `${def.key}_denoised.png`);
      }
      card.appendChild(details); models.appendChild(card);
    }
    $('#resultNotice').textContent = `${mode === 'preview' ? 'PREVIEW FILTERS — not neural network inference. ' : 'REAL MODEL INFERENCE. '}${size.width} × ${size.height} px · ${developerMode ? 'Evaluation against a matching clean reference (PSNR and SSIM).' : 'Individual output PNG sizes and measured inference times.'}`;
    show('results');
  }

  async function processFile(file) {
    if (working) return;
    if (!file || !file.type.startsWith('image/')) return alert('Please upload an image.');
    if (file.size > 10 * 1024 * 1024) return alert('Image must be 10 MB or smaller.');
    const reference = developerMode ? $('#reference').files[0] : null;
    if (developerMode && !reference) return alert('Developer Testing Mode requires a clean reference image. Select it before uploading the noisy image.');
    if (reference && (!reference.type.startsWith('image/') || reference.size > 10 * 1024 * 1024)) return alert('The clean reference must be an image of 10 MB or less.');
    working = true;
    devToggle.disabled = true;
    prepareProgress();
    let finished = false;
    let original = null, dimensions = null, mode = null;
    const outputs = [];
    try {
      const payload = new FormData(); payload.append('image', file);
      if (reference) payload.append('reference', reference);
      const response = await fetch('/api/denoise/stream', { method: 'POST', body: payload });
      if (!response.ok) {
        let message = 'Server could not process image';
        try { const error = await response.json(); message = error.detail || message; } catch (_) {}
        throw Error(message);
      }
      await consumeLines(response, event => {
        if (event.event === 'start') {
          original = event.input; dimensions = event.size; mode = event.mode;
        } else if (event.event === 'model_finished' || event.event === 'model_failed') {
          outputs.push(event);
        } else if (event.event === 'done') { finished = true; }
        readEvent(event);
      });
      if (!finished || !original) throw Error('Model processing stream ended unexpectedly.');
      if (animationTimer) clearInterval(animationTimer);
      display(original, dimensions, mode, outputs);
    } catch (error) {
      alert(`Denoising failed: ${error.message}`);
      show('upload');
    } finally {
      if (animationTimer) clearInterval(animationTimer);
      animationTimer = null; working = false; devToggle.disabled = false; $('#file').value = '';
    }
  }

  refreshStatus();
})();
