/* ═══════════════════════════════════════════════════════════════════════════ */
/* DEEPGUARD — Main JavaScript for Predict Page                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

// ─── Tab Switching ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.add('hidden');
      p.classList.remove('active');
    });
    const panel = document.getElementById(`panel-${tab}`);
    panel.classList.remove('hidden');
    panel.classList.add('active');
  });
});

// ─── Helpers ─────────────────────────────────────────────────────────────────
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showState(prefix, state) {
  // state = 'idle' | 'loading' | 'result' | 'error'
  ['idle', 'loading', 'result', 'error'].forEach(s => {
    const el = document.getElementById(`${prefix}-${s}-state`);
    if (el) {
      if (s === state) {
        el.classList.remove('hidden');
      } else {
        el.classList.add('hidden');
      }
    }
  });
}

function buildModelResults(containerId, results) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  results.forEach(r => {
    const card = document.createElement('div');
    card.className = `model-result-card ${r.status !== 'success' ? 'error' : r.label === 'FAKE' ? 'fake' : 'real'}`;
    if (r.status !== 'success') {
      card.innerHTML = `
        <div class="model-result-header">
          <span class="model-result-name">${r.model_name}</span>
          <span class="model-result-sub">${r.subtitle || ''}</span>
        </div>
        <div class="model-result-body">
          <span class="model-result-error">⚠ Error: ${r.error || 'Unknown error'}</span>
        </div>
      `;
    } else {
      const icon = r.label === 'FAKE' ? '🔴' : '🟢';
      card.innerHTML = `
        <div class="model-result-header">
          <span class="model-result-name">${r.model_name}</span>
          <span class="model-result-sub">${r.subtitle || ''}</span>
        </div>
        <div class="model-result-body">
          <span class="model-result-badge ${r.label === 'FAKE' ? 'badge-fake' : 'badge-real'}">${icon} ${r.label}</span>
          <div class="model-result-conf">
            <span class="model-result-conf-label">Confidence</span>
            <div class="conf-bar">
              <div class="conf-bar-fill ${r.label === 'FAKE' ? 'conf-bar-fake' : 'conf-bar-real'}" style="width: ${r.confidence}%"></div>
            </div>
            <span class="model-result-conf-value">${r.confidence}%</span>
          </div>
        </div>
      `;
    }
    container.appendChild(card);
  });
}

function renderVerdict(prefix, ensemble) {
  const card = document.getElementById(`${prefix}-verdict-card`);
  const icon = document.getElementById(`${prefix}-verdict-icon`);
  const label = document.getElementById(`${prefix}-verdict-label`);
  const conf = document.getElementById(`${prefix}-verdict-conf`);

  card.className = `verdict-card ${ensemble.label === 'FAKE' ? 'verdict-fake' : 'verdict-real'}`;
  icon.textContent = ensemble.label === 'FAKE' ? '🔴' : '🟢';
  label.textContent = ensemble.label === 'FAKE' ? 'DEEPFAKE DETECTED' : 'APPEARS AUTHENTIC';
  conf.textContent = `Confidence: ${ensemble.confidence}%`;

  // Votes (image only)
  const votes = document.getElementById(`${prefix}-verdict-votes`);
  if (votes && ensemble.votes) {
    votes.textContent = `Votes — Fake: ${ensemble.votes.fake} | Real: ${ensemble.votes.real}`;
  }
}

/* ════════════════════════════════════════════════════════════════════════════
   IMAGE DETECTION
   ════════════════════════════════════════════════════════════════════════════ */
(function () {
  const dropzone = document.getElementById('img-dropzone');
  const fileInput = document.getElementById('img-file-input');
  const previewContainer = document.getElementById('img-preview-container');
  const previewImg = document.getElementById('img-preview');
  const removeBtn = document.getElementById('img-remove-btn');
  const analyzeBtn = document.getElementById('img-analyze-btn');
  const resetBtn = document.getElementById('img-reset-btn');
  const errorResetBtn = document.getElementById('img-error-reset-btn');

  let selectedFile = null;

  function setFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewContainer.classList.remove('hidden');
      dropzone.classList.add('hidden');
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function reset() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewContainer.classList.add('hidden');
    dropzone.classList.remove('hidden');
    analyzeBtn.disabled = true;
    showState('img', 'idle');
  }

  // Drag & Drop
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  removeBtn.addEventListener('click', reset);
  resetBtn.addEventListener('click', reset);
  errorResetBtn.addEventListener('click', reset);

  // Analyze
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showState('img', 'loading');
    analyzeBtn.disabled = true;

    // Animate loading steps
    const steps = ['img-step1', 'img-step2', 'img-step3', 'img-step4', 'img-step5'];
    let si = 0;
    const stepInterval = setInterval(() => {
      if (si < steps.length) {
        document.getElementById(steps[si]).textContent =
          document.getElementById(steps[si]).textContent.replace('⬜', '✅');
        si++;
      }
    }, 600);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const resp = await fetch('/api/predict/image', { method: 'POST', body: formData });
      clearInterval(stepInterval);
      steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = el.textContent.replace('⬜', '✅');
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await resp.json();
      renderVerdict('img', data.ensemble);
      buildModelResults('img-model-results', data.results);
      showState('img', 'result');
    } catch (err) {
      clearInterval(stepInterval);
      document.getElementById('img-error-msg').textContent = err.message;
      showState('img', 'error');
      analyzeBtn.disabled = false;
    }
  });
})();

/* ════════════════════════════════════════════════════════════════════════════
   AUDIO DETECTION
   ════════════════════════════════════════════════════════════════════════════ */
(function () {
  const dropzone = document.getElementById('aud-dropzone');
  const fileInput = document.getElementById('aud-file-input');
  const previewContainer = document.getElementById('aud-preview-container');
  const filenameEl = document.getElementById('aud-filename');
  const filesizeEl = document.getElementById('aud-filesize');
  const playerEl = document.getElementById('aud-player');
  const removeBtn = document.getElementById('aud-remove-btn');
  const analyzeBtn = document.getElementById('aud-analyze-btn');
  const resetBtn = document.getElementById('aud-reset-btn');
  const errorResetBtn = document.getElementById('aud-error-reset-btn');

  let selectedFile = null;

  function setFile(file) {
    selectedFile = file;
    filenameEl.textContent = file.name;
    filesizeEl.textContent = formatFileSize(file.size);
    const url = URL.createObjectURL(file);
    playerEl.src = url;
    previewContainer.classList.remove('hidden');
    dropzone.classList.add('hidden');
    analyzeBtn.disabled = false;
  }

  function reset() {
    selectedFile = null;
    fileInput.value = '';
    playerEl.src = '';
    previewContainer.classList.add('hidden');
    dropzone.classList.remove('hidden');
    analyzeBtn.disabled = true;
    // Reset loading steps text
    ['aud-step1', 'aud-step2', 'aud-step3'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = el.textContent.replace('✅', '⬜');
    });
    showState('aud', 'idle');
  }

  // Drag & Drop
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  removeBtn.addEventListener('click', reset);
  resetBtn.addEventListener('click', reset);
  errorResetBtn.addEventListener('click', reset);

  // Analyze
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showState('aud', 'loading');
    analyzeBtn.disabled = true;

    const steps = ['aud-step1', 'aud-step2', 'aud-step3'];
    let si = 0;
    const stepInterval = setInterval(() => {
      if (si < steps.length) {
        const el = document.getElementById(steps[si]);
        if (el) el.textContent = el.textContent.replace('⬜', '✅');
        si++;
      }
    }, 800);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const resp = await fetch('/api/predict/audio', { method: 'POST', body: formData });
      clearInterval(stepInterval);
      steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = el.textContent.replace('⬜', '✅');
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await resp.json();
      renderVerdict('aud', data.ensemble);
      buildModelResults('aud-model-results', data.results);
      showState('aud', 'result');
    } catch (err) {
      clearInterval(stepInterval);
      document.getElementById('aud-error-msg').textContent = err.message;
      showState('aud', 'error');
      analyzeBtn.disabled = false;
    }
  });
})();
