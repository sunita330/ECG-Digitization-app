/* ============================================================
   ECG DIGITIZATION SYSTEM — script.js
   Production build — uses relative /process endpoint
   ============================================================ */

'use strict';

/* ── State ──────────────────────────────────────────────────── */
const state = {
  file:       null,
  model:      'classical',
  processing: false,
  params:     { hKernel: 40, vKernel: 40, dilate: 2, closeKernel: 5 }
};

/* ── Selectors ──────────────────────────────────────────────── */
const $  = id  => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

/*
 * API endpoint — ALWAYS relative so it works on any host
 * (local dev on port 5000, Render, any other deployment)
 * Do NOT change this to an absolute URL.
 */
const FLASK_URL = '/process';

/* ── Loader stage messages ──────────────────────────────────── */
const PIPELINE_STAGES = [
  { pct:  8, msg: 'Uploading image to server…'     },
  { pct: 20, msg: 'Grayscale conversion…'          },
  { pct: 35, msg: 'Applying Otsu threshold…'       },
  { pct: 52, msg: 'Removing ECG grid lines…'       },
  { pct: 68, msg: 'Enhancing waveform…'            },
  { pct: 82, msg: 'Extracting signal coordinates…' },
  { pct: 93, msg: 'Calculating heart rate…'        },
  { pct: 97, msg: 'Generating signal plot…'        },
];


/* ============================================================
   GRID CANVAS BACKGROUND
   ============================================================ */
(function initGridCanvas() {
  const canvas = $('gridCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
  }
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(0,229,255,0.04)';
    ctx.lineWidth   = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }
  }
  window.addEventListener('resize', resize);
  resize();
})();


/* ============================================================
   HERO CARD TILT
   ============================================================ */
(function initTilt() {
  const card = document.querySelector('.ecg-display');
  if (!card) return;
  card.addEventListener('mousemove', e => {
    const r  = card.getBoundingClientRect();
    const dx = (e.clientX - r.left - r.width  / 2) / (r.width  / 2);
    const dy = (e.clientY - r.top  - r.height / 2) / (r.height / 2);
    card.style.transform = `perspective(800px) rotateY(${dx*8}deg) rotateX(${-dy*6}deg) scale(1.02)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transition = 'transform .5s ease';
    card.style.transform  = 'perspective(800px) rotateY(0) rotateX(0) scale(1)';
  });
  card.addEventListener('mouseenter', () => {
    card.style.transition = 'transform .1s ease';
  });
})();


/* ============================================================
   SCROLL REVEAL
   ============================================================ */
(function initReveal() {
  const els = $$(
    '.step-card, .stage-card, .signal-card, .settings-card, .upload-card, .compare-table-wrap'
  );
  els.forEach(el => el.classList.add('reveal'));

  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('in-view'), i * 60);
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  els.forEach(el => io.observe(el));
})();


/* ============================================================
   PARAMETER SLIDERS
   ============================================================ */
(function initSliders() {
  [
    { id: 'hKernel',     valId: 'hKernelVal', key: 'hKernel',     sfx: 'px' },
    { id: 'vKernel',     valId: 'vKernelVal', key: 'vKernel',     sfx: 'px' },
    { id: 'dilate',      valId: 'dilateVal',  key: 'dilate',      sfx: ''   },
    { id: 'closeKernel', valId: 'closeVal',   key: 'closeKernel', sfx: 'px' },
  ].forEach(({ id, valId, key, sfx }) => {
    const inp = $(id);
    const out = $(valId);
    if (!inp || !out) return;
    const sync = () => {
      state.params[key] = parseInt(inp.value, 10);
      out.textContent   = inp.value + sfx;
    };
    inp.addEventListener('input', sync);
    sync();
  });
})();


/* ============================================================
   MODEL TOGGLE
   ============================================================ */
$$('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.model = btn.dataset.val;
  });
});


/* ============================================================
   FILE UPLOAD — drag & drop + browse
   ============================================================ */
const dropZone    = $('dropZone');
const fileInput   = $('fileInput');
const browseBtn   = $('browseBtn');
const previewArea = $('previewArea');
const previewImg  = $('previewImg');
const fileMeta    = $('fileMeta');
const clearBtn    = $('clearBtn');

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach(ev =>
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  })
);
['dragleave', 'drop'].forEach(ev =>
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  })
);
dropZone.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
  else showNotification('Please drop a valid image file.', 'error');
});

function handleFile(file) {
  if (file.size > 25 * 1024 * 1024) {
    showNotification('File exceeds 25 MB limit.', 'error');
    return;
  }
  state.file = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src            = ev.target.result;
    fileMeta.textContent      =
      `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB  ·  ${file.type}`;
    dropZone.style.display    = 'none';
    previewArea.style.display = 'flex';
    previewArea.classList.add('visible');
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  state.file                = null;
  fileInput.value           = '';
  previewImg.src            = '';
  previewArea.style.display = 'none';
  previewArea.classList.remove('visible');
  dropZone.style.display    = 'flex';
});


/* ============================================================
   PROCESS BUTTON
   ============================================================ */
$('processBtn').addEventListener('click', async () => {
  if (!state.file)      { showNotification('No image selected.', 'error'); return; }
  if (state.processing) return;
  await processECG();
});

async function processECG() {
  state.processing = true;
  showLoader();

let si = 0;
const warmupMsgs = [
  'Server warming up — this takes ~30s on first use…',
  'Still processing — please wait…',
  'Almost there — server is busy…',
  'Hang tight, running pipeline…',
];
let warmupIdx = 0;
const tick = setInterval(() => {
  if (si < PIPELINE_STAGES.length) {
    const s = PIPELINE_STAGES[si++];
    updateLoader(s.pct, s.msg);
  } else {
    // Keep showing reassuring messages instead of freezing
    updateLoader(97, warmupMsgs[warmupIdx % warmupMsgs.length]);
    warmupIdx++;
  }
}, 450);

  const fd = new FormData();
  fd.append('file',         state.file);
  fd.append('model',        state.model);
  fd.append('h_kernel',     state.params.hKernel);
  fd.append('v_kernel',     state.params.vKernel);
  fd.append('dilate_iter',  state.params.dilate);
  fd.append('close_kernel', state.params.closeKernel);

  try {
    updateLoader(3, 'Sending to server…');

    const res = await fetch(FLASK_URL, {
      method: 'POST',
      body:   fd,
      // No Content-Type header — browser sets it with boundary for FormData
    });

    clearInterval(tick);

    if (!res.ok) {
      let msg = `Server error ${res.status}`;
      try {
        const j = await res.json();
        msg = j.error || msg;
      } catch (_) {}
      throw new Error(msg);
    }

    const data = await res.json();
    if (!data.success) throw new Error(data.error || 'Processing failed on server.');

    updateLoader(100, 'Complete!');
    await sleep(280);
    populateResults(data);

  } catch (err) {
    clearInterval(tick);
    hideLoader();

    // Provide a helpful error for the two most common failure modes
    let userMsg = err.message;
    if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
      userMsg = 'Network error — the server could not be reached. '
              + 'If running locally, make sure app.py is running.';
    } else if (err.message.includes('502') || err.message.includes('503')) {
      userMsg = 'Server is starting up (cold start). '
              + 'Wait 30 seconds and try again.';
    }

    showNotification(userMsg, 'error');
    console.error('[ECG]', err);
  }

  state.processing = false;
}


/* ============================================================
   POPULATE RESULTS
   ============================================================ */
function populateResults(data) {
  hideLoader();

  const stageMap = {
    grayscale: 'img-grayscale',
    binary:    'img-binary',
    nogrid:    'img-nogrid',
    enhanced:  'img-enhanced',
  };

  Object.entries(stageMap).forEach(([key, id]) => {
    const src = data.stages?.[key];
    if (!src) return;
    const img = $(id);
    if (!img) return;
    img.src    = src;
    img.onload = () => {
      img.classList.add('loaded');
      const ph = img.closest('.stage-img-wrap')?.querySelector('.placeholder-content');
      if (ph) ph.style.display = 'none';
    };
  });

  if (data.signal_plot) {
    const sig = $('img-signal');
    if (sig) {
      sig.src    = data.signal_plot;
      sig.onload = () => {
        sig.classList.add('loaded');
        const ph = document.querySelector('.signal-img-wrap .placeholder-content');
        if (ph) ph.style.display = 'none';
      };
    }
    const dl = $('downloadSignal');
    if (dl) dl.onclick = () => downloadImage(data.signal_plot, 'ecg_signal.png');
  }

  if (data.metrics) {
    _setMetric('m-points', data.metrics.points);
    _setMetric('m-hr',     data.metrics.hr);
    _setMetric('m-time',   data.metrics.time_ms);

    // Rename "Dice Score" label to "Coverage"
    const diceLabel = document.querySelector('.metric:has(#m-dice) .metric-label');
    if (diceLabel) diceLabel.textContent = 'Coverage';
    _setMetric('m-dice', data.metrics.dice);
  }

  if (data.meta?.mode) {
    const badge = document.querySelector('.hero-badge');
    if (badge) {
      badge.innerHTML =
        `<span class="badge-dot"></span>Mode: ${data.meta.mode.toUpperCase()}`;
    }
  }

  setTimeout(() => {
    document.querySelector('#results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);

  showNotification('ECG processed successfully!', 'success');
}

function _setMetric(id, val) {
  const el = $(id);
  if (el) el.textContent = val ?? '—';
}


/* ============================================================
   LOADER
   ============================================================ */
function showLoader() {
  $('loaderOverlay')?.classList.add('active');
  updateLoader(3, 'Initialising pipeline…');
}
function hideLoader() {
  $('loaderOverlay')?.classList.remove('active');
}
function updateLoader(pct, msg) {
  const fill = $('loaderFill');
  const stat = $('loaderStatus');
  if (fill) fill.style.width = `${pct}%`;
  if (stat) stat.textContent = msg;
}


fetch('/health').then(r => r.json()).then(data => {
  if (!data.unet_loaded) {
    const unetBtn = document.querySelector('[data-val="unet"]');
    const hybridBtn = document.querySelector('[data-val="hybrid"]');
    if (unetBtn)  unetBtn.title = 'U-Net model not loaded — will use Classical';
    if (hybridBtn) hybridBtn.title = 'U-Net model not loaded — will use Classical';
  }
}).catch(() => {});
/* ============================================================
   NOTIFICATION TOAST
   ============================================================ */
function showNotification(msg, type = 'info') {
  document.querySelector('.toast')?.remove();

  const palette = {
    success: { bg: 'rgba(0,255,157,.14)',  border: 'rgba(0,255,157,.38)',  icon: '✓' },
    error:   { bg: 'rgba(255,58,110,.14)', border: 'rgba(255,58,110,.38)', icon: '✕' },
    info:    { bg: 'rgba(0,229,255,.11)',  border: 'rgba(0,229,255,.28)',  icon: 'ℹ' },
  };
  const p = palette[type] || palette.info;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span class="toast-icon">${p.icon}</span>${msg}`;
  Object.assign(toast.style, {
    position:             'fixed',
    bottom:               '32px',
    right:                '32px',
    zIndex:               '9999',
    display:              'flex',
    alignItems:           'center',
    gap:                  '10px',
    padding:              '14px 20px',
    borderRadius:         '12px',
    background:           p.bg,
    border:               `1px solid ${p.border}`,
    backdropFilter:       'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    color:                '#e8edf5',
    fontFamily:           "'Cabinet Grotesk', sans-serif",
    fontSize:             '.9rem',
    boxShadow:            '0 8px 32px rgba(0,0,0,.4)',
    animation:            'toastIn .3s ease both',
    maxWidth:             '440px',
    lineHeight:           '1.45',
  });

  if (!document.getElementById('__toast_styles')) {
    const s = document.createElement('style');
    s.id = '__toast_styles';
    s.textContent = `
      @keyframes toastIn  { from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);} }
      @keyframes toastOut { from{opacity:1;transform:translateY(0);}to{opacity:0;transform:translateY(14px);} }
    `;
    document.head.appendChild(s);
  }
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastOut .3s ease both';
    setTimeout(() => toast.remove(), 320);
  }, 4500);
}


/* ============================================================
   ACTIVE NAV LINK
   ============================================================ */
(function initActiveNav() {
  const navLinks = $$('.nav-link');
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => {
          link.classList.toggle(
            'active-link',
            link.getAttribute('href') === `#${entry.target.id}`
          );
        });
      }
    });
  }, { threshold: 0.5 });
  $$('section[id]').forEach(s => io.observe(s));

  if (!document.getElementById('__nav_styles')) {
    const s = document.createElement('style');
    s.id = '__nav_styles';
    s.textContent = `.nav-link.active-link{color:#e8edf5;background:rgba(0,229,255,.08);}`;
    document.head.appendChild(s);
  }
})();


/* ============================================================
   UTILITIES
   ============================================================ */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function downloadImage(src, filename) {
  const a = document.createElement('a');
  a.href     = src;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

(function prefetchHealth() {// pre-warms server on page load
  fetch('/health').catch(() => {});  // fire and forget
})();