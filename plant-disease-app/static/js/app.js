/* ═══════════════════════════════════════
   PlantScan AI — Frontend Logic
═══════════════════════════════════════ */

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const imagePreview= document.getElementById('imagePreview');
const previewImg  = document.getElementById('previewImg');
const scanBar     = document.getElementById('scanBar');
const clearBtn    = document.getElementById('clearBtn');
const analyzeBtn  = document.getElementById('analyzeBtn');

// Source tabs (Upload vs Camera)
const sourceTabs        = document.querySelectorAll('.source-tab');
const uploadSourcePanel = document.getElementById('uploadSourcePanel');
const cameraSourcePanel = document.getElementById('cameraSourcePanel');

// Camera elements
const cameraVideo    = document.getElementById('cameraVideo');
const cameraError    = document.getElementById('cameraError');
const cameraFrameGuide = document.getElementById('cameraFrameGuide');
const cameraStatus     = document.getElementById('cameraStatus');
const autoCaptureToggle= document.getElementById('autoCaptureToggle');
const captureBtn     = document.getElementById('captureBtn');
const switchCameraBtn= document.getElementById('switchCameraBtn');
const retryCameraBtn = document.getElementById('retryCameraBtn');
const captureCanvas  = document.getElementById('captureCanvas');

const modelSelect      = document.getElementById('modelSelect');
const modelHint        = document.getElementById('modelHint');
const activeModelBadge = document.getElementById('activeModelBadge');
const loadingModelName = document.getElementById('loadingModelName');

const MODEL_HINTS = {
  efficientnet: 'Highest accuracy — recommended for most cases.',
  mobilenet: 'Smaller & faster — good for quick scans / low-power devices.'
};

function updateModelUI() {
  const opt = modelSelect.options[modelSelect.selectedIndex];
  const label = opt.text.split('—')[0].trim(); // e.g. "EfficientNetB0"
  activeModelBadge.textContent = label;
  loadingModelName.textContent = label;
  modelHint.textContent = MODEL_HINTS[modelSelect.value] || '';
}

modelSelect.addEventListener('change', updateModelUI);
updateModelUI(); // set initial state on page load

const emptyState  = document.getElementById('emptyState');
const loadingState= document.getElementById('loadingState');
const resultContent=document.getElementById('resultContent');

const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

let selectedFile = null;

// ── Source Tabs (Upload vs Camera) ──────────────────────────
function getActiveMode() {
  return document.querySelector('.source-tab.active').dataset.mode;
}

function showSourceInput() {
  imagePreview.style.display = 'none';
  if (getActiveMode() === 'camera') {
    uploadSourcePanel.style.display = 'none';
    cameraSourcePanel.style.display = 'flex';
    startCamera();
  } else {
    cameraSourcePanel.style.display = 'none';
    uploadSourcePanel.style.display = 'flex';
    dropZone.style.display = 'flex';
    stopCamera();
  }
}

sourceTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.classList.contains('active')) return;
    sourceTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    selectedFile = null;
    analyzeBtn.disabled = true;
    showSourceInput();
    showEmpty();
  });
});

// ── Camera (live capture for demos, with auto-capture quality check) ─────
let cameraStream = null;
let facingMode = 'environment'; // back camera by default — better for pointing at a leaf

async function startCamera() {
  stopCamera();
  cameraError.style.display = 'none';
  cameraVideo.style.display = 'block';
  cameraFrameGuide.style.display = 'block';
  cameraStatus.style.display = 'block';
  cameraFrameGuide.className = 'camera-frame-guide';
  cameraStatus.textContent = 'Position the leaf in frame';

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError('Camera access isn\'t supported in this browser.');
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode },
      audio: false
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.onloadedmetadata = () => startAutoCaptureLoop();
  } catch (err) {
    showCameraError('Could not access camera — check browser permissions and try again.');
  }
}

function stopCamera() {
  stopAutoCaptureLoop();
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
}

function showCameraError(msg) {
  stopAutoCaptureLoop();
  cameraError.textContent = msg;
  cameraError.style.display = 'flex';
  cameraVideo.style.display = 'none';
  cameraFrameGuide.style.display = 'none';
  cameraStatus.style.display = 'none';
  retryCameraBtn.style.display = 'inline-flex';
}

retryCameraBtn.addEventListener('click', () => {
  retryCameraBtn.style.display = 'none';
  startCamera();
});

switchCameraBtn.addEventListener('click', () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
});

// Final, full-resolution capture — used by BOTH the manual shutter button
// and the auto-capture trigger below.
function captureFrame() {
  if (!cameraStream || !cameraVideo.videoWidth) return;

  captureCanvas.width = cameraVideo.videoWidth;
  captureCanvas.height = cameraVideo.videoHeight;
  captureCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);

  captureCanvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
    stopCamera();
    loadFile(file);
  }, 'image/jpeg', 0.92);
}

captureBtn.addEventListener('click', captureFrame);

// ── Auto-capture: lightweight "is this frame good?" check ───────────────
// Runs on a small downsampled canvas (cheap) every 150ms, checking:
//   1. Brightness  — not too dark / not blown out
//   2. Sharpness   — variance of a Laplacian-style edge filter (low = blurry)
//   3. Motion      — frame-to-frame difference (high = hand/leaf still moving)
// Once all three pass for several consecutive frames in a row, it auto-fires
// the same captureFrame() the manual shutter button uses.
//
// NOTE: these thresholds are reasonable starting points, not universal —
// lighting, camera quality, and leaf texture vary, so tune QUALITY below if
// it triggers too eagerly (capturing blurry shots) or never triggers at all.
const QUALITY = {
  minBrightness: 45,    // 0–255 grayscale average
  maxBrightness: 225,
  minSharpness: 18,     // Laplacian variance — raise to demand a sharper image
  maxMotion: 6,         // avg per-pixel grayscale change vs previous frame
  requiredStreak: 6     // consecutive good frames needed (~6 × 150ms ≈ 0.9s held steady)
};

const SAMPLE_W = 96, SAMPLE_H = 72;
const sampleCanvas = document.createElement('canvas');
sampleCanvas.width = SAMPLE_W;
sampleCanvas.height = SAMPLE_H;
const sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });

let prevGray = null;
let goodFrameStreak = 0;
let autoCaptureLoopId = null;
let isCapturingNow = false;

function toGrayscale(imageData) {
  const { data } = imageData;
  const gray = new Float32Array(SAMPLE_W * SAMPLE_H);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return gray;
}

function analyzeFrame() {
  if (!cameraStream || !cameraVideo.videoWidth || isCapturingNow) return;

  sampleCtx.drawImage(cameraVideo, 0, 0, SAMPLE_W, SAMPLE_H);
  const gray = toGrayscale(sampleCtx.getImageData(0, 0, SAMPLE_W, SAMPLE_H));

  // 1. Brightness
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const brightness = sum / gray.length;

  // 2. Sharpness — variance of a simple 4-neighbor Laplacian
  let edgeSum = 0, edgeSumSq = 0, edgeCount = 0;
  for (let y = 1; y < SAMPLE_H - 1; y++) {
    for (let x = 1; x < SAMPLE_W - 1; x++) {
      const idx = y * SAMPLE_W + x;
      const lap = 4 * gray[idx] - gray[idx - 1] - gray[idx + 1] - gray[idx - SAMPLE_W] - gray[idx + SAMPLE_W];
      edgeSum += lap;
      edgeSumSq += lap * lap;
      edgeCount++;
    }
  }
  const edgeMean = edgeSum / edgeCount;
  const sharpness = (edgeSumSq / edgeCount) - (edgeMean * edgeMean);

  // 3. Motion — mean absolute difference vs previous frame
  let motion = 0;
  if (prevGray) {
    let diffSum = 0;
    for (let i = 0; i < gray.length; i++) diffSum += Math.abs(gray[i] - prevGray[i]);
    motion = diffSum / gray.length;
  }
  prevGray = gray;

  const tooDark   = brightness < QUALITY.minBrightness;
  const tooBright = brightness > QUALITY.maxBrightness;
  const tooShaky  = motion > QUALITY.maxMotion;
  const tooBlurry = sharpness < QUALITY.minSharpness;

  let state, message;
  if (tooDark)        { state = 'bad';  message = 'Too dark — find better lighting'; }
  else if (tooBright) { state = 'bad';  message = 'Too bright — avoid glare'; }
  else if (tooShaky)  { state = 'bad';  message = 'Hold steady…'; }
  else if (tooBlurry) { state = 'warn'; message = 'Focusing…'; }
  else                 { state = 'good'; message = 'Looking good — hold still'; }

  cameraStatus.textContent = message;
  cameraFrameGuide.className = 'camera-frame-guide state-' + state;

  if (state === 'good' && autoCaptureToggle.checked) {
    goodFrameStreak++;
    if (goodFrameStreak >= QUALITY.requiredStreak) {
      isCapturingNow = true;
      cameraStatus.textContent = '✓ Captured!';
      cameraFrameGuide.className = 'camera-frame-guide state-captured';
      stopAutoCaptureLoop();
      setTimeout(captureFrame, 150); // brief pause so the green flash is visible
    }
  } else {
    goodFrameStreak = 0;
  }
}

function startAutoCaptureLoop() {
  stopAutoCaptureLoop();
  goodFrameStreak = 0;
  prevGray = null;
  isCapturingNow = false;
  autoCaptureLoopId = setInterval(analyzeFrame, 150);
}

function stopAutoCaptureLoop() {
  if (autoCaptureLoopId) {
    clearInterval(autoCaptureLoopId);
    autoCaptureLoopId = null;
  }
}

// Stop the camera if the user navigates away / closes the tab
window.addEventListener('beforeunload', stopCamera);

// ── Upload / Drop ──────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadSourcePanel.style.display = 'none';
    cameraSourcePanel.style.display = 'none';
    imagePreview.style.display = 'block';
    analyzeBtn.disabled = false;
    // Trigger scan animation
    scanBar.classList.remove('active');
    void scanBar.offsetWidth;
    scanBar.classList.add('active');
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  analyzeBtn.disabled = true;
  showSourceInput();
  showEmpty();
});

// ── Analyze ───────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  modelSelect.disabled = true;
  analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing…';
  showLoading();

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('model', modelSelect.value);

  try {
    // Step 1
    activateStep(step1, step2, step3, 0);
    await delay(500);

    // Step 2
    activateStep(step1, step2, step3, 1);
    const response = await fetch('/predict', { method: 'POST', body: formData });
    await delay(300);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Server error');
    }

    // Step 3
    activateStep(step1, step2, step3, 2);
    await delay(300);

    const data = await response.json();
    await delay(300);

    showResults(data);

  } catch (err) {
    showEmpty();
    alert('Error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    modelSelect.disabled = false;
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Plant';
  }
}

function activateStep(s1, s2, s3, idx) {
  const steps = [s1, s2, s3];
  steps.forEach((s, i) => {
    s.classList.remove('active', 'done');
    if (i < idx)  s.classList.add('done');
    if (i === idx) s.classList.add('active');
  });
}

// ── Show / Hide States ────────────────────────────────────
function showEmpty() {
  emptyState.style.display = 'flex';
  loadingState.style.display = 'none';
  resultContent.style.display = 'none';
}

function showLoading() {
  emptyState.style.display = 'none';
  loadingState.style.display = 'flex';
  resultContent.style.display = 'none';
  // Reset steps
  [step1, step2, step3].forEach(s => s.classList.remove('active','done'));
}

function showResults(data) {
  loadingState.style.display = 'none';
  resultContent.style.display = 'flex';

  const t = data.treatment;

  // Thumbnail
  document.getElementById('resultThumb').src = `data:image/jpeg;base64,${data.image}`;

  // Disease name
  document.getElementById('diseaseName').textContent = t.disease_name || data.label;

  // Which model produced this result
  document.getElementById('resultModelName').textContent = data.model_used || '—';

  // Confidence bar
  const conf = data.confidence;
  document.getElementById('confidenceLabel').textContent = conf + '% Confidence';
  setTimeout(() => {
    document.getElementById('confidenceFill').style.width = conf + '%';
  }, 100);

  // Severity badge
  const badge = document.getElementById('severityBadge');
  const sev = (t.severity || 'Unknown').toLowerCase();
  badge.textContent = 'Severity: ' + (t.severity || 'Unknown');
  badge.className = 'severity-badge';
  if (sev === 'none') badge.classList.add('none');
  else if (sev === 'high') badge.classList.add('high');

  // Description
  document.getElementById('descriptionText').textContent = t.description || '—';

  // Symptoms
  const symptomsSec = document.getElementById('symptomsSection');
  const symptomsList = document.getElementById('symptomsList');
  if (t.symptoms && t.symptoms.length > 0) {
    symptomsList.innerHTML = t.symptoms.map(s => `<li>${s}</li>`).join('');
    symptomsSec.style.display = '';
  } else {
    symptomsSec.style.display = 'none';
  }

  // Treatment
  document.getElementById('treatmentList').innerHTML =
    (t.treatment || []).map(s => `<li>${s}</li>`).join('');

  // Prevention
  document.getElementById('preventionList').innerHTML =
    (t.prevention || []).map(s => `<li>${s}</li>`).join('');
}

// ── New Scan Button ───────────────────────────────────────
document.getElementById('newScanBtn').addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  analyzeBtn.disabled = true;
  showSourceInput();
  showEmpty();
});

// ── Utility ───────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
