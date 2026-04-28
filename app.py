"""
ECG Digitization System — Flask Backend  (Production-Ready)
=============================================================
Run:  python app.py
Open: http://127.0.0.1:5000
"""

import os
import io
import sys
import time
import base64
import logging
import traceback
import requests

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


def download_model():
    url = "https://drive.google.com/uc?id=1-r_pKLKCmhdzfSquA-ia82N39WyY3AA6"
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")

        response = requests.get(url, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Model downloaded!")
# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('ecg')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'unet_ecg_best.pth')
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE       = 256
UNET_THRESHOLD = 0.5
MAX_IMG_BYTES  = 25 * 1024 * 1024
VALID_MIMES    = {
    'image/png', 'image/jpeg', 'image/jpg',
    'image/bmp', 'image/tiff', 'image/webp'
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = MAX_IMG_BYTES
CORS(app, resources={r"/*": {"origins": "*"}})

from flask import Flask, render_template

# ══════════════════════════════════════════════════════════════
# U-NET ARCHITECTURE
# ══════════════════════════════════════════════════════════════
class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = self.conv(x)
        return s, self.pool(s)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.enc1 = _EncoderBlock(in_ch,   64)
        self.enc2 = _EncoderBlock(64,     128)
        self.enc3 = _EncoderBlock(128,    256)
        self.enc4 = _EncoderBlock(256,    512)
        self.btn  = _DoubleConv(512,     1024)
        self.dec4 = _DecoderBlock(1024,   512)
        self.dec3 = _DecoderBlock(512,    256)
        self.dec2 = _DecoderBlock(256,    128)
        self.dec1 = _DecoderBlock(128,     64)
        self.out  = nn.Sequential(nn.Conv2d(64, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b      = self.btn(p4)
        d      = self.dec4(b,  s4)
        d      = self.dec3(d,  s3)
        d      = self.dec2(d,  s2)
        d      = self.dec1(d,  s1)
        return self.out(d)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING — lazy singleton
# ─────────────────────────────────────────────────────────────
_unet_model  = None
_model_error = None

def load_unet():
    download_model()
    global _unet_model, _model_error
    if _unet_model is not None:
        return _unet_model
    if _model_error is not None:
        return None
    if not os.path.exists(MODEL_PATH):
        _model_error = f"Model file not found: {MODEL_PATH}"
        log.warning(_model_error)
        return None
    try:
        log.info(f"Loading U-Net from {MODEL_PATH} on {DEVICE.upper()} ...")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            cfg   = ckpt.get('cfg', {})
            in_ch = cfg.get('in_channels',  1)
            ou_ch = cfg.get('out_channels', 1)
            state = ckpt['model_state']
        elif isinstance(ckpt, dict) and all(
            isinstance(v, torch.Tensor) for v in ckpt.values()
        ):
            in_ch, ou_ch = 1, 1
            state = ckpt
        else:
            raise ValueError("Unrecognised checkpoint format.")

        model = UNet(in_ch, ou_ch).to(DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()
        _unet_model = model
        log.info("U-Net loaded successfully.")
        return _unet_model

    except Exception as exc:
        _model_error = str(exc)
        log.error(f"Failed to load U-Net: {exc}")
        return None


# ══════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════

def stage_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image passed to grayscale stage.")
    if len(img_bgr.shape) == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def stage_threshold(gray: np.ndarray) -> np.ndarray:
    """CLAHE equalisation + Otsu binarization."""
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq   = clahe.apply(gray)
    _, binary = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def stage_remove_grid(binary: np.ndarray,
                      h_len: int = 40,
                      v_len: int = 40) -> np.ndarray:
    """
    Multi-pass morphological grid removal:
      Pass 1 — user-defined kernel lengths.
      Pass 2 — 1.5× longer kernel to catch faint lines.
      Pass 3 — noise cleanup via small opening.
    """
    def _detect(src, kh, kv):
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
        return cv2.add(
            cv2.morphologyEx(src, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(src, cv2.MORPH_OPEN, vk)
        )

    no_grid = cv2.subtract(binary, _detect(binary, h_len, v_len))
    no_grid = cv2.subtract(no_grid, _detect(no_grid, int(h_len * 1.5), int(v_len * 1.5)))

    # Remove isolated noise pixels
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    no_grid = cv2.morphologyEx(no_grid, cv2.MORPH_OPEN, noise_k)
    return no_grid


def stage_enhance(no_grid: np.ndarray,
                  dilate_iter: int = 2,
                  close_k:     int = 5) -> np.ndarray:
    """Dilation + morphological closing + median denoise."""
    dilate_iter = max(1, min(dilate_iter, 8))
    close_k     = max(3, min(close_k, 21))
    if close_k % 2 == 0:
        close_k += 1

    dk       = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated  = cv2.dilate(no_grid, dk, iterations=dilate_iter)
    ck       = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    enhanced = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, ck)
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced


def stage_unet_mask(img_bgr: np.ndarray) -> np.ndarray:
    """U-Net inference → binary mask at original resolution."""
    model = load_unet()
    if model is None:
        raise RuntimeError(
            f"U-Net unavailable. Ensure '{MODEL_PATH}' exists and is valid."
        )
    tf     = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    gray   = stage_grayscale(img_bgr)
    tensor = tf(Image.fromarray(gray)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(tensor).squeeze().cpu().numpy()

    mask = (prob > UNET_THRESHOLD).astype(np.uint8) * 255
    h, w = img_bgr.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def stage_extract_signal(enhanced: np.ndarray):
    """
    Column-wise centroid with gap interpolation and light Gaussian smoothing.
    Returns (x_float_array, y_float_array).
    """
    h, w = enhanced.shape
    x_raw, y_raw = [], []

    for col in range(w):
        active = np.where(enhanced[:, col] > 0)[0]
        if len(active) > 0:
            x_raw.append(float(col))
            y_raw.append(float(h) - float(np.mean(active)))

    if len(x_raw) < 2:
        return np.array(x_raw), np.array(y_raw)

    xa = np.array(x_raw)
    ya = np.array(y_raw)

    # Interpolate across gaps
    x_full = np.arange(xa[0], xa[-1] + 1, dtype=float)
    y_full = np.interp(x_full, xa, ya)

    # Gaussian smoothing (adaptive kernel)
    ks = max(3, min(7, (w // 100) * 2 + 1))
    y_full = cv2.GaussianBlur(
        y_full.reshape(1, -1).astype(np.float32), (ks, 1), 0
    ).flatten()

    return x_full, y_full


# ══════════════════════════════════════════════════════════════
# HEART RATE ESTIMATION
# ══════════════════════════════════════════════════════════════
def estimate_heart_rate(y: np.ndarray, image_width: int) -> str:
    """
    Robust HR estimation using bandpass filtering + scipy peak detection.
    Assumes a standard 10-second ECG strip width.
    Falls back to zero-crossing method if scipy is unavailable.
    Returns string like "72 BPM" or "N/A".
    """
    if y is None or len(y) < 20:
        return "N/A"

    try:
        from scipy.signal import find_peaks, butter, filtfilt

        # Bandpass filter to suppress baseline wander and HF noise
        if len(y) > 30:
            b, a   = butter(2, [0.01, 0.45], btype='band', fs=1.0)
            y_filt = filtfilt(b, a, y)
        else:
            y_filt = y.copy()

        # Normalise to [0, 1]
        span = y_filt.max() - y_filt.min()
        if span < 1e-6:
            return "N/A"
        y_norm = (y_filt - y_filt.min()) / span

        # Primary peak detection
        min_dist = max(5, image_width // 20)
        peaks, _ = find_peaks(y_norm, distance=min_dist,
                               prominence=0.20, height=0.40)
        if len(peaks) < 2:
            peaks, _ = find_peaks(y_norm, distance=min_dist, prominence=0.10)
        if len(peaks) < 2:
            return "N/A"

        rr = np.diff(peaks).astype(float)

        # Remove outliers (±3× median)
        med   = np.median(rr)
        valid = rr[(rr > 0.3 * med) & (rr < 3.0 * med)]
        if len(valid) == 0:
            valid = rr

        avg_rr          = float(np.mean(valid))
        pixels_per_sec  = max(image_width / 10.0, 1.0)
        rr_sec          = avg_rr / pixels_per_sec
        hr              = int(round(np.clip(60.0 / max(rr_sec, 1e-4), 30, 250)))
        return f"{hr} BPM"

    except ImportError:
        return _hr_zero_crossing(y, image_width)
    except Exception as exc:
        log.warning(f"HR estimation error: {exc}")
        return "N/A"


def _hr_zero_crossing(y: np.ndarray, image_width: int) -> str:
    try:
        mean       = np.mean(y)
        crossings  = np.where(np.diff(np.sign(y - mean)) > 0)[0]
        if len(crossings) < 2:
            return "N/A"
        avg_period = float(np.mean(np.diff(crossings)))
        pps        = max(image_width / 10.0, 1.0)
        hr         = int(round(np.clip(60.0 / (avg_period / pps), 30, 250)))
        return f"{hr} BPM"
    except Exception:
        return "N/A"


# ══════════════════════════════════════════════════════════════
# SIGNAL PLOT
# ══════════════════════════════════════════════════════════════
def make_signal_plot(x: np.ndarray, y: np.ndarray, hr_label: str = "") -> bytes:
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    if len(x) > 0 and len(y) > 0:
        ax.plot(x, y, color='#ff4f5e', linewidth=0.9, label='ECG Signal', alpha=0.92)
    else:
        ax.text(0.5, 0.5, 'No signal extracted',
                transform=ax.transAxes, ha='center', va='center',
                color='#888', fontsize=12)

    title = 'Extracted ECG Signal'
    if hr_label and hr_label != 'N/A':
        title += f'  |  HR ≈ {hr_label}'

    ax.set_title(title, fontsize=13, fontweight='bold', color='#e8edf5', pad=10)
    ax.set_xlabel('Time (pixels)', fontsize=10, color='#8899aa')
    ax.set_ylabel('Amplitude (pixels)', fontsize=10, color='#8899aa')
    ax.tick_params(colors='#8899aa', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#2a3a4a')
    ax.grid(True, linestyle='--', alpha=0.20, color='#aaaaaa')
    if len(x) > 0:
        ax.legend(loc='upper right', fontsize=9,
                  facecolor='#1a2535', edgecolor='#2a3a4a', labelcolor='#c0d0e0')

    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def ndarray_to_b64(arr: np.ndarray, ext: str = '.png') -> str:
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise RuntimeError(f"cv2.imencode({ext}) failed.")
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode()}"


def bytes_to_b64(raw: bytes, mime: str = 'image/png') -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _validate_upload(req):
    """Returns (img_bgr, error_string). img_bgr is None on error."""
    if 'file' not in req.files:
        return None, "No 'file' field in request."
    f = req.files['file']
    if not f or f.filename == '':
        return None, "Empty filename."
    mime = (f.mimetype or '').lower()
    if mime and mime not in VALID_MIMES and not mime.startswith('image/'):
        return None, f"Unsupported file type: {mime}"
    raw = f.read()
    if len(raw) == 0:
        return None, "Uploaded file is empty."
    if len(raw) > MAX_IMG_BYTES:
        return None, f"File too large ({len(raw)//1024//1024} MB). Max 25 MB."
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image. Ensure it is a valid PNG / JPG / BMP."
    if img.shape[0] < 32 or img.shape[1] < 32:
        return None, "Image too small (minimum 32 × 32 px)."
    return img, None


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(img_bgr:     np.ndarray,
                 h_kernel:    int = 40,
                 v_kernel:    int = 40,
                 dilate_iter: int = 2,
                 close_k:     int = 5,
                 mode:        str = 'classical') -> dict:

    t0 = time.perf_counter()

    # Stage 1 — Grayscale
    gray = stage_grayscale(img_bgr)

    # Stage 2 — Threshold
    binary = stage_threshold(gray)

    # Stage 3 — Grid removal
    no_grid = stage_remove_grid(binary, h_kernel, v_kernel)

    # Stage 4 — Enhancement / U-Net mask
    unet_ok   = False
    unet_mask = None

    if mode in ('unet', 'hybrid'):
        try:
            unet_mask = stage_unet_mask(img_bgr)
            unet_ok   = True
        except Exception as exc:
            log.warning(f"U-Net skipped: {exc}. Using classical fallback.")

    if mode == 'unet' and unet_ok:
        enhanced = unet_mask
    elif mode == 'hybrid' and unet_ok:
        classical = stage_enhance(no_grid, dilate_iter, close_k)
        # Merge: OR then restrict to dilated U-Net region (suppress FP)
        merged    = cv2.bitwise_or(classical, unet_mask)
        dil_unet  = cv2.dilate(
            unet_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        enhanced = cv2.bitwise_and(merged, dil_unet)
    else:
        enhanced = stage_enhance(no_grid, dilate_iter, close_k)

    # Stage 5 — Signal extraction
    x, y = stage_extract_signal(enhanced)

    # Stage 6 — HR estimation & plot
    img_w    = img_bgr.shape[1]
    hr_str   = estimate_heart_rate(y, img_w) if len(y) >= 10 else "N/A"
    plot_png = make_signal_plot(x, y, hr_label=hr_str)

    # Coverage metric (replaces fake Dice)
    covered  = int(np.sum(np.any(enhanced > 0, axis=0)))
    coverage = f"{round(covered / max(enhanced.shape[1], 1) * 100, 1)}%"

    elapsed  = round((time.perf_counter() - t0) * 1000)
    eff_mode = mode if (mode == 'classical' or unet_ok) else 'classical (fallback)'

    return {
        'success': True,
        'stages': {
            'grayscale': ndarray_to_b64(gray),
            'binary':    ndarray_to_b64(binary),
            'nogrid':    ndarray_to_b64(no_grid),
            'enhanced':  ndarray_to_b64(enhanced),
        },
        'signal_plot': bytes_to_b64(plot_png),
        'metrics': {
            'points':  f"{len(x):,}",
            'hr':      hr_str,
            'dice':    coverage,
            'time_ms': f"{elapsed} ms",
        },
        'meta': {
            'mode':       eff_mode,
            'unet_used':  unet_ok,
            'image_size': f"{img_bgr.shape[1]}×{img_bgr.shape[0]}",
        }
    }


# ══════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    try:
        img_bgr, err = _validate_upload(request)
        if err:
            return jsonify({'success': False, 'error': err}), 400

        def _int(key, default, lo, hi):
            try:
                return max(lo, min(hi, int(request.form.get(key, default))))
            except (TypeError, ValueError):
                return default

        mode     = request.form.get('model', 'classical').strip().lower()
        h_kernel = _int('h_kernel',     40,  5, 200)
        v_kernel = _int('v_kernel',     40,  5, 200)
        dilate   = _int('dilate_iter',   2,  1,   8)
        close_k  = _int('close_kernel',  5,  3,  21)

        if mode not in ('classical', 'unet', 'hybrid'):
            mode = 'classical'

        result = run_pipeline(img_bgr, h_kernel, v_kernel, dilate, close_k, mode)
        log.info(
            f"OK | {result['meta']['image_size']} | mode={result['meta']['mode']} "
            f"| pts={result['metrics']['points']} | HR={result['metrics']['hr']} "
            f"| {result['metrics']['time_ms']}"
        )
        return jsonify(result)

    except Exception:
        log.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error':   'Internal server error. Check server logs for details.'
        }), 500


@app.route('/health')
def health():
    model = load_unet()
    return jsonify({
        'status':      'ok',
        'device':      DEVICE.upper(),
        'unet_loaded': model is not None,
        'unet_error':  _model_error,
        'model_path':  MODEL_PATH,
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({'success': False, 'error': 'File too large. Max 25 MB.'}), 413


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({'success': False, 'error': 'Method not allowed.'}), 405


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info("=" * 60)
    log.info("  ECG Digitization System — Flask Backend")
    log.info("=" * 60)
    log.info(f"  Device     : {DEVICE.upper()}")
    log.info(f"  Model path : {MODEL_PATH}")
    load_unet()
    log.info("  Server     : http://127.0.0.1:5000")
    log.info("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)