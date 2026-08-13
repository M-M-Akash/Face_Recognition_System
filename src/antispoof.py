"""
Passive presentation-attack detection (anti-spoofing / liveness).

Wraps the Apache-2.0 MiniFASNet ensemble from
github.com/minivision-ai/Silent-Face-Anti-Spoofing, using the ONNX exports
published by github.com/yakhyo/face-anti-spoofing. Two checkpoints vote:

    MiniFASNetV2     crop scale 2.7, 80x80
    MiniFASNetV1SE   crop scale 4.0, 80x80

The scale is baked into each checkpoint's training — it is how much context
around the face the model expects to see — so it travels with the filename and
must not be "unified". Both produce a 3-class softmax (class 1 = real) and the
per-model P(live) values are averaged, matching minivision's own test.py.

Preprocessing must match these exports exactly, and is *not* what any of the
upstream READMEs imply:

  * BGR, straight from cv2 — no channel swap
  * raw 0-255 floats — dividing by 255 saturates the exports to a constant
    output regardless of input, which looks like a working model until you
    measure it
  * the crop is clamped to the frame, never zero-padded

Weights are not vendored. `ensure_models()` fetches them on first use and
verifies SHA-256; see MODELS below for the pinned digests. If the download
fails, `available` is False and the system degrades to recognition-only rather
than refusing to start.

tools/eval_antispoof.py measures whether this actually separates live faces
from attacks on *your* camera. Do not trust a spoof model here without running
it — on our own footage the previous CelebA-Spoof-trained backend scored
ACER 38% (unusable) where this one scores 2.6%.
"""
import hashlib
import logging
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SPOOF_MODEL_DIR", "models")
RELEASE = ("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights")

# (filename, crop scale, sha256)
MODELS = [
    ("MiniFASNetV2.onnx", 2.7,
     "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"),
    ("MiniFASNetV1SE.onnx", 4.0,
     "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676"),
]
INPUT_SIZE = 80
LIVE_CLASS = 1          # 3-class output: class 1 is the bona-fide class
DOWNLOAD_TIMEOUT = 60


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(model_dir=None):
    """Download any missing weights into `model_dir` and verify their digests.

    Returns the list of (path, scale) that are present and valid. Downloads to
    a .part file and renames only after the hash checks out, so an interrupted
    run can't leave a truncated model that loads and scores garbage.
    """
    d = model_dir or MODEL_DIR
    os.makedirs(d, exist_ok=True)
    ready = []
    for name, scale, digest in MODELS:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            if _sha256(path) == digest:
                ready.append((path, scale))
                continue
            logger.warning(f"{path} failed its checksum — re-downloading")
            os.remove(path)

        url = f"{RELEASE}/{name}"
        tmp = path + ".part"
        try:
            logger.info(f"Downloading anti-spoofing weights: {url}")
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r, \
                    open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            got = _sha256(tmp)
            if got != digest:
                os.remove(tmp)
                logger.error(f"Checksum mismatch for {name}: expected {digest}, "
                             f"got {got} — refusing to use it")
                continue
            os.replace(tmp, path)
            logger.info(f"Saved {path}")
            ready.append((path, scale))
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            logger.warning(f"Could not download {name} ({e}) — liveness checks "
                           "will be disabled")
    return ready


def scaled_crop(img, bbox, scale):
    """Crop `scale` x the bbox, centered on the face, clamped to the frame so
    the result is never zero-padded (it just gets smaller near an edge). The
    scale is also capped so the box can always fit. Aspect ratio is the
    bbox's, not square. Mirrors AntiSpoofingONNX._crop_face in
    github.com/yakhyo/face-anti-spoofing, which is where these weights and
    their exact preprocessing come from."""
    src_h, src_w = img.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    box_w, box_h = bx2 - bx1, by2 - by1
    if box_w <= 0 or box_h <= 0:
        return None

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = bx1 + box_w / 2, by1 + box_h / 2

    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(src_w - 1, int(cx + new_w / 2))
    y2 = min(src_h - 1, int(cy + new_h / 2))

    crop = img[y1:y2 + 1, x1:x2 + 1]
    return crop if crop.size else None


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class MiniVisionAntiSpoof:
    """MiniFASNet ensemble. `.score(frame_bgr, bbox)` -> mean P(live) in [0,1],
    or None when unavailable or the crop is degenerate."""

    def __init__(self, providers=None, model_dir=None, download=True, size=INPUT_SIZE):
        self.size = size
        self.sessions = []      # (session, input_name, scale, path)
        found = (ensure_models(model_dir) if download
                 else [(os.path.join(model_dir or MODEL_DIR, n), s)
                       for n, s, _ in MODELS
                       if os.path.isfile(os.path.join(model_dir or MODEL_DIR, n))])
        for path, scale in found:
            sess = ort.InferenceSession(
                path, providers=providers or ["CPUExecutionProvider"])
            self.sessions.append((sess, sess.get_inputs()[0].name, scale, path))
        if self.sessions:
            names = ", ".join(f"{os.path.basename(p)}@{s}" for _, _, s, p in self.sessions)
            logger.info(f"Anti-spoofing ensemble loaded ({names}, input {size}x{size})")
        else:
            logger.warning("No anti-spoofing weights available — liveness checks disabled")

    @property
    def available(self):
        return bool(self.sessions)

    def score(self, frame_bgr, bbox):
        if not self.sessions:
            return None
        total, n = 0.0, 0
        for sess, input_name, scale, _ in self.sessions:
            crop = scaled_crop(frame_bgr, bbox, scale)
            if crop is None:
                continue
            # BGR, raw 0-255: these exports fold the scaling in, and feeding
            # /255 saturates them to a constant output regardless of input.
            img = cv2.resize(crop, (self.size, self.size))
            blob = img.transpose(2, 0, 1)[None].astype(np.float32)
            out = sess.run(None, {input_name: blob})[0][0]
            total += float(_softmax(out)[LIVE_CLASS])
            n += 1
        return total / n if n else None


def load_antispoof(providers=None, **kwargs):
    """Build the anti-spoofing backend, downloading weights if needed."""
    return MiniVisionAntiSpoof(providers=providers, **kwargs)


AntiSpoof = MiniVisionAntiSpoof
