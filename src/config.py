"""
Single source of truth for every tunable in the system.

Settings resolve in three layers, each overriding the one before:

    built-in DEFAULTS  <  config.toml  <  environment variables

The file is the place a human edits. Environment variables stay on top so a
container, a CI run, or a one-off `SPOOF_THRESH=0.6 python ...` can override a
setting without editing a file that is bind-mounted read-only — which is exactly
what docker-compose does. Only the handful of settings listed in ENV_OVERRIDES
can be set that way; everything else lives in the file.

TOML because Python 3.11 reads it with stdlib `tomllib` (no dependency) and it
takes comments, which a file whose purpose is to explain the knobs needs.

Usage:

    from src.config import config
    if config.runtime.gpu: ...
    engine = FaceEngine(det_size=config.detection.det_size_wh())

Unknown keys are warned about rather than ignored: a typo in a config file that
silently does nothing is worse than one that complains.
"""
import logging
import os
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("FR_CONFIG", "config.toml")

# Every setting, with the value used when config.toml is absent or silent.
# This dict is also the schema: a key not present here is flagged as unknown.
DEFAULTS = {
    "runtime": {
        "gpu": False,
        # 0 = auto-detect physical cores. Raising this past the physical core
        # count measured no faster; see src/runtime.py.
        "threads": 0,
        # Let ORT threads spin while idle. False is right for this camera-paced
        # workload — same fps, a third less CPU. See session_options().
        "spin_wait": False,
    },
    "cameras": {
        # Device indices (ints) or RTSP/HTTP URLs (strings), mixed freely.
        "urls": [0],
        # Which camera the terminal enrollment tool opens. null = the first above.
        "enroll_camera": None,
    },
    "paths": {
        "database": "Dataset/faces.db",
        "captures": "Dataset/captures",
        "spoof_models": "models",
    },
    "detection": {
        # Detector input size. Cost scales with area: 640 -> 12.7 ms, 320 -> 3.0 ms
        # on a 2-core CPU, at the price of missing smaller/more distant faces.
        "det_size": 640,
        # Cosine similarity below which a face is "unknown".
        "rec_thresh": 0.45,
    },
    "smoothing": {
        "window": 10,          # recognitions that vote per track
        "min_votes": 6,        # votes a name needs before it is displayed
        "iou_thresh": 0.3,     # min IOU to associate a detection with a track
        "max_misses": 5,       # frames without a detection before a track dies
        "reverify_s": 1.0,     # how often a settled track re-runs recognition
        "bbox_alpha": 0.4,     # box smoothing; 1.0 disables it
        "snap_iou": 0.5,       # below this the face moved fast — snap, don't lag
    },
    "antispoof": {
        # Re-derive on new hardware with tools/eval_antispoof.py. This value came
        # from our own capture conditions (ACER 2.62%).
        "threshold": 0.48,
        "votes": 3,            # scores before a track's liveness is trusted
        "window": 5,           # rolling scores kept; the median is the verdict
    },
    "events": {
        "cooldown_s": 30,      # min seconds between events for the same person+camera
        "keep": 500,           # events retained; older ones are pruned
    },
    "enrollment": {
        "samples_per_pose": 5,
        "max_seconds": 90,
        "min_det_score": 0.6,
        "min_face_px": 80,     # smallest acceptable face box side
        "min_blur_var": 20.0,  # variance of Laplacian on the face crop
        "dedup_sim": 0.99,     # skip samples near-identical to one already taken
        "sample_cooldown_s": 0.25,
    },
    "stream": {
        "jpeg_quality": 80,
        "capture_timeout_s": 2.0,   # how long /api/capture waits for a live face
    },
    "timing": {
        "reload_check_s": 2.0,      # poll the store for out-of-process enrollments
        "reconnect_s": 5.0,         # retry a dead camera this often
        "max_missed_reads": 30,     # empty reads before a camera counts as dead
        "frame_wait_s": 0.005,      # cameras live, but no new frame yet
        "idle_sleep_s": 0.1,        # nothing connected — don't spin while retrying
        "fps_alpha": 0.2,           # smoothing for the dashboard FPS readout
    },
}


def _as_bool(raw):
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# (section, key) -> (env var, parser). Kept deliberately small: these are the
# settings a deployment overrides per-environment rather than per-install.
ENV_OVERRIDES = {
    ("runtime", "gpu"): ("USE_GPU", _as_bool),
    ("runtime", "threads"): ("ORT_THREADS", int),
    ("antispoof", "threshold"): ("SPOOF_THRESH", float),
    ("paths", "spoof_models"): ("SPOOF_MODEL_DIR", str),
    ("paths", "captures"): ("CAPTURE_DIR", str),
    ("paths", "database"): ("DB_PATH", str),
}


class Section(dict):
    """A config section. Both `config.runtime.gpu` and `config["runtime"]["gpu"]`
    work; a missing key raises rather than returning None, so a typo surfaces at
    the point of use instead of silently disabling a feature."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"No config key '{key}' in this section. "
                f"Known keys: {', '.join(sorted(self))}") from None


class DetectionSection(Section):
    def det_size_wh(self):
        """insightface wants a (w, h) tuple; the file takes a single number."""
        n = int(self["det_size"])
        return (n, n)


class Config(Section):
    # Where the settings came from. Set by load(); reported at startup, because
    # this module is imported before an entry point calls logging.basicConfig,
    # so anything logged in here goes nowhere.
    source = "built-in defaults"

    def providers(self):
        """ONNX Runtime provider list. One place, so the GPU switch is one edit."""
        if self.runtime.gpu:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def ctx_id(self):
        """insightface's device id: 0 for CUDA, -1 to force CPU."""
        return 0 if self.runtime.gpu else -1

    def enroll_camera(self):
        chosen = self.cameras.enroll_camera
        if chosen is not None:
            return chosen
        urls = self.cameras.urls
        return urls[0] if urls else 0


def _merge(defaults, loaded, source, path=()):
    """Overlay `loaded` onto `defaults`, warning about keys we don't know."""
    out = {}
    for key, fallback in defaults.items():
        if isinstance(fallback, dict):
            out[key] = _merge(fallback, loaded.get(key) or {}, source, path + (key,))
        else:
            out[key] = loaded[key] if key in loaded else fallback
    for key in loaded:
        if key not in defaults:
            where = ".".join(path + (key,))
            logger.warning(f"Unknown setting '{where}' in {source} — ignored. "
                           f"Check src/config.py for the valid keys.")
    return out


def _apply_env(data):
    for (section, key), (var, parse) in ENV_OVERRIDES.items():
        raw = os.environ.get(var)
        if raw is None or raw == "":
            continue
        try:
            data[section][key] = parse(raw)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring {var}={raw!r} — not a valid "
                           f"{parse.__name__} for {section}.{key}")
    return data


def _wrap(data):
    sections = {"detection": DetectionSection}
    return Config({k: sections.get(k, Section)(v) if isinstance(v, dict) else v
                   for k, v in data.items()})


def load(path=None):
    """Read the config file, layer environment overrides on top, and return it.
    A missing file is fine — the defaults are a working configuration."""
    path = Path(path or CONFIG_PATH)
    loaded = {}
    if path.is_file():
        with open(path, "rb") as f:
            loaded = tomllib.load(f)
    cfg = _wrap(_apply_env(_merge(DEFAULTS, loaded, path)))
    overridden = sorted(var for _, (var, _) in ENV_OVERRIDES.items()
                        if os.environ.get(var))
    cfg.source = (f"{path}" if path.is_file()
                  else f"built-in defaults ({path} not found)")
    if overridden:
        cfg.source += f", overridden by {', '.join(overridden)}"
    return cfg


config = load()
