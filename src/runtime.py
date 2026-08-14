"""
Thread budget for ONNX Runtime and OpenCV.

Every ONNX session, and OpenCV itself, sizes its own thread pool from the logical
CPU count unless told otherwise. This process runs four sessions — SCRFD and the
recognizer inside insightface, plus the two anti-spoofing models — alongside one
frame-grabber thread per camera, so the defaults oversubscribe badly.

Measured on a 2-physical-core i3-1115G4: SCRFD at 320x320 takes 4.7 ms on the auto
setting and 3.0 ms pinned to 2 intra-op threads. These models stop scaling at the
physical core count (intra_op=4 was no faster than intra_op=2 for every model in
the bundle), so the default budget is physical cores — not `os.cpu_count()`, which
counts hyperthreads. Override with `runtime.threads` in config.toml on a bigger
deployment box.

insightface does not accept a SessionOptions, and fails to say so: FaceAnalysis
forwards **kwargs to model_zoo.get_model, which reads `providers` and
`provider_options` and drops everything else on the floor. Passing sess_options
there raises nothing and changes nothing, which is worse than an error. The only
hook in the call path is the PickableInferenceSession that ModelRouter.get_model
resolves as a module global, so that is what insightface_session_options() swaps.
"""
import contextlib
import logging
import os

import cv2
import onnxruntime as ort

from src.config import config

logger = logging.getLogger(__name__)


def physical_cores():
    """Physical cores, not hyperthreads. Falls back to the logical count."""
    try:
        pairs, phys, core = set(), None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "physical id":
                    phys = value.strip()
                elif key == "core id":
                    core = value.strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return os.cpu_count() or 1


def thread_budget():
    """Intra-op threads per session. `runtime.threads` in config.toml (or the
    ORT_THREADS env var) wins; 0 means auto = physical cores."""
    configured = config.runtime.threads
    if configured and int(configured) > 0:
        return int(configured)
    return physical_cores()


def session_options(threads=None):
    n = threads or thread_budget()
    so = ort.SessionOptions()
    so.intra_op_num_threads = n
    # Our graphs are one serial chain; parallel branches to overlap don't exist,
    # so an inter-op pool would only add threads to contend with the intra-op one.
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Deliberately NOT disabling session.intra_op.allow_spinning. Turning the
    # spin-wait off looks right — idle ORT threads stop holding cores while the
    # loop grabs frames and encodes JPEGs — but measured at this thread budget it
    # is a straight loss: tools/bench_pipeline.py puts the recogniser at 6.7 ms
    # spinning against 9.3 ms not, and the detector 16.2 ms against 18.2 ms. The
    # per-op wake-up cost on these many-layered graphs outweighs the contention
    # it saves. Re-measure before changing it.
    return so


@contextlib.contextmanager
def insightface_session_options(threads=None):
    """
    Make FaceAnalysis build its sessions with our thread budget.

    Patches insightface.model_zoo.model_zoo.PickableInferenceSession for the
    duration of the block. ModelRouter.get_model looks that name up as a module
    global on every call, so rebinding it is enough — and it is the only point in
    the path that still has the kwargs, since get_model() has already discarded
    everything but `providers` by the time it constructs the session.
    """
    from insightface.model_zoo import model_zoo

    so = session_options(threads)
    original = model_zoo.PickableInferenceSession

    class _PinnedSession(original):
        def __init__(self, model_path, **kwargs):
            kwargs.setdefault("sess_options", so)
            super().__init__(model_path, **kwargs)

    model_zoo.PickableInferenceSession = _PinnedSession
    try:
        yield so
    finally:
        model_zoo.PickableInferenceSession = original


def configure_opencv(threads=None):
    """OpenCV keeps its own pool for imencode/resize/cvtColor — same budget."""
    cv2.setNumThreads(threads or thread_budget())
