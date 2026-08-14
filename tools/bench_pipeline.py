"""
Benchmark the recognition pipeline.

Two modes:

  stages     per-model latency — SCRFD across detector input sizes, the
             recogniser per face, the anti-spoofing pair, JPEG encode — swept
             over intra-op thread counts and ORT's spin-wait setting.

  pipeline   the real per-frame path (src.pipeline.recognize_frame) on real
             captured frames, comparing per-track recognition scheduling
             against embedding every face on every frame.

Read this before trusting any number here: on a 2-physical-core box the timings
drift enough between runs that a sequential A-then-B comparison is worthless.
The same detector config measured 12.7 ms and 19.2 ms on two runs of identical
code. So every measurement below is **interleaved** — each round runs every case
once and the reported figure is the median across rounds, which cancels thermal
drift and background load in a way that back-to-back runs cannot. The spread
column is there so you can see when a difference is smaller than the noise and
should not be acted on.

    python -m tools.bench_pipeline
    python -m tools.bench_pipeline --pipeline --camera 0 --cameras 4
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.antispoof import INPUT_SIZE, ensure_models  # noqa: E402
from src.face_engine import FaceEngine  # noqa: E402
from src.pipeline import recognize_frame  # noqa: E402
from src.runtime import physical_cores, thread_budget  # noqa: E402
from src.smoother import IdentitySmoother  # noqa: E402

DET_SIZES = [640, 480, 416, 320]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", action="store_true",
                   help="benchmark the end-to-end per-frame path instead of individual models")
    p.add_argument("--camera", default="0",
                   help="frame source for --pipeline (device index or RTSP URL)")
    p.add_argument("--image", help="use a still image instead of a camera for --pipeline")
    p.add_argument("--frames", type=int, default=60,
                   help="frames to capture and replay (--pipeline)")
    p.add_argument("--cameras", type=int, default=4,
                   help="how many cameras to project per-camera FPS for (--pipeline)")
    p.add_argument("--source-fps", type=float, default=30.0,
                   help="assumed capture rate, used to advance the pipeline's clock (--pipeline)")
    p.add_argument("--db", default="Dataset/faces.db", help="face index to match against")
    p.add_argument("--rounds", type=int, default=7, help="interleaved measurement rounds")
    p.add_argument("--iters", type=int, default=8, help="iterations timed per round")
    return p.parse_args()


# ---------------------------------------------------------------- measurement

def _time(fn, iters):
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000


def interleaved(cases, rounds, iters, warmup=3):
    """
    cases: {label: callable}. Runs one iteration of every case per round so all
    cases see the same machine conditions, and returns {label: (median, min)}.
    """
    for fn in cases.values():
        for _ in range(warmup):
            fn()
    samples = {k: [] for k in cases}
    for _ in range(rounds):
        for label, fn in cases.items():
            samples[label].append(_time(fn, iters))
    return {k: (statistics.median(v), min(v)) for k, v in samples.items()}


def report(title, measured, unit="ms"):
    print(f"\n{title}")
    width = max(len(k) for k in measured)
    print(f"  {'case'.ljust(width)}   median      best     spread")
    for label, (med, best) in measured.items():
        spread = (med - best) / med * 100 if med else 0.0
        print(f"  {label.ljust(width)}  {med:7.2f}{unit}  {best:7.2f}{unit}  {spread:5.1f}%")


# -------------------------------------------------------------------- stages

def make_session(path, threads, spinning):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning",
                                "1" if spinning else "0")
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def runner(session, tensor):
    name = session.get_inputs()[0].name
    feed = {name: tensor}
    return lambda: session.run(None, feed)


def bench_stages(args):
    from insightface.utils import ensure_available

    model_dir = ensure_available("models", "buffalo_sc", root="~/.insightface")
    det = os.path.join(model_dir, "det_500m.onnx")
    rec = os.path.join(model_dir, "w600k_mbf.onnx")
    budget = thread_budget()
    threads = sorted({1, budget, budget * 2})

    cases = {}
    for n in threads:
        for spin in (1, 0):
            session = make_session(det, n, spin)
            for size in DET_SIZES:
                tensor = np.random.rand(1, 3, size, size).astype(np.float32)
                cases[f"det {size:>3}  intra={n} spin={spin}"] = runner(session, tensor)
    report("SCRFD detection — input size x threads x spin-wait",
           interleaved(cases, args.rounds, max(2, args.iters // 2)))

    cases = {}
    tensor = np.random.rand(1, 3, 112, 112).astype(np.float32)
    for n in threads:
        for spin in (1, 0):
            cases[f"rec 112  intra={n} spin={spin}"] = runner(
                make_session(rec, n, spin), tensor)
    report("Recognition embedding — per face", interleaved(cases, args.rounds, args.iters))

    spoof_models = ensure_models()          # [(path, crop scale)]
    if spoof_models:
        cases = {}
        tensor = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        for path, _ in spoof_models:
            name = os.path.basename(path)
            for n in threads:
                cases[f"{name:<20} intra={n}"] = runner(
                    make_session(path, n, True), tensor)
        report("Anti-spoofing — one model, one crop",
               interleaved(cases, args.rounds, args.iters * 2))
    else:
        print("\nAnti-spoofing weights unavailable — skipped")

    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    cases = {}
    for n in threads:
        def encode(n=n):
            cv2.setNumThreads(n)
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cases[f"imencode 640x480 q80  cv2 threads={n}"] = encode
    report("JPEG encode", interleaved(cases, args.rounds, args.iters * 2))
    cv2.setNumThreads(budget)


# ------------------------------------------------------------------ pipeline

def capture(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read {args.image}")
        print(f"Replaying one still image {frame.shape[1]}x{frame.shape[0]} "
              f"x{args.frames} — a motionless subject flatters scheduling, so "
              f"prefer --camera for a representative number.")
        return [frame] * args.frames

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        sys.exit(f"Could not open camera {args.camera!r}. Pass --image instead.")
    print(f"Capturing {args.frames} frames from {args.camera!r} — "
          "stand in view and move as you normally would.")
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        sys.exit("Camera opened but produced no frames.")
    print(f"Captured {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


def replay(engine, frames, schedule, source_fps):
    """One pass over the frames. Returns (ms/frame, embeddings/frame, faces/frame)."""
    smoother = IdentitySmoother()
    counters = {"embeds": 0, "faces": 0}
    original_embed = engine.embed
    original_detect = engine.detect_faces

    def counting_embed(frame, face):
        counters["embeds"] += 1
        return original_embed(frame, face)

    def counting_detect(frame):
        faces = original_detect(frame)
        counters["faces"] += len(faces)
        return faces

    engine.embed, engine.detect_faces = counting_embed, counting_detect
    try:
        start = time.perf_counter()
        for i, frame in enumerate(frames):
            # a synthetic clock at the capture rate, so the re-verify interval
            # means the same thing it would in production regardless of how
            # fast this replay actually runs
            recognize_frame(engine, smoother, frame,
                            now=i / source_fps, schedule=schedule)
        elapsed = time.perf_counter() - start
    finally:
        engine.embed, engine.detect_faces = original_embed, original_detect
    n = len(frames)
    return elapsed / n * 1000, counters["embeds"] / n, counters["faces"] / n


def bench_pipeline(args):
    frames = capture(args)
    engine = FaceEngine(db_path=args.db)
    print(f"Face index: {len(engine.labels)} embeddings, "
          f"{len(set(engine.labels))} people")

    modes = {"scheduled (per-track)": True, "legacy (embed every face)": False}
    samples = {k: [] for k in modes}
    detail = {}
    for fn in modes.values():                      # warm up both paths
        replay(engine, frames[:5], fn, args.source_fps)
    for _ in range(args.rounds):
        for label, schedule in modes.items():      # interleaved, not A-then-B
            ms, embeds, faces = replay(engine, frames, schedule, args.source_fps)
            samples[label].append(ms)
            detail[label] = (embeds, faces)

    measured = {k: (statistics.median(v), min(v)) for k, v in samples.items()}
    report("Per-frame pipeline, one camera", measured)

    print(f"\n  embeddings per frame  "
          f"scheduled {detail['scheduled (per-track)'][0]:.2f}  "
          f"vs legacy {detail['legacy (embed every face)'][0]:.2f}  "
          f"({detail['legacy (embed every face)'][1]:.2f} faces detected per frame)")

    print(f"\n  Projected at {args.cameras} cameras on one worker thread:")
    for label, (med, _) in measured.items():
        print(f"    {label:<28} {1000 / (med * args.cameras):5.1f} FPS/camera")


def main():
    args = parse_args()
    ort.set_default_logger_severity(4)   # SCRFD floods warnings at non-default sizes
    print(f"physical cores {physical_cores()} / logical {os.cpu_count()}   "
          f"thread budget {thread_budget()}   ORT {ort.__version__}")
    if args.pipeline:
        bench_pipeline(args)
    else:
        bench_stages(args)


if __name__ == "__main__":
    main()
