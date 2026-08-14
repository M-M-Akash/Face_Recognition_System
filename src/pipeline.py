"""
The per-frame recognition pipeline.

One function, shared by the web service and tools/bench_pipeline.py, so the
benchmark measures the code that actually runs rather than a copy of it that
quietly drifts out of date.
"""
import time


def recognize_frame(engine, smoother, frame, now=None, schedule=True):
    """
    Detect faces, run recognition on the ones that need it, and return the
    smoothed results.

    `schedule=False` embeds every face on every frame — the behaviour from
    before per-track scheduling existed. It is kept so the benchmark can A/B the
    two on identical frames, which is the only way to get a trustworthy number
    on a machine with few cores and noisy timings. Production wants True.
    """
    now = time.monotonic() if now is None else now
    faces = engine.detect_faces(frame)
    tracks = smoother.associate([(f.bbox, f.det_score) for f in faces], now)
    for face, track in zip(faces, tracks):
        if schedule and not track.needs_embed:
            continue
        label, score = engine.match(engine.embed(frame, face))
        smoother.record(track, label, score, now)
    return smoother.results()
