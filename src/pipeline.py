"""
The per-frame recognition pipeline.

One function, shared by the web service and tools/bench_pipeline.py, so the
benchmark measures the code that actually runs rather than a copy of it that
quietly drifts out of date.
"""
import time


def recognize_frame(engine, smoother, frame, now=None, schedule=True,
                    antispoof=None):
    """
    Detect faces, run recognition and liveness on the ones that need it, and
    return the smoothed results.

    `antispoof` is optional: without it, results carry `liveness=None` and
    `live=True`, so a deployment with no anti-spoofing weights behaves exactly
    as it did before.

    `schedule=False` embeds and scores every face on every frame — the behaviour
    from before per-track scheduling existed. It is kept so the benchmark can A/B
    the two on identical frames, which is the only way to get a trustworthy
    number on a machine with few cores and noisy timings. Production wants True.
    """
    now = time.monotonic() if now is None else now
    faces = engine.detect_faces(frame)
    tracks = smoother.associate([(f.bbox, f.det_score) for f in faces], now)
    live_model = antispoof is not None and antispoof.available
    for face, track in zip(faces, tracks):
        if not schedule or track.needs_embed:
            label, score = engine.match(engine.embed(frame, face))
            smoother.record(track, label, score, now)
        if live_model and (not schedule or track.needs_liveness):
            # the raw detection, not the smoothed display box: scaled_crop's
            # geometry is what SPOOF_THRESH was measured against
            bbox = tuple(int(round(v)) for v in track.raw_bbox)
            smoother.record_liveness(track, antispoof.score(frame, bbox), now)
    return smoother.results()
