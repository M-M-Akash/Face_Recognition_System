"""
The per-frame recognition pipeline.

One function, shared by the web service and tools/bench_pipeline.py, so the
benchmark measures the code that actually runs rather than a copy of it that
quietly drifts out of date.
"""
import time


def recognize_frame(engine, smoother, frame, now=None, schedule=True,
                    antispoof=None, timings=None):
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

    Pass a dict as `timings` to have per-stage milliseconds accumulated into it
    (`detect`, `embed`, `liveness`). Costs two perf_counter calls per stage, so
    it is safe to leave on; callers that don't care pass nothing.
    """
    now = time.monotonic() if now is None else now
    clock = time.perf_counter

    t0 = clock()
    faces = engine.detect_faces(frame)
    detect_ms = (clock() - t0) * 1000

    tracks = smoother.associate([(f.bbox, f.det_score) for f in faces], now)
    live_model = antispoof is not None and antispoof.available
    embed_ms = liveness_ms = 0.0
    for face, track in zip(faces, tracks):
        if not schedule or track.needs_embed:
            t0 = clock()
            label, score = engine.match(engine.embed(frame, face))
            embed_ms += (clock() - t0) * 1000
            smoother.record(track, label, score, now)
        if live_model and (not schedule or track.needs_liveness):
            # the raw detection, not the smoothed display box: scaled_crop's
            # geometry is what SPOOF_THRESH was measured against
            bbox = tuple(int(round(v)) for v in track.raw_bbox)
            t0 = clock()
            score = antispoof.score(frame, bbox)
            liveness_ms += (clock() - t0) * 1000
            smoother.record_liveness(track, score, now)
    if timings is not None:
        timings["detect"] = detect_ms
        timings["embed"] = embed_ms
        timings["liveness"] = liveness_ms
    return smoother.results()
