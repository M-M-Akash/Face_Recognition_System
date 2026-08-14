"""
Temporal identity smoothing for video recognition.

Per-frame matching flickers: one borderline frame can flash the wrong name. This
associates detections across frames into tracks by bbox IOU, majority-votes the
label over a short window, and exponentially smooths the displayed box so the
rectangle stops jittering on a still face.

It also decides *when recognition needs to run at all*, which is where the CPU
goes. Detection costs the same regardless of who is in frame, but embedding costs
~7.7 ms per face, and re-embedding a track whose identity is already settled
learns nothing. So a track runs recognition every frame until it settles, then
drops to an occasional re-check.

Backing off any earlier would be a mistake worth spelling out: the votes in the
window *are* recognitions, not frames. Embedding a fresh track once per second
would stretch confirmation from ~6 frames to ~6 seconds. Every-frame-until-
settled keeps confirmation latency exactly where it was.

A track un-settles the moment a re-check disagrees with its established identity,
which is what makes an identity swap converge in ~6 frames instead of ~6 seconds
when one person steps out of frame and another steps into the same spot.

Pure bookkeeping — no model inference happens in here.
"""
import time
from collections import Counter, deque
from statistics import median

REVERIFY_S = 1.0      # how often a settled track re-runs recognition
BBOX_ALPHA = 0.4      # EMA weight for the displayed box; 1.0 disables smoothing
SNAP_IOU = 0.5        # below this the face moved fast — snap, don't lag behind it
LIVENESS_VOTES = 3    # scores collected before a track's liveness is trusted
LIVENESS_WINDOW = 5   # rolling scores kept per track; the median is the verdict


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class _Track:
    def __init__(self, bbox, window):
        self.bbox = tuple(float(v) for v in bbox)      # smoothed, for display
        self.raw_bbox = self.bbox                      # last raw detection
        self.embed_bbox = self.bbox                    # raw bbox at last embed
        self.votes = deque(maxlen=window)              # (label, score) per recognition
        self.misses = 0
        self.det_score = 0.0
        self.last_embed = 0.0
        self.settled = False        # identity established; safe to re-check lazily
        self.needs_embed = True     # set per frame by associate()
        self.liveness_scores = deque(maxlen=LIVENESS_WINDOW)
        self.last_liveness = 0.0
        self.needs_liveness = True  # set per frame by associate()

    @property
    def liveness(self):
        """Median P(live) over the recent window, or None if never scored."""
        if not self.liveness_scores:
            return None
        return median(self.liveness_scores)

    def observe(self, bbox, alpha):
        raw = tuple(float(v) for v in bbox)
        if _iou(raw, self.raw_bbox) < SNAP_IOU:
            self.bbox = raw
        else:
            self.bbox = tuple(s + alpha * (r - s) for s, r in zip(self.bbox, raw))
        self.raw_bbox = raw


class IdentitySmoother:
    """
    One instance per camera.

    Args:
        window: how many recent recognitions vote per track
        min_votes: votes a label needs in the window before it is displayed
        iou_thresh: min IOU to associate a detection with an existing track
        max_misses: drop a track after this many frames without a detection
        reverify_s: how often a settled track re-runs recognition
        bbox_alpha: EMA weight for the displayed box (1.0 = raw, no smoothing)
        spoof_thresh: min median P(live) for a track to count as a live face
    """

    def __init__(self, window=10, min_votes=6, iou_thresh=0.3, max_misses=5,
                 reverify_s=REVERIFY_S, bbox_alpha=BBOX_ALPHA, spoof_thresh=0.5):
        self.window = window
        self.min_votes = min_votes
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.reverify_s = reverify_s
        self.bbox_alpha = bbox_alpha
        self.spoof_thresh = spoof_thresh
        self.tracks = []
        self.embeds = 0     # cumulative recognitions recorded; see record()

    # -- the three-call path: associate -> embed only what needs it -> results --

    def associate(self, detections, now=None):
        """
        Match this frame's detections to tracks. `detections` is a sequence of
        (bbox, det_score). Returns the matched track per detection, in the same
        order; check `track.needs_embed` to decide whether to run recognition.
        """
        now = time.monotonic() if now is None else now
        for t in self.tracks:
            t.misses += 1

        matched, claimed = [], set()
        for bbox, det_score in detections:
            best, best_iou = None, self.iou_thresh
            for t in self.tracks:
                if id(t) in claimed:
                    continue
                iou = _iou(bbox, t.raw_bbox)
                if iou >= best_iou:
                    best, best_iou = t, iou
            if best is None:
                best = _Track(bbox, self.window)
                self.tracks.append(best)
            claimed.add(id(best))

            best.misses = 0
            best.det_score = float(det_score)
            best.observe(bbox, self.bbox_alpha)
            best.needs_embed = self._needs_embed(best, now)
            best.needs_liveness = self._needs_liveness(best, now)
            matched.append(best)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return matched

    def record(self, track, label, score, now=None):
        """Feed one recognition result into a track's vote window."""
        now = time.monotonic() if now is None else now
        self.embeds += 1
        previous, _ = self._verdict(track)
        track.votes.append((label, score))
        track.last_embed = now
        track.embed_bbox = track.raw_bbox

        if previous != "unknown" and label != previous:
            # a re-check disagreed with the established identity — go back to
            # every-frame recognition until the window reasserts itself
            track.settled = False
        else:
            verdict, _ = self._verdict(track)
            # settled means "there is nothing more to learn by asking again":
            # either a name has won, or the window is full and stayed unknown
            # (a stranger standing in frame shouldn't cost full price forever)
            track.settled = (verdict != "unknown"
                             or len(track.votes) == self.window)

    def record_liveness(self, track, score, now=None):
        """Feed one anti-spoofing score into a track's liveness window. A None
        score (degenerate crop, or no model) is not evidence of an attack and is
        discarded — the same fail-open rule GuidedEnrollment._spoofed uses."""
        track.last_liveness = time.monotonic() if now is None else now
        if score is not None:
            track.liveness_scores.append(float(score))

    def results(self):
        """Result dicts for the tracks seen in the most recent frame."""
        out = []
        for t in self.tracks:
            if t.misses:
                continue
            label, score = self._verdict(t)
            raw_label, raw_score = t.votes[-1] if t.votes else ("unknown", 0.0)
            liveness = t.liveness
            out.append({
                # smoothed box for drawing; raw box for anything that needs to
                # be positionally accurate (face crops, anti-spoofing)
                "bbox": tuple(int(round(v)) for v in t.bbox),
                "raw_bbox": tuple(int(round(v)) for v in t.raw_bbox),
                "label": label,
                "score": score,
                "raw_label": raw_label,
                "raw_score": raw_score,
                "det_score": t.det_score,
                # liveness is None until the model has been run; `live` fails
                # open in that case, so an absent anti-spoofing model never
                # blocks recognition. Consumers that must not be fooled should
                # check liveness_n >= LIVENESS_VOTES before trusting `live`.
                "liveness": liveness,
                "liveness_n": len(t.liveness_scores),
                "live": liveness is None or liveness >= self.spoof_thresh,
            })
        return out

    # -- internals --

    def _needs_liveness(self, track, now):
        # Score every frame until the window has enough samples to be trusted,
        # then re-check lazily. Sampling lazily from the start would delay the
        # first trustworthy verdict by LIVENESS_VOTES seconds, which would show
        # up as events that take seconds to appear.
        if len(track.liveness_scores) < LIVENESS_VOTES:
            return True
        if now - track.last_liveness >= self.reverify_s:
            return True
        # the box jumped — this may not be the same face we scored
        return _iou(track.raw_bbox, track.embed_bbox) < SNAP_IOU

    def _needs_embed(self, track, now):
        if not track.settled:
            return True
        if now - track.last_embed >= self.reverify_s:
            return True
        # the box jumped since we last looked — it may not be the same person
        return _iou(track.raw_bbox, track.embed_bbox) < SNAP_IOU

    def _verdict(self, track):
        if not track.votes:
            return "unknown", 0.0
        counts = Counter(label for label, _ in track.votes)
        label, n = counts.most_common(1)[0]
        if label == "unknown" or n < self.min_votes:
            return "unknown", track.votes[-1][1]
        scores = [s for l, s in track.votes if l == label]
        return label, sum(scores) / len(scores)
