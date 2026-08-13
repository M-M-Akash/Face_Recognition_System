"""
Temporal identity smoothing for video recognition.

Per-frame matching flickers: one borderline frame can flash the wrong name.
This associates detections across frames into tracks by bbox IOU and only
displays an identity once it wins a majority vote over a short window.
Pure bookkeeping — no extra model inference per frame.
"""
from collections import Counter, deque


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
        self.bbox = bbox
        self.votes = deque(maxlen=window)  # (label, score) per frame
        self.misses = 0


class IdentitySmoother:
    """
    One instance per camera.

    Args:
        window: how many recent frames vote per track
        min_votes: votes a label needs in the window before it is displayed
        iou_thresh: min IOU to associate a detection with an existing track
        max_misses: drop a track after this many frames without a detection
    """

    def __init__(self, window=7, min_votes=4, iou_thresh=0.3, max_misses=5):
        self.window = window
        self.min_votes = min_votes
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.tracks = []

    def update(self, results):
        """
        Smooth detect_and_recognize() results in place and return them.
        The raw per-frame values are preserved as 'raw_label' / 'raw_score'.
        """
        for t in self.tracks:
            t.misses += 1

        claimed = set()
        for r in results:
            best, best_iou = None, self.iou_thresh
            for t in self.tracks:
                if id(t) in claimed:
                    continue
                iou = _iou(r["bbox"], t.bbox)
                if iou >= best_iou:
                    best, best_iou = t, iou
            if best is None:
                best = _Track(r["bbox"], self.window)
                self.tracks.append(best)
            claimed.add(id(best))

            best.bbox = r["bbox"]
            best.misses = 0
            best.votes.append((r["label"], r["score"]))
            r["raw_label"], r["raw_score"] = r["label"], r["score"]
            r["label"], r["score"] = self._verdict(best)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return results

    def _verdict(self, track):
        counts = Counter(label for label, _ in track.votes)
        label, n = counts.most_common(1)[0]
        if label == "unknown" or n < self.min_votes:
            return "unknown", track.votes[-1][1]
        scores = [s for l, s in track.votes if l == label]
        return label, sum(scores) / len(scores)
