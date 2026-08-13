"""
Face detection + recognition engine using insightface (ONNX runtime).
Replaces the paddlehub + insightface_paddle stack.
"""
import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.face_store import FaceStore

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self, db_path="Dataset/faces.db", rec_thresh=0.45,
                 det_size=(640, 640), providers=None):
        """
        Args:
            db_path: SQLite database holding people + embeddings (see FaceStore)
            rec_thresh: cosine similarity threshold for recognition
            det_size: detection input size (smaller = faster, larger = better small-face recall)
            providers: ONNX runtime providers. None = CPU. ["CUDAExecutionProvider"] for GPU.
        """
        self.rec_thresh = rec_thresh
        self.store = FaceStore(db_path)

        # 'buffalo_sc' is the small CPU-friendly bundle (SCRFD detector + MobileFaceNet-style recognizer).
        # Use 'buffalo_l' if you have GPU and want higher accuracy.
        self.app = FaceAnalysis(
            name="buffalo_sc",
            providers=providers or ["CPUExecutionProvider"],
        )
        # ctx_id=-1 forces CPU; set to 0 for GPU
        ctx_id = 0 if providers and "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # one-time migration from the legacy pickle index
        legacy = self.store.db_path.with_name("index.pkl")
        if self.store.count() == 0 and legacy.exists():
            n = self.store.migrate_from_pickle(legacy)
            logger.info(f"Migrated {n} embeddings from {legacy} to {self.store.db_path}")

        self._loaded_version = -1
        self._reload()

    def _reload(self):
        self.labels, self.embeddings = self.store.load_all()
        self._loaded_version = self.store.version()
        self._refresh_norm_cache()

    def maybe_reload(self):
        """
        Hot-reload the index if another process (e.g. enrollment) wrote to the
        store since we last loaded. Cheap to poll. Returns True if reloaded.
        """
        if self.store.version() != self._loaded_version:
            self._reload()
            return True
        return False

    def _refresh_norm_cache(self):
        # pre-normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings_norm = self.embeddings / norms

    def detect(self, frame_bgr):
        """
        Single detection + embedding inference pass. Returns insightface Face
        objects (bbox, kps, det_score, normed_embedding) — no index matching.
        """
        return self.app.get(frame_bgr)

    def match(self, embedding):
        """
        Match one L2-normalized embedding against the index.
        Returns (label, cosine_similarity); label is "unknown" below rec_thresh.
        """
        if len(self.labels) == 0:
            return "unknown", 0.0
        sims = self.embeddings_norm @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < self.rec_thresh:
            return "unknown", best_sim
        return self.labels[best_idx], best_sim

    def detect_and_recognize(self, frame_bgr):
        """
        Returns list of dicts: [{"bbox": (x1,y1,x2,y2), "label": str, "score": float}, ...]
        """
        results = []
        for f in self.detect(frame_bgr):
            label, score = self.match(f.normed_embedding)
            x1, y1, x2, y2 = f.bbox.astype(int)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "score": float(score),
                "det_score": float(f.det_score),
            })
        return results

    def add_embedding(self, person_name, embedding):
        """Persist one embedding to the store and refresh the in-memory index."""
        self.store.add_embedding(person_name, embedding)
        self._reload()


def draw_results(frame, results):
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        label = f"{r['label']} {r['score']:.2f}"
        color = (0, 255, 0) if r["label"] != "unknown" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
