"""
Face detection + recognition engine using insightface (ONNX runtime).
Replaces the paddlehub + insightface_paddle stack.
"""
import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from src.config import config
from src.face_store import FaceStore
from src.runtime import insightface_session_options

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self, db_path=None, rec_thresh=None, det_size=None,
                 providers=None):
        """
        Every argument defaults to config.toml; pass one only to override it.

        Args:
            db_path: SQLite database holding people + embeddings (see FaceStore)
            rec_thresh: cosine similarity threshold for recognition
            det_size: detection input size (smaller = faster, larger = better small-face recall)
            providers: ONNX runtime providers. None = whatever [runtime] gpu says.
        """
        db_path = config.paths.database if db_path is None else db_path
        rec_thresh = config.detection.rec_thresh if rec_thresh is None else rec_thresh
        det_size = config.detection.det_size_wh() if det_size is None else det_size
        providers = config.providers() if providers is None else providers
        self.rec_thresh = rec_thresh
        self.store = FaceStore(db_path)

        # 'buffalo_sc' is the small CPU-friendly bundle (SCRFD detector + MobileFaceNet-style recognizer).
        # Use 'buffalo_l' if you have GPU and want higher accuracy.
        # The context manager is what pins the ONNX thread budget — FaceAnalysis
        # silently ignores a sess_options kwarg, see src/runtime.py.
        with insightface_session_options():
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

    def detect_faces(self, frame_bgr):
        """
        Detection only — the embedding model is not run. Returns Face objects
        with bbox/kps/det_score but no `normed_embedding`; pass each to embed()
        for the ones you actually need identified.

        Splitting the two halves of app.get() is the point: detection costs the
        same regardless of who is in frame, but embedding costs ~7.7 ms *per
        face* and is pure waste on a track whose identity is already settled.
        """
        bboxes, kpss = self.app.det_model.detect(frame_bgr, max_num=0,
                                                 metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            faces.append(Face(bbox=bboxes[i, 0:4],
                              kps=kpss[i] if kpss is not None else None,
                              det_score=bboxes[i, 4]))
        return faces

    def embed(self, frame_bgr, face):
        """
        Run the recognition model on one detected face, filling its
        `normed_embedding` in place. `frame_bgr` must be the full-resolution
        frame the face was detected in — the aligned 112x112 crop is warped
        from it, not from the detector's downscaled input.
        """
        self.app.models["recognition"].get(frame_bgr, face)
        return face.normed_embedding

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

    def add_embedding(self, person_name, embedding):
        """Persist one embedding to the store and refresh the in-memory index."""
        self.store.add_embedding(person_name, embedding)
        self._reload()


def draw_results(frame, results):
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        if r.get("live") is False:
            # Deliberately show the spoof verdict INSTEAD of the name: a
            # presentation attack that displays the victim's identity on screen
            # is exactly the outcome the liveness check exists to prevent.
            color = (0, 0, 255)                       # red (BGR)
            label = f"SPOOF? {r['liveness']:.2f}"
        else:
            color = (0, 255, 0) if r["label"] != "unknown" else (0, 165, 255)
            label = f"{r['label']} {r['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
