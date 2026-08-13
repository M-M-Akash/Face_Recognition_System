"""
Pose-guided enrollment logic shared by the CLI (enroll_new_faces.py) and the
web service (app.py): walk the person through center / left / right head
poses, quality-gate every sample, and commit embeddings only at the end.

The pose choreography doubles as active liveness (a flat photo can't turn
its head); pass an AntiSpoof instance to also passively liveness-check every
sample, so nobody gets enrolled from a photo or a screen.
"""
import time

import cv2
import numpy as np

SAMPLES_PER_POSE = 5
MAX_SECONDS = 90          # give up if the poses aren't filled by then
MIN_DET_SCORE = 0.6
MIN_FACE_PX = 80          # smallest acceptable face box side, in pixels
MIN_BLUR_VAR = 20.0       # variance of Laplacian on the face crop
DEDUP_SIM = 0.99          # skip samples nearly identical to one already taken
SAMPLE_COOLDOWN = 0.25    # seconds between accepted samples

# Yaw-ratio bins (see estimate_yaw). The gap between CENTER_MAX and SIDE_MIN
# is a dead zone so no samples are taken while the head is mid-turn.
CENTER_MAX = 0.12
SIDE_MIN = 0.25

POSES = [
    ("center", "Look straight at the camera"),
    ("left", "Turn your head slowly to your LEFT"),
    ("right", "Turn your head slowly to your RIGHT"),
]
INSTRUCTIONS = dict(POSES)


def estimate_yaw(kps):
    """
    Cheap yaw proxy from the detector's 5-point keypoints — no landmark model
    needed, so it works with the buffalo_sc bundle.

    kps rows: [left_eye, right_eye, nose, left_mouth, right_mouth], image coords.
    Returns the nose's horizontal offset from the eye midpoint, normalized by
    inter-eye distance: ~0 when frontal, positive when the person turns to
    THEIR left, negative to their right. |0.25|+ is roughly a 25-35 degree turn.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-6:
        return 0.0
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    return float((nose[0] - eye_mid_x) / inter_eye)


def pose_of(yaw):
    """Map a yaw ratio to a pose name, or None while the head is mid-turn."""
    if abs(yaw) <= CENTER_MAX:
        return "center"
    if yaw >= SIDE_MIN:
        return "left"
    if yaw <= -SIDE_MIN:
        return "right"
    return None


def sample_problem(frame, face):
    """Return a short reason if the face isn't good enough to sample, else None."""
    if face.det_score < MIN_DET_SCORE:
        return "low detection confidence"
    x1, y1, x2, y2 = face.bbox.astype(int)
    if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
        return "move closer"
    h, w = frame.shape[:2]
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if crop.size == 0:
        return "face out of frame"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR_VAR:
        return "too blurry"
    return None


class GuidedEnrollment:
    """
    Frame-at-a-time enrollment state machine.

    Feed each frame plus the faces engine.detect() found in it to step();
    the returned dict drives UI feedback (instruction, progress, problems).
    When state leaves "running" ("complete" / "timeout" / "cancelled"),
    call finish() once to commit or discard the samples. Nothing touches
    the store until finish().
    """

    def __init__(self, person_name, engine, cam_id=None, max_seconds=MAX_SECONDS,
                 antispoof=None, min_liveness=0.5):
        self.person_name = person_name
        self.engine = engine
        self.cam_id = cam_id
        self.antispoof = antispoof
        self.min_liveness = min_liveness
        self.captured = {pose: [] for pose, _ in POSES}
        self.session_embs = []          # everything captured, for dedup
        self.deadline = time.time() + max_seconds
        self.last_sample = 0.0
        self.state = "running"
        self.warning = None             # duplicate-person warning, set once
        self.saved = 0
        self.message = ""
        self.last_instruction = ""
        self.last_problem = None
        self._checked_duplicate = False
        self._finished = False

    def counts(self):
        return {pose: len(self.captured[pose]) for pose, _ in POSES}

    def total(self):
        return len(self.session_embs)

    def _pending(self):
        return [p for p, _ in POSES if len(self.captured[p]) < SAMPLES_PER_POSE]

    def step(self, frame, faces):
        """Process one frame. Returns a dict for UI feedback."""
        info = {"state": self.state, "counts": self.counts(), "instruction": "",
                "problem": None, "bbox": None, "accepted": False,
                "warning": self.warning}
        if self.state != "running":
            return info

        pending = self._pending()
        if time.time() > self.deadline:
            self.state = "timeout"
            info["state"] = "timeout"
            return info
        instruction = INSTRUCTIONS[pending[0]]

        face = None
        if faces:
            # largest face = the person being enrolled
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        problem = "no face detected" if face is None else sample_problem(frame, face)
        accepted = False
        if face is not None:
            info["bbox"] = tuple(int(v) for v in face.bbox)
        if face is not None and problem is None:
            pose = pose_of(estimate_yaw(face.kps))
            now = time.time()
            if pose in pending and now - self.last_sample >= SAMPLE_COOLDOWN:
                emb = face.normed_embedding.astype(np.float32)
                if self.session_embs and float(np.max(np.stack(self.session_embs) @ emb)) > DEDUP_SIM:
                    problem = "too similar to a previous sample"
                elif self._spoofed(frame, face):
                    problem = "looks like a photo/screen - use a live face"
                else:
                    self.captured[pose].append(emb)
                    self.session_embs.append(emb)
                    self.last_sample = now
                    accepted = True

        # One-time check against the existing index once the frontal samples
        # are in, so re-enrolling someone under a new name gets flagged.
        if (not self._checked_duplicate and self.engine.labels
                and len(self.captured["center"]) >= SAMPLES_PER_POSE):
            mean = np.mean(self.captured["center"], axis=0)
            mean /= np.linalg.norm(mean)
            label, sim = self.engine.match(mean)
            if label != "unknown" and label != self.person_name:
                self.warning = (f"This face matches already-enrolled '{label}' "
                                f"(similarity {sim:.2f})")
            self._checked_duplicate = True

        if not self._pending():
            self.state = "complete"

        self.last_instruction = instruction
        self.last_problem = problem
        info.update(state=self.state, counts=self.counts(), instruction=instruction,
                    problem=problem, accepted=accepted, warning=self.warning)
        return info

    def _spoofed(self, frame, face):
        """Liveness-gate a would-be sample. Only runs on frames that passed
        every other check, so it costs a few ms per accepted sample."""
        if self.antispoof is None or not self.antispoof.available:
            return False
        score = self.antispoof.score(frame, face.bbox.astype(int))
        return score is not None and score < self.min_liveness

    def cancel(self):
        if self.state == "running":
            self.state = "cancelled"
            self.finish()

    def finish(self):
        """Commit or discard the session's samples. Idempotent.
        Returns (saved_count, message)."""
        if self._finished:
            return self.saved, self.message
        self._finished = True
        total = self.total()
        if self.state == "cancelled":
            self.message = "Enrollment cancelled — nothing saved."
        elif total == 0 or (self.state == "timeout" and total < SAMPLES_PER_POSE):
            self.message = (f"Timed out with only {total} samples — nothing saved. "
                            "Try better lighting or move closer.")
        else:
            for pose, _ in POSES:
                for emb in self.captured[pose]:
                    self.engine.add_embedding(self.person_name, emb)
            self.saved = total
            unfilled = self._pending()
            note = f" (timed out before filling: {', '.join(unfilled)})" if unfilled else ""
            self.message = f"Saved {total} embeddings for '{self.person_name}'{note}."
        return self.saved, self.message
