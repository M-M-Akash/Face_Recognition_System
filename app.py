"""
FastAPI web service for the face recognition system.

- /                          dashboard: camera grid, enrollment, people, sightings
- /stream/{cam_id}           annotated MJPEG stream per camera
- /api/cameras               camera list + status; PATCH toggles enabled/recognition
- /api/recognition           POST {"paused": bool} — global recognition pause
- /api/enroll/start|cancel   guided enrollment control
- /api/enroll/status         live enrollment progress (polled by the UI)
- /api/people                list / delete enrolled people
- /api/events                recent sightings; /api/events/{id}/snapshot for the face crop

Capture, processing, and viewing are independent per camera:
- `enabled` off releases the device entirely (desired state, persisted in the DB);
- `recognition` off (or the global pause) streams raw frames with zero inference;
- JPEG encoding only happens while someone is actually watching the stream.
Recognition output is written to the `events` table (one sighting per person
per camera per EVENT_COOLDOWN_S), so the streams are just a view — the events
log is the product. When the anti-spoofing model is present, each event is
liveness-gated: SPOOF_VOTES scores are collected on consecutive frames, the
median is stored on the event, and events below SPOOF_THRESH are flagged
(`live: false` in the API) — attendance/door-unlock consumers must check it.

One background thread owns all camera reads and inference; request handlers
only read the latest JPEG or flip small state. Run with a SINGLE worker —
state is in-process:

    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from src.VideoStream import VideoStream
from src.antispoof import load_antispoof
from src.enrollment import SAMPLES_PER_POSE, GuidedEnrollment
from src.face_engine import FaceEngine, draw_results
from src.smoother import IdentitySmoother

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]
CAMERA_URLS_FILE = os.environ.get("CAMERA_URLS_FILE", "camera_urls.json")

JPEG_QUALITY = 80
RELOAD_CHECK_S = 2.0    # poll the store for out-of-process enrollments/deletes
RECONNECT_S = 5.0       # retry a dead camera this often
MAX_MISSED_READS = 30   # consecutive empty reads before a camera counts as dead
EVENT_COOLDOWN_S = 30   # min seconds between events for the same person+camera
SPOOF_THRESH = float(os.environ.get("SPOOF_THRESH", "0.5"))  # min P(live) for a clean sighting
SPOOF_VOTES = 3            # liveness scores (one per frame) collected before an event is written
SPOOF_PENDING_TTL_S = 5.0  # abandon a half-collected liveness check if the person vanishes
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _placeholder_jpeg(text):
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (20, 190), FONT, 0.8, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Camera:
    def __init__(self, cam_id, url, enabled=True, recognition=True):
        self.id = cam_id
        self.url = url
        self.enabled = enabled
        self.recognition = recognition
        self.stream = None
        self.smoother = IdentitySmoother()
        self.connected = False
        self.misses = 0
        self.last_attempt = 0.0
        self.viewers = 0
        self.idle_published = False
        self.jpeg = _placeholder_jpeg(f"Camera {cam_id}: starting...")
        self.seq = 0

    def publish(self, jpeg):
        self.jpeg = jpeg
        self.seq += 1

    def release(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.exit()
            self.stream = None
        self.connected = False


class AppState:
    def __init__(self):
        self.engine = None
        self.antispoof = None
        self.cameras = {}
        self.paused = False           # global recognition pause
        self.enroll = None            # current/last GuidedEnrollment
        self.lock = threading.Lock()  # guards enroll session swaps
        self.viewer_lock = threading.Lock()
        self.last_event_at = {}       # (cam_id, person) -> monotonic-ish timestamp
        self.pending_liveness = {}    # (cam_id, person) -> liveness votes being collected
        self.stop = threading.Event()
        self.thread = None


state = AppState()


def _read_frame(cam, now):
    """Grab the latest frame, handling connect/reconnect. None if unavailable."""
    if cam.stream is None:
        if now - cam.last_attempt < RECONNECT_S:
            return None
        cam.last_attempt = now
        vs = VideoStream(cam.url)
        if not vs.stream.isOpened():
            vs.stream.release()
            cam.publish(_placeholder_jpeg(f"Camera {cam.id}: no signal"))
            return None
        cam.stream = vs.start()
        cam.connected = True
        cam.misses = 0
        logger.info(f"Camera {cam.id} connected ({cam.url})")

    frame = cam.stream.read()
    if frame is None:
        cam.misses += 1
        if cam.misses >= MAX_MISSED_READS:
            logger.warning(f"Camera {cam.id} disconnected")
            cam.release()
            cam.publish(_placeholder_jpeg(f"Camera {cam.id}: disconnected"))
        return None
    cam.misses = 0
    return frame


def _face_snapshot(frame, bbox):
    """JPEG crop of the face with 25% padding, for the events log."""
    x1, y1, x2, y2 = bbox
    pad_w, pad_h = int(0.25 * (x2 - x1)), int(0.25 * (y2 - y1))
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def _emit_events(cam, frame, results, now):
    """One sighting per (camera, person) per cooldown window. Uses the
    smoothed label, so a sighting means the identity won the majority vote.

    When the anti-spoofing model is loaded, an event isn't written until
    SPOOF_VOTES liveness scores have been collected on consecutive frames;
    the median goes on the event, and a median below SPOOF_THRESH marks the
    sighting as a suspected spoof (still recorded, for the audit trail).
    Costs a few ms per event, nothing on ordinary frames."""
    for r in results:
        person = r["label"]
        if person == "unknown":
            continue
        key = (cam.id, person)
        if now - state.last_event_at.get(key, 0.0) < EVENT_COOLDOWN_S:
            continue

        if not state.antispoof.available:
            state.last_event_at[key] = now
            state.engine.store.add_event(person, cam.id, r["score"],
                                         _face_snapshot(frame, r["bbox"]))
            logger.info(f"Sighting: {person} on camera {cam.id} "
                        f"(similarity {r['score']:.2f})")
            continue

        pending = state.pending_liveness.get(key)
        if pending is None:
            pending = state.pending_liveness[key] = {
                "scores": [], "snapshot": _face_snapshot(frame, r["bbox"]),
                "similarity": r["score"], "last": now}
        pending["last"] = now
        score = state.antispoof.score(frame, r["bbox"])
        if score is not None:
            pending["scores"].append(score)
        if len(pending["scores"]) < SPOOF_VOTES:
            continue

        liveness = float(np.median(pending["scores"]))
        del state.pending_liveness[key]
        state.last_event_at[key] = now
        state.engine.store.add_event(person, cam.id, pending["similarity"],
                                     pending["snapshot"], liveness)
        if liveness < SPOOF_THRESH:
            logger.warning(f"SPOOF suspected: {person} on camera {cam.id} "
                           f"(liveness {liveness:.2f} < {SPOOF_THRESH})")
        else:
            logger.info(f"Sighting: {person} on camera {cam.id} (similarity "
                        f"{pending['similarity']:.2f}, liveness {liveness:.2f})")

    # a person who vanished mid-check shouldn't leave a stale entry behind
    for key in [k for k, p in state.pending_liveness.items()
                if k[0] == cam.id and now - p["last"] > SPOOF_PENDING_TTL_S]:
        del state.pending_liveness[key]


def _draw_enroll_overlay(frame, session, info):
    if info["bbox"] is not None:
        x1, y1, x2, y2 = info["bbox"]
        color = (0, 255, 0) if info["accepted"] else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    counts = "  ".join(f"{p} {n}/{SAMPLES_PER_POSE}" for p, n in info["counts"].items())
    cv2.putText(frame, f"ENROLLING: {session.person_name}", (10, 30), FONT, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, info["instruction"], (10, 60), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, counts, (10, 90), FONT, 0.6, (0, 255, 0), 2)
    if info["problem"]:
        cv2.putText(frame, info["problem"], (10, 120), FONT, 0.6, (0, 165, 255), 2)


def _process_loop():
    last_reload = 0.0
    while not state.stop.is_set():
        now = time.time()
        if now - last_reload >= RELOAD_CHECK_S:
            if state.engine.maybe_reload():
                logger.info("Store changed — reloaded face index")
            last_reload = now

        got_frame = False
        for cam in state.cameras.values():
            # reconcile desired state: disabled -> device released, stream parked
            if not cam.enabled:
                if cam.stream is not None:
                    cam.release()
                    logger.info(f"Camera {cam.id} stopped")
                if not cam.idle_published:
                    cam.publish(_placeholder_jpeg(f"Camera {cam.id}: stopped"))
                    cam.idle_published = True
                continue
            cam.idle_published = False

            frame = _read_frame(cam, now)
            if frame is None:
                continue
            got_frame = True

            session = state.enroll
            enrolling = (session is not None and session.state == "running"
                         and session.cam_id == cam.id)
            if enrolling:
                faces = state.engine.detect(frame)
                info = session.step(frame, faces)
                _draw_enroll_overlay(frame, session, info)
                if session.state in ("complete", "timeout"):
                    _, message = session.finish()
                    logger.info(message)
                    cam.smoother = IdentitySmoother()  # drop stale tracks
            elif cam.recognition and not state.paused:
                results = cam.smoother.update(state.engine.detect_and_recognize(frame))
                _emit_events(cam, frame, results, now)  # snapshot before boxes are drawn
                draw_results(frame, results)
            else:
                tag = "recognition paused" if state.paused else "recognition off"
                cv2.putText(frame, tag, (10, 25), FONT, 0.6, (140, 140, 140), 2)

            # encoding is demand-driven: skip it when nobody is watching
            if cam.viewers > 0 or enrolling:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    cam.publish(buf.tobytes())

        if not got_frame:
            time.sleep(0.1)


def _mjpeg_stream(cam):
    with state.viewer_lock:
        cam.viewers += 1
    try:
        last_seq = -1
        while not state.stop.is_set():
            if cam.seq != last_seq:
                last_seq = cam.seq
                jpg = cam.jpeg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            else:
                time.sleep(0.03)
    finally:
        with state.viewer_lock:
            cam.viewers -= 1


def _camera_payload(cam):
    session = state.enroll
    return {
        "id": cam.id,
        "connected": cam.connected,
        "enabled": cam.enabled,
        "recognition": cam.recognition,
        "viewers": cam.viewers,
        "enrolling": bool(session and session.state == "running" and session.cam_id == cam.id),
    }


def _enroll_payload():
    session = state.enroll
    if session is None:
        return {"active": False, "state": "idle"}
    return {
        "active": session.state == "running",
        "state": session.state,
        "name": session.person_name,
        "camera": session.cam_id,
        "counts": session.counts(),
        "needed": SAMPLES_PER_POSE,
        "instruction": session.last_instruction,
        "problem": session.last_problem,
        "warning": session.warning,
        "message": session.message,
        "saved": session.saved,
    }


app = FastAPI(title="Face Recognition")


@app.on_event("startup")
def startup():
    with open(CAMERA_URLS_FILE) as f:
        urls = json.load(f)
    state.engine = FaceEngine(providers=PROVIDERS)
    state.antispoof = load_antispoof(providers=PROVIDERS)
    store = state.engine.store
    state.cameras = {}
    for i, url in enumerate(urls):
        cam_id = i + 1
        enabled, recognition = store.camera_config(cam_id)
        state.cameras[cam_id] = Camera(cam_id, url, enabled, recognition)
    state.paused = store.get_setting("recognition_paused", "0") == "1"
    state.thread = threading.Thread(target=_process_loop, daemon=True)
    state.thread.start()
    logger.info(f"Started with {len(state.cameras)} camera(s), paused={state.paused}, "
                f"anti-spoofing={'on' if state.antispoof.available else 'OFF'}")


@app.on_event("shutdown")
def shutdown():
    state.stop.set()
    if state.thread:
        state.thread.join(timeout=5)
    for cam in state.cameras.values():
        cam.release()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/cameras")
def cameras():
    return {"paused": state.paused,
            "cameras": [_camera_payload(cam) for cam in state.cameras.values()]}


class CameraPatch(BaseModel):
    enabled: Optional[bool] = None
    recognition: Optional[bool] = None


@app.patch("/api/cameras/{cam_id}")
def patch_camera(cam_id: int, req: CameraPatch):
    cam = state.cameras.get(cam_id)
    if cam is None:
        raise HTTPException(404, f"No camera {cam_id}")
    state.engine.store.set_camera_config(cam_id, req.enabled, req.recognition)
    if req.enabled is not None:
        cam.enabled = req.enabled
        if req.enabled:
            cam.last_attempt = 0.0  # reconnect on the next loop pass
        else:
            with state.lock:
                session = state.enroll
                if session and session.state == "running" and session.cam_id == cam_id:
                    session.cancel()
    if req.recognition is not None:
        cam.recognition = req.recognition
        cam.smoother = IdentitySmoother()  # don't carry tracks across the toggle
    logger.info(f"Camera {cam_id}: enabled={cam.enabled} recognition={cam.recognition}")
    return _camera_payload(cam)


class PauseRequest(BaseModel):
    paused: bool


@app.post("/api/recognition")
def set_recognition(req: PauseRequest):
    state.paused = req.paused
    state.engine.store.set_setting("recognition_paused", "1" if req.paused else "0")
    logger.info(f"Global recognition pause: {state.paused}")
    return {"paused": state.paused}


@app.get("/stream/{cam_id}")
def stream(cam_id: int):
    cam = state.cameras.get(cam_id)
    if cam is None:
        raise HTTPException(404, f"No camera {cam_id}")
    return StreamingResponse(_mjpeg_stream(cam),
                             media_type="multipart/x-mixed-replace; boundary=frame")


class EnrollRequest(BaseModel):
    name: str
    camera: int


@app.post("/api/enroll/start")
def enroll_start(req: EnrollRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name required")
    cam = state.cameras.get(req.camera)
    if cam is None:
        raise HTTPException(404, f"No camera {req.camera}")
    if not cam.enabled:
        raise HTTPException(409, f"Camera {req.camera} is stopped — start it first")
    if not cam.connected:
        raise HTTPException(409, f"Camera {req.camera} is not connected")
    with state.lock:
        if state.enroll is not None and state.enroll.state == "running":
            raise HTTPException(409, "An enrollment is already running")
        state.enroll = GuidedEnrollment(name, state.engine, cam_id=req.camera,
                                        antispoof=state.antispoof,
                                        min_liveness=SPOOF_THRESH)
    logger.info(f"Enrollment started: '{name}' on camera {req.camera}")
    return _enroll_payload()


@app.post("/api/enroll/cancel")
def enroll_cancel():
    with state.lock:
        session = state.enroll
        if session is None or session.state != "running":
            raise HTTPException(409, "No enrollment running")
        session.cancel()  # commits nothing; safe from a request thread
    return _enroll_payload()


@app.get("/api/enroll/status")
def enroll_status():
    return _enroll_payload()


@app.get("/api/people")
def people():
    return [{"name": name, "samples": samples}
            for name, samples in state.engine.store.list_people()]


@app.delete("/api/people/{name}")
def delete_person(name: str):
    if not state.engine.store.delete_person(name):
        raise HTTPException(404, f"No person named '{name}'")
    logger.info(f"Deleted person '{name}'")
    # the processing loop's maybe_reload() picks this up within ~2s
    return {"deleted": name}


@app.get("/api/events")
def events(limit: int = 30):
    evs = state.engine.store.list_events(min(max(limit, 1), 200))
    for e in evs:
        # attendance / door-unlock consumers must honor this flag
        e["live"] = e["liveness"] is None or e["liveness"] >= SPOOF_THRESH
    return evs


@app.get("/api/events/{event_id}/snapshot")
def event_snapshot(event_id: int):
    snapshot = state.engine.store.event_snapshot(event_id)
    if snapshot is None:
        raise HTTPException(404, "No snapshot for this event")
    return Response(content=snapshot, media_type="image/jpeg")
