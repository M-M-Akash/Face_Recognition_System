"""
FastAPI web service for the face recognition system.

- /                          dashboard: camera grid, enrollment, people, sightings
- /stream/{cam_id}           annotated MJPEG stream per camera
- /api/cameras               camera list + status; PATCH toggles enabled/recognition
- /api/recognition           POST {"paused": bool} — global recognition pause
- /api/enroll/start|cancel   guided enrollment control
- /api/enroll/status         live enrollment progress (polled by the UI)
- /api/people                list / delete enrolled people
- /api/events                recent sightings; /api/events/{id}/snapshot for legacy crops
- /api/capture               POST {"camera": id} — take an attendance photo
- /api/captures              recent photos; /api/captures/{id}/photo serves the file

Capture, processing, and viewing are independent per camera:
- `enabled` off releases the device entirely (desired state, persisted in the DB);
- `recognition` off (or the global pause) streams raw frames with zero inference;
- JPEG encoding only happens while someone is actually watching the stream.
Recognition output is written to the `events` table (one sighting per person
per camera per EVENT_COOLDOWN_S), so the streams are just a view — the events
log is the product. Sightings store no image; photos are taken deliberately.

Liveness lives on the track, scored on a schedule by the smoother, so every
result carries `liveness` and `live`. Three things consume it: a face below
SPOOF_THRESH draws a red SPOOF? box instead of a name, /api/capture refuses to
save a photo of it, and events record the median for the audit trail (still
written when spoofed, flagged `live: false` — attendance/door-unlock consumers
must check that flag).

One background thread owns all camera reads and inference; request handlers
only read the latest JPEG or flip small state. Run with a SINGLE worker —
state is in-process:

    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from src.VideoStream import VideoStream
from src.antispoof import load_antispoof
from src.config import CONFIG_PATH, config
from src.enrollment import SAMPLES_PER_POSE, GuidedEnrollment
from src.face_engine import FaceEngine, draw_results
from src.pipeline import recognize_frame
from src.runtime import configure_opencv
from src.smoother import IdentitySmoother

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Everything tunable lives in config.toml; see src/config.py for the layering.
PROVIDERS = config.providers()
JPEG_QUALITY = config.stream.jpeg_quality
RELOAD_CHECK_S = config.timing.reload_check_s
RECONNECT_S = config.timing.reconnect_s
MAX_MISSED_READS = config.timing.max_missed_reads
EVENT_COOLDOWN_S = config.events.cooldown_s
SPOOF_THRESH = config.antispoof.threshold
SPOOF_VOTES = config.antispoof.votes
CAPTURE_DIR = config.paths.captures
CAPTURE_TIMEOUT_S = config.stream.capture_timeout_s
FPS_ALPHA = config.timing.fps_alpha
FRAME_WAIT_S = config.timing.frame_wait_s
IDLE_SLEEP_S = config.timing.idle_sleep_s
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _new_smoother():
    """Every smoother needs the same spoof threshold — it decides `live` on each
    result, which drives both the red box and the capture gate."""
    return IdentitySmoother(spoof_thresh=SPOOF_THRESH)


def _placeholder_jpeg(text):
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (20, 190), FONT, 0.8, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class CaptureRequest:
    """A pending 'take a photo' ask, handed from a request thread to the worker.

    The worker owns every frame (see the module docstring), so the handler can't
    grab one itself. It parks one of these on the camera instead and waits. The
    worker only resolves it once it can decide — a face present and scored —
    which is what lets someone press the button a moment before stepping into
    view. The handler's timeout is what ends an ask nobody ever satisfies.
    """

    def __init__(self):
        self.done = threading.Event()
        self.error = None       # human-readable refusal, None on success
        self.payload = None     # capture row on success

    def resolve(self, payload=None, error=None):
        self.payload, self.error = payload, error
        self.done.set()


class Camera:
    def __init__(self, cam_id, url, enabled=True, recognition=True):
        self.id = cam_id
        self.url = url
        self.enabled = enabled
        self.recognition = recognition
        self.stream = None
        self.smoother = _new_smoother()
        self.connected = False
        self.misses = 0
        self.last_attempt = 0.0
        self.viewers = 0
        self.idle_published = False
        self.jpeg = _placeholder_jpeg(f"Camera {cam_id}: starting...")
        self.seq = 0
        self.fps = 0.0          # EMA of processed frames/sec — the number to tune on
        self.embeds = 0.0       # EMA of recognitions run per frame; see tick()
        self.last_frame_at = 0.0
        self.last_seq = -1      # last VideoStream frame we actually processed
        self.capture = None     # pending CaptureRequest, serviced by the worker
        self.spoof = False      # latest frame showed a suspected presentation attack

    def publish(self, jpeg):
        self.jpeg = jpeg
        self.seq += 1

    def tick(self, now, embeds=0):
        """
        Record that a frame was processed, for the dashboard readout. `embeds` is
        how many recognitions that frame actually ran — the number that says
        whether per-track scheduling is engaging. It should sit near zero while
        known people stand in view, and rise only while someone is unidentified.
        """
        if self.last_frame_at:
            dt = now - self.last_frame_at
            if dt > 0:
                instant = 1.0 / dt
                self.fps = instant if not self.fps else self.fps + FPS_ALPHA * (instant - self.fps)
        self.embeds += FPS_ALPHA * (embeds - self.embeds)
        self.last_frame_at = now

    def release(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.exit()
            self.stream = None
        self.connected = False
        self.fps = 0.0
        self.embeds = 0.0
        self.last_frame_at = 0.0
        self.last_seq = -1
        if self.capture is not None:
            self.capture.resolve(error=f"Camera {self.id} was stopped")
            self.capture = None


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

    seq, frame = cam.stream.read_latest()
    if frame is None:
        cam.misses += 1
        if cam.misses >= MAX_MISSED_READS:
            logger.warning(f"Camera {cam.id} disconnected")
            cam.release()
            cam.publish(_placeholder_jpeg(f"Camera {cam.id}: disconnected"))
        return None
    cam.misses = 0
    if seq == cam.last_seq:
        # the grabber hasn't produced a new frame since our last pass; running
        # inference again on the same buffer would burn cores for no new result
        return None
    cam.last_seq = seq
    return frame


def _safe_name(person):
    """A person's name reaches us from enrollment, so it is not a safe filename."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in person)
    return cleaned.strip("_") or "unknown"


def _service_capture(cam, frame, results, now):
    """Take a deliberate attendance photo, if the scene allows it right now.

    Returns without resolving the request when there is nothing to decide on yet
    (no face, or liveness not sampled) so the worker retries on the next frame.
    A spoof is a decision, not a retry: it resolves immediately as a refusal.

    `frame` must be the clean frame — this runs before draw_results, so the
    stored photo has no boxes burnt into it.
    """
    request = cam.capture
    if not results:
        return                       # nobody in view yet — keep waiting
    # largest face wins, the same convention capture_pad_dataset.py uses
    target = max(results, key=lambda r: ((r["raw_bbox"][2] - r["raw_bbox"][0]) *
                                         (r["raw_bbox"][3] - r["raw_bbox"][1])))

    if state.antispoof.available:
        if target["liveness_n"] < SPOOF_VOTES:
            return                   # still scoring — keep waiting
        if not target["live"]:
            cam.capture = None
            request.resolve(error=(f"Looks like a photo or screen "
                                   f"(liveness {target['liveness']:.2f} < {SPOOF_THRESH}) "
                                   f"— use a live face"))
            logger.warning(f"Capture refused on camera {cam.id}: suspected spoof "
                           f"(liveness {target['liveness']:.2f})")
            return

    person = target["label"]
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(now))
    day = stamp.split("_")[0]
    directory = Path(CAPTURE_DIR) / day
    directory.mkdir(parents=True, exist_ok=True)
    relative = f"{day}/{_safe_name(person)}_{stamp.split('_')[1]}_cam{cam.id}.jpg"
    path = Path(CAPTURE_DIR) / relative
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        cam.capture = None
        request.resolve(error="Could not write the photo to disk")
        logger.error(f"Capture failed: cv2.imwrite returned False for {path}")
        return

    capture_id = state.engine.store.add_capture(
        person, cam.id, relative, target["score"], target["liveness"])
    cam.capture = None
    request.resolve(payload={
        "id": capture_id, "person": person, "camera": cam.id,
        "path": relative, "similarity": target["score"],
        "liveness": target["liveness"],
    })
    logger.info(f"Captured {person} on camera {cam.id} -> {path}")


def _emit_events(cam, results, now):
    """One sighting per (camera, person) per cooldown window. Uses the
    smoothed label, so a sighting means the identity won the majority vote.

    Liveness comes from the track (see IdentitySmoother.record_liveness) rather
    than being collected here: one source of truth, and no second set of
    anti-spoofing calls on the same face. When the model is loaded, an event
    waits until the track has SPOOF_VOTES scores, so a sighting is never written
    on a liveness verdict too thin to trust. A median below SPOOF_THRESH still
    records the sighting, flagged, for the audit trail.

    No image is stored. Attendance photos are taken deliberately via
    /api/capture; a sighting is a log line, not a picture."""
    for r in results:
        person = r["label"]
        if person == "unknown":
            continue
        key = (cam.id, person)
        if now - state.last_event_at.get(key, 0.0) < EVENT_COOLDOWN_S:
            continue

        liveness = r["liveness"]
        if state.antispoof.available and r["liveness_n"] < SPOOF_VOTES:
            continue    # not enough scores yet — try again next frame

        state.last_event_at[key] = now
        state.engine.store.add_event(person, cam.id, r["score"], None, liveness)
        if liveness is None:
            logger.info(f"Sighting: {person} on camera {cam.id} "
                        f"(similarity {r['score']:.2f})")
        elif liveness < SPOOF_THRESH:
            logger.warning(f"SPOOF suspected: {person} on camera {cam.id} "
                           f"(liveness {liveness:.2f} < {SPOOF_THRESH})")
        else:
            logger.info(f"Sighting: {person} on camera {cam.id} (similarity "
                        f"{r['score']:.2f}, liveness {liveness:.2f})")


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

            embeds = 0
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
                    cam.smoother = _new_smoother()  # drop stale tracks
            elif cam.recognition and not state.paused:
                before = cam.smoother.embeds
                results = recognize_frame(state.engine, cam.smoother, frame, now,
                                          antispoof=state.antispoof)
                embeds = cam.smoother.embeds - before
                cam.spoof = any(not r["live"] for r in results)
                _emit_events(cam, results, now)
                if cam.capture is not None:      # capture the clean frame,
                    _service_capture(cam, frame, results, now)   # before boxes
                draw_results(frame, results)
            else:
                tag = "recognition paused" if state.paused else "recognition off"
                cv2.putText(frame, tag, (10, 25), FONT, 0.6, (140, 140, 140), 2)

            cam.tick(now, embeds)

            # encoding is demand-driven: skip it when nobody is watching
            if cam.viewers > 0 or enrolling:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    cam.publish(buf.tobytes())

        if not got_frame:
            # Distinguish "cameras are live, just haven't produced a new frame
            # yet" from "nothing is connected". The first wants a short wait so
            # we pick the next frame up promptly; the second wants a long one so
            # reconnect attempts don't spin a core.
            live = any(c.enabled and c.stream is not None for c in state.cameras.values())
            time.sleep(FRAME_WAIT_S if live else IDLE_SLEEP_S)


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
        "fps": round(cam.fps, 1),
        "embeds": round(cam.embeds, 2),
        "spoof": cam.spoof,
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
    urls = config.cameras.urls
    if not urls:
        raise RuntimeError("No cameras configured — set `urls` under [cameras] "
                           f"in {CONFIG_PATH}")
    if Path("camera_urls.json").is_file():
        logger.warning("camera_urls.json is no longer read — cameras now come "
                       f"from [cameras] urls in {CONFIG_PATH}. Delete the old file.")
    configure_opencv()   # keep OpenCV's pool inside the same budget as ORT's
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
    logger.info(f"Config: {config.source}")
    logger.info(f"Started with {len(state.cameras)} camera(s), paused={state.paused}, "
                f"anti-spoofing={'on' if state.antispoof.available else 'OFF'}, "
                f"gpu={'on' if config.runtime.gpu else 'off'}")


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
        cam.smoother = _new_smoother()  # don't carry tracks across the toggle
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


class CaptureCommand(BaseModel):
    camera: int


@app.post("/api/capture")
def capture(req: CaptureCommand):
    """Take an attendance photo. Refuses a suspected spoof."""
    cam = state.cameras.get(req.camera)
    if cam is None:
        raise HTTPException(404, f"No camera {req.camera}")
    if not cam.enabled:
        raise HTTPException(409, f"Camera {req.camera} is stopped — start it first")
    if not cam.connected:
        raise HTTPException(409, f"Camera {req.camera} is not connected")
    if not cam.recognition or state.paused:
        raise HTTPException(409, "Recognition is off for this camera — "
                                 "there is no face to verify")
    if cam.capture is not None:
        raise HTTPException(409, "A capture is already in progress")

    request = CaptureRequest()
    cam.capture = request
    if not request.done.wait(timeout=CAPTURE_TIMEOUT_S):
        cam.capture = None
        raise HTTPException(504, "No live face in view — stand in front of the camera")
    if request.error:
        raise HTTPException(409, request.error)
    return request.payload


@app.get("/api/captures")
def captures(limit: int = 30):
    return state.engine.store.list_captures(min(max(limit, 1), 200))


@app.get("/api/captures/{capture_id}/photo")
def capture_photo(capture_id: int):
    relative = state.engine.store.capture_path(capture_id)
    if relative is None:
        raise HTTPException(404, "No such capture")
    root = Path(CAPTURE_DIR).resolve()
    path = (root / relative).resolve()
    # the path came out of our own DB, but resolve-and-check anyway: a stored
    # row is still untrusted input as far as the filesystem is concerned
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Photo file is missing")
    return FileResponse(path, media_type="image/jpeg")


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
