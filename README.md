# Face Recognition Attendance

Real-time, multi-camera face recognition with passive liveness detection, built for
attendance. It runs on CPU — no GPU required — and everything is configured from a
single annotated file.

![Live face recognised in green, a printed photo of the same person rejected in red](assets/liveness-demo.png)

Both faces in that frame are the same person. The live one is recognised — `Akash 0.71`
in green. The printed photo held beside it is rejected — `SPOOF? 0.02` in red, with the
name deliberately withheld. The `PHOTO` button greys out while a spoof is on screen, so
an attendance photo cannot be taken of a photograph.

---

## What it does

- **Recognises faces from several cameras at once**, drawing a labelled box on a live
  MJPEG stream per camera.
- **Rejects presentation attacks.** A passive anti-spoofing model scores every face; a
  printed photo or a phone screen turns the box red and blocks attendance capture.
- **Takes attendance photos on demand.** A `PHOTO` button per camera saves a full frame
  to disk, refusing if the face is a suspected spoof.
- **Enrolls people from the browser**, with a Face ID-style ring that fills as the
  person turns their head through centre, left and right.
- **Logs sightings** to a database as an audit trail, each with its liveness score.
- **Reports where its time goes** — per-camera capture / processing / streaming rates,
  per-stage milliseconds, dropped frames and CPU — so a slowdown can be diagnosed
  rather than guessed at.

### How it works

Detection uses **SCRFD**, recognition uses a **MobileFaceNet**-style embedder (the
`buffalo_sc` bundle from [insightface](https://github.com/deepinsight/insightface),
running on ONNX Runtime). Liveness uses the Apache-2.0 **MiniFASNet** ensemble from
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

A face is identified by cosine similarity against enrolled embeddings, then smoothed
over a short voting window so a single bad frame cannot flash the wrong name. Both
recognition and liveness are scheduled **per face**: a face is re-identified every frame
until its identity settles, then only re-checked about once a second. That is what makes
several cameras viable on a modest CPU — on a 2-core i3 it cut per-frame cost from
32 ms to 20 ms and embeddings from 1.00 to 0.12 per frame.

---

## Running it

Requires **Python 3.11+**. Model weights download automatically on first run.

### Docker (recommended)

```bash
docker compose up web              # dashboard at http://localhost:8000
docker compose run --rm enroll     # terminal enrollment (needs stdin)
docker compose up recognize        # OpenCV windows instead of the web UI
```

`config.toml` and `Dataset/` are bind-mounted, so edits take effect on the next start
with no rebuild, and enrollments survive container restarts.

### Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

> Run with a **single** uvicorn worker. One background thread owns all camera reads and
> inference, and the state is in-process.

### The three entry points

| Command | What it is |
|---|---|
| `uvicorn app:app` | The web service — dashboard, streams, JSON API. This is the one you want. |
| `python enroll_new_faces.py` | Guided enrollment from the terminal. Works headless. |
| `python face_recognition.py` | Recognition in OpenCV windows, no web server. Press `q` to quit. |

### First run

1. Start the web service and open the dashboard.
2. Type a name, pick a camera, press **Start guided enrollment**.
3. Follow the prompts — look straight ahead, then turn slowly left, then right. The ring
   fills as each pose is captured.
4. Once enrolled, the name appears on the live stream within about a second of the person
   entering frame.
5. Press **PHOTO** to save an attendance photo. It refuses if it sees a spoof.

---

## Configuration

Everything lives in [`config.toml`](config.toml). Nothing has to be set — every key shows
its default, so deleting a line changes nothing. Point at a different file with
`FR_CONFIG=/path/to/other.toml`.

Settings resolve in three layers: **built-in defaults → `config.toml` → environment
variables**. Environment variables win so a container can override a read-only mount;
only `USE_GPU`, `ORT_THREADS`, `SPOOF_THRESH`, `SPOOF_MODEL_DIR`, `CAPTURE_DIR` and
`DB_PATH` work that way. A typo'd key is reported at startup rather than silently ignored.

### `[runtime]`

| Key | Default | What it does |
|---|---|---|
| `gpu` | `false` | Use CUDA. Needs `onnxruntime-gpu` instead of `onnxruntime`. |
| `threads` | `0` | Threads per model, for ONNX Runtime **and** OpenCV. `0` = physical cores. Not the thread count your system monitor shows — raising it above physical cores costs CPU without adding throughput, and does **not** affect camera frame rate. |
| `spin_wait` | `false` | Let ONNX threads spin while idle. Leave off: this workload is camera-paced, so spinning just burns cores between frames. |

### `[cameras]`

| Key | Default | What it does |
|---|---|---|
| `urls` | `[0]` | Device indices or RTSP/HTTP URLs, mixed freely: `[0, "rtsp://user:pass@host:554/stream1"]`. |
| `enroll_camera` | first of `urls` | Which camera the terminal enrollment tool opens. |

### `[paths]`

| Key | Default | What it does |
|---|---|---|
| `database` | `Dataset/faces.db` | People, embeddings, sightings, capture index. |
| `captures` | `Dataset/captures` | Attendance photos, as `YYYY-MM-DD/name_time.jpg`. |
| `spoof_models` | `models` | Anti-spoofing weights, downloaded on first use. |

### `[detection]`

| Key | Default | What it does |
|---|---|---|
| `det_size` | `640` | Detector input size. Cost scales with area — 640 ≈ 12.7 ms, 480 ≈ 6.8 ms, 320 ≈ 3.0 ms on a 2-core CPU. Lower misses smaller, more distant faces, but does **not** reduce recognition accuracy for faces still detected. |
| `rec_thresh` | `0.45` | Cosine similarity below which a face is `unknown`. Raise to reduce false matches, lower if enrolled people are missed. |

### `[antispoof]`

| Key | Default | What it does |
|---|---|---|
| `threshold` | `0.48` | Minimum median P(live). Below it the box turns red, capture is refused, and the sighting is flagged. **Hardware- and camera-specific** — see below. |
| `votes` | `3` | Liveness scores a face needs before its verdict is trusted. |
| `window` | `5` | Rolling scores kept per face; the median is the verdict. |

### `[smoothing]`

| Key | Default | What it does |
|---|---|---|
| `window` / `min_votes` | `10` / `6` | A name is shown once it wins `min_votes` of the last `window` recognitions. Raising `min_votes` steadies a flickering *name*; it does nothing for a jittering *box*. |
| `iou_thresh` | `0.3` | Minimum box overlap to treat a detection as the same face. |
| `max_misses` | `5` | Frames a face may vanish for before its track is dropped. |
| `reverify_s` | `1.0` | How often a settled face is re-checked. The main CPU lever. |
| `bbox_alpha` | `0.4` | Box smoothing, 0–1. Lower is steadier but lags a moving face; `1.0` disables it. |
| `snap_iou` | `0.5` | Below this overlap the face jumped — snap the box instead of easing. |

### `[events]`, `[enrollment]`, `[stream]`, `[timing]`

Sighting cooldown and retention; enrollment sample counts and quality gates (detection
confidence, face size, blur, near-duplicate rejection); MJPEG quality and the capture
timeout; loop and reconnection behaviour. Each key is commented in the file.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /stream/{id}` | Annotated MJPEG stream |
| `GET /api/cameras` · `PATCH /api/cameras/{id}` | Status; toggle `enabled` / `recognition` |
| `POST /api/recognition` | Global pause |
| `POST /api/capture` · `GET /api/captures` · `GET /api/captures/{id}/photo` | Attendance photos |
| `POST /api/enroll/start` · `/cancel` · `GET /api/enroll/status` | Guided enrollment |
| `GET /api/people` · `DELETE /api/people/{name}` | Enrolled people |
| `GET /api/events` · `GET /api/events/{id}/snapshot` | Sighting log |
| `GET /api/diagnostics` | Per-camera rates, stage timings, dropped frames, CPU |

---

## Tools

```bash
# Anti-spoofing: capture a labelled dataset, then measure the model.
python -m tools.capture_pad_dataset --label live   --count 100
python -m tools.capture_pad_dataset --label replay --count 100
python -m tools.eval_antispoof --dataset Dataset/pad

# Performance. Cases are interleaved and medians reported, because timings on a
# small-core machine drift enough between runs to make A-then-B comparisons lie.
python -m tools.bench_pipeline                                    # per-model latency
python -m tools.bench_pipeline --pipeline --camera 0 --cameras 4  # end-to-end
```

---

## Things worth knowing

**Re-derive the spoof threshold on your hardware.** `0.48` was measured on one camera
under one set of lighting conditions (ACER 2.62%, APCER 3.24%, BPCER 2.00%). It is not a
universal constant. Use `tools/eval_antispoof.py` — an earlier model that looked fine by
eye scored ACER 38% on the same data, which only measurement caught.

**There is no authentication on any endpoint.** Anyone who can reach the port can view
streams, enroll people, delete them, and retrieve stored face images by walking integer
ids. `Dataset/` is gitignored as biometric data, but that protects the repository, not
the network. Do not expose this beyond a trusted network without putting auth in front
of it.

**Attendance photos are never pruned**, deliberately — sightings are capped at 500 rows
and would otherwise evict them. Disk grows at roughly 45 KB per photo.

**If frame rate looks low, check the light before the code.** Webcams lengthen exposure
in dim conditions and halve their own frame rate; this system measured 16.7 fps in the
evening and 30.5 fps in daylight from the same camera, with no code change. Open the `i`
panel on a camera card — if `capture` is low, the pipeline is not the problem.

**Biometric data carries legal obligations** (GDPR Article 9, BIPA and similar). Consent,
retention and deletion are your responsibility; `DELETE /api/people/{name}` removes a
person and all their embeddings.

## Licence and credits

Face detection and recognition models from
[insightface](https://github.com/deepinsight/insightface). Anti-spoofing weights are
Apache-2.0 from [minivision-ai](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
using the ONNX exports published by
[yakhyo](https://github.com/yakhyo/face-anti-spoofing). Weights are downloaded at runtime
and verified against pinned SHA-256 digests rather than vendored.
