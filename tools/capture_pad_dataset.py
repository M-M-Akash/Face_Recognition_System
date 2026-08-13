"""
Record a labelled presentation-attack dataset from a live camera.

Anti-spoofing models are extremely sensitive to capture conditions, so the
only way to know whether one works *here* is to measure it on frames from the
real camera. This writes the dataset tools/eval_antispoof.py consumes.

Run it once per class, varying pose/distance/lighting within each run:

    python -m tools.capture_pad_dataset --label live    --count 100
    python -m tools.capture_pad_dataset --label print   --count 100
    python -m tools.capture_pad_dataset --label replay  --count 100

"live" is the bona-fide class; every other label is treated as a separate
attack species (ISO/IEC 30107-3 reports APCER per species, worst case wins).

Frames are stored as PNG, not JPEG, on purpose: re-encoding a face crop at
JPEG quality 85 was measured to move this model's scores by an order of
magnitude, so a lossy dataset would benchmark the encoder rather than the
model. Detection runs through the same FaceEngine the service uses, so the
stored bbox is exactly what production would have fed the spoof model.

Fully non-interactive (a countdown, then N captures) so it works over SSH and
inside containers. Ctrl-C is safe — the manifest is appended per sample.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.face_engine import FaceEngine  # noqa: E402

MANIFEST = "manifest.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True,
                   help="class name: 'live' for real faces, otherwise the attack "
                        "species (print, replay, mask, ...)")
    p.add_argument("--count", type=int, default=100, help="samples to capture")
    p.add_argument("--camera", default="0", help="device index or RTSP URL")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="Dataset/pad", help="dataset root")
    p.add_argument("--interval", type=float, default=0.2,
                   help="seconds between accepted samples, to avoid near-duplicates")
    p.add_argument("--countdown", type=int, default=5, help="seconds before capture starts")
    p.add_argument("--min-det", type=float, default=0.6, help="minimum detector confidence")
    p.add_argument("--no-sheet", action="store_true",
                   help="skip the contact sheet written at the end of the run")
    return p.parse_args()


def contact_sheet(root, label, files, cols=8, cw=240, ch=135):
    """Thumbnail grid of everything just captured, so mislabelled frames are
    obvious at a glance. Worth the two seconds: an earlier replay run silently
    recorded the operator's own live face — the detector locks onto whichever
    face is largest, and a bare frame count can't tell you that happened."""
    import numpy as np
    if not files:
        return None
    rows = (len(files) + cols - 1) // cols
    sheet = np.full((rows * (ch + 18), cols * cw, 3), 30, np.uint8)
    for i, f in enumerate(files):
        img = cv2.imread(str(root / label / f))
        if img is None:
            continue
        r, c = divmod(i, cols)
        y, x = r * (ch + 18), c * cw
        sheet[y + 18:y + 18 + ch, x:x + cw] = cv2.resize(img, (cw, ch))
        cv2.putText(sheet, f"#{i+1}", (x + 4, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    path = root / f"{label}_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def open_camera(source, width, height):
    cam = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        sys.exit(f"Could not open camera: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):          # let auto-exposure/white-balance settle
        cap.read()
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (width, height):
        print(f"note: camera gave {actual[0]}x{actual[1]}, not {width}x{height}")
    return cap, actual


def main():
    args = parse_args()
    root = Path(args.out)
    (root / args.label).mkdir(parents=True, exist_ok=True)

    engine = FaceEngine()
    cap, (fw, fh) = open_camera(args.camera, args.width, args.height)

    print(f"\nCapturing {args.count} '{args.label}' samples at {fw}x{fh}.")
    if args.label == "live":
        print("Vary distance, angle and lighting. Do NOT hold up a photo.")
    else:
        print(f"Present the '{args.label}' attack to the camera. Vary angle, "
              "distance and glare.")
        print("KEEP YOUR OWN FACE OUT OF FRAME. Only the attack instrument may be\n"
              "visible — detection picks the largest face, so if you lean into shot\n"
              "it records YOUR face under this label and the evaluation counts a\n"
              "correct 'live' call as an attack that got through.")
    for s in range(args.countdown, 0, -1):
        print(f"  starting in {s}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")

    manifest = open(root / MANIFEST, "a")
    written = []
    saved, attempts, last = 0, 0, 0.0
    max_attempts = args.count * 40
    try:
        while saved < args.count and attempts < max_attempts:
            attempts += 1
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() - last < args.interval:
                continue

            faces = engine.detect(frame)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if float(face.det_score) < args.min_det:
                continue

            name = f"{args.label}_{int(time.time()*1000)}.png"
            cv2.imwrite(str(root / args.label / name), frame)
            manifest.write(json.dumps({
                "file": f"{args.label}/{name}",
                "label": args.label,
                "bbox": [int(v) for v in face.bbox],
                "det_score": round(float(face.det_score), 4),
                "frame_w": frame.shape[1],
                "frame_h": frame.shape[0],
                "camera": str(args.camera),
                "ts": time.time(),
            }) + "\n")
            manifest.flush()
            written.append(name)
            saved += 1
            last = time.time()
            print(f"  captured {saved}/{args.count}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        manifest.close()
        cap.release()

    print(f"\nSaved {saved} '{args.label}' samples to {root/args.label}")
    if saved < args.count:
        print(f"(only {saved} of {args.count} — no face was detected in most frames; "
              "check framing and lighting)")
    print(f"Manifest: {root/MANIFEST}")
    if not args.no_sheet:
        sheet = contact_sheet(root, args.label, written)
        if sheet:
            print(f"\nContact sheet: {sheet}")
            print("OPEN IT NOW and check every thumbnail really shows "
                  f"'{args.label}'. Delete any that don't, and drop their lines "
                  "from the manifest — a mislabelled frame corrupts the metrics "
                  "silently.")
    print("\nWhen every class is captured, evaluate with:")
    print(f"  python -m tools.eval_antispoof --dataset {root}")


if __name__ == "__main__":
    main()
