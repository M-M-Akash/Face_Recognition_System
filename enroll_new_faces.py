"""
Guided face enrollment from the terminal: walks the person through
center / left / right head poses with quality-checked samples.

The enrollment state machine lives in src/enrollment.py and is shared with
the web service (app.py) — this script is just the terminal/preview frontend.
Runs headless with terminal progress; shows a mirrored preview window when a
display is available. Press 'q' in the preview to abort without saving.
"""
import os

import cv2

from src.antispoof import load_antispoof
from src.enrollment import SAMPLES_PER_POSE, GuidedEnrollment
from src.face_engine import FaceEngine

_cam = os.environ.get("CAMERA_URL", "0")
CAMERA_URL = int(_cam) if _cam.isdigit() else _cam
# same knob app.py reads, so both enrollment paths gate samples identically
SPOOF_THRESH = float(os.environ.get("SPOOF_THRESH", "0.5"))


def has_display():
    """Check if a GUI display is available."""
    try:
        if not os.environ.get("DISPLAY"):
            return False
        cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_test")
        return True
    except cv2.error:
        return False


def main():
    person_name = input("Enter the person's name: ").strip()
    if not person_name:
        print("Name required.")
        return

    engine = FaceEngine()
    antispoof = load_antispoof()
    if not antispoof.available:
        print("Note: anti-spoofing model missing — enrolling without liveness checks.")
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open camera: {CAMERA_URL}")
        print("Set CAMERA_URL env var or edit the script.")
        return

    gui = has_display()
    if gui:
        print("Display detected — mirrored preview window. Press 'q' to abort.")
    else:
        print("No display — running headless. Follow the prompts below.")

    session = GuidedEnrollment(person_name, engine, antispoof=antispoof,
                               min_liveness=SPOOF_THRESH)
    warned = False

    while session.state == "running":
        ret, frame = cap.read()
        if not ret:
            continue

        info = session.step(frame, engine.detect(frame))

        if info["warning"] and not warned:
            print(f"\nWARNING: {info['warning']}. Continuing anyway.")
            warned = True

        status = "  ".join(f"{p} {n}/{SAMPLES_PER_POSE}" for p, n in info["counts"].items())
        line = f"{status}  |  {info['instruction']}" + (f"  ({info['problem']})" if info["problem"] else "")
        print(f"\r{line:<100}", end="", flush=True)

        if gui:
            preview = cv2.flip(frame, 1)  # mirror so "turn LEFT" feels natural
            if info["bbox"] is not None:
                w = frame.shape[1]
                x1, y1, x2, y2 = info["bbox"]
                color = (0, 255, 0) if info["accepted"] else (0, 165, 255)
                cv2.rectangle(preview, (w - x2, y1), (w - x1, y2), color, 2)
            cv2.putText(preview, info["instruction"], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(preview, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if info["problem"]:
                cv2.putText(preview, info["problem"], (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("Enroll", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                session.cancel()

    cap.release()
    if gui:
        cv2.destroyAllWindows()
    print()  # newline after \r progress

    _, message = session.finish()
    print(message)


if __name__ == "__main__":
    main()
