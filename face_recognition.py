"""
Multi-camera real-time face recognition.
Set USE_GPU=1 env var to use CUDA (requires onnxruntime-gpu installed).
"""
import json
import logging
import os
import time

import cv2

from src.VideoStream import VideoStream
from src.face_engine import FaceEngine, draw_results
from src.smoother import IdentitySmoother

logging.basicConfig(level=logging.INFO)

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]

RELOAD_CHECK_S = 2.0  # how often to poll the store for new enrollments


def run(camera_urls):
    engine = FaceEngine(providers=providers)
    cameras = [(i, VideoStream(url).start(), IdentitySmoother())
               for i, url in enumerate(camera_urls)]
    last_reload_check = 0.0

    try:
        while cameras:
            now = time.time()
            if now - last_reload_check >= RELOAD_CHECK_S:
                if engine.maybe_reload():
                    logging.info("Store changed — reloaded face index")
                last_reload_check = now

            for i, cam, smoother in list(cameras):
                frame = cam.read()
                if frame is None:
                    logging.warning(f"Camera {i+1} disconnected")
                    cam.stop()
                    cameras = [c for c in cameras if c[0] != i]
                    cv2.destroyWindow(f"Camera {i+1}")
                    continue

                results = smoother.update(engine.detect_and_recognize(frame))
                draw_results(frame, results)
              
                cv2.imshow(f"Camera {i+1}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cam, _ in cameras:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("camera_urls.json") as f:
        urls = json.load(f)
    run(urls)
