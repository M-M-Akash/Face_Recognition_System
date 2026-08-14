"""
Multi-camera real-time face recognition.
Cameras, GPU, and every threshold come from config.toml — see src/config.py.
"""
import logging
import time

import cv2

from src.VideoStream import VideoStream
from src.config import config
from src.face_engine import FaceEngine, draw_results
from src.pipeline import recognize_frame
from src.runtime import configure_opencv
from src.smoother import IdentitySmoother

logging.basicConfig(level=logging.INFO)

RELOAD_CHECK_S = config.timing.reload_check_s


def run(camera_urls):
    logging.info(f"Config: {config.source}")
    configure_opencv()
    engine = FaceEngine()
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

                results = recognize_frame(engine, smoother, frame, now)
                draw_results(frame, results)
              
                cv2.imshow(f"Camera {i+1}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cam, _ in cameras:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(config.cameras.urls)
