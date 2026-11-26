#!/usr/bin/env python
"""Reset wrist (1) and overhead (2) Innomaker cams to the same stock defaults.

Stop any teleop / camera preview first so the devices are free.

    python examples/reset_cameras_default.py
"""

import os
import time

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

# Identical defaults for every camera. No per-view customization.
DEFAULTS = [
    (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75),
    (cv2.CAP_PROP_BRIGHTNESS, 32),
    (cv2.CAP_PROP_CONTRAST, 32),
    (cv2.CAP_PROP_SATURATION, 32),
    (cv2.CAP_PROP_HUE, 0),
    (cv2.CAP_PROP_GAIN, 0),
    (cv2.CAP_PROP_SHARPNESS, 32),
    (cv2.CAP_PROP_GAMMA, 100),
    (cv2.CAP_PROP_AUTO_WB, 1),
]

CAMERAS = [(1, "wrist"), (2, "overhead")]


def main() -> None:
    for index, name in CAMERAS:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"{name} (index {index}): BUSY — stop teleop / other apps using the camera, then retry")
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for prop, value in DEFAULTS:
            cap.set(prop, value)
        time.sleep(1.0)
        for _ in range(20):
            cap.read()
        print(
            f"{name}: brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS)} "
            f"contrast={cap.get(cv2.CAP_PROP_CONTRAST)} "
            f"gain={cap.get(cv2.CAP_PROP_GAIN)} "
            f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}"
        )
        cap.release()
        time.sleep(0.3)
    print("Done. Both cameras use the same defaults. Teleop no longer changes them.")


if __name__ == "__main__":
    main()
