#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drive a calibrated SO-100 follower from a PS4 DualShock controller.

The stick position is a speed, not a target. Releasing the sticks holds the arm
where it is. Square switches between moving the gripper in space and jogging
each joint. L1 slows every motion down and R1 speeds it up, one step per press.
Triangle or Ctrl+C stops the loop and turns torque off.

Wrist + overhead cameras open a side-by-side preview at startup and while driving.

    python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower
    python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower --mode joint
    python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower --no-camera
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pygame
import pygame._sdl2.controller as gamepad

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import MotorNormMode
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from so100_kinematics import ARM_JOINTS, ArmKinematics, rot_z

DEADZONE = 0.15
LOOP_DT = 0.02
# SDL game-controller axes. These stay on the same physical control when Windows
# presents a DualShock as an Xbox pad, which raw axis numbers do not.
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5
BUTTON_SQUARE = 2
BUTTON_TRIANGLE = 3
BUTTON_L1 = 9
BUTTON_R1 = 10
BUTTON_DPAD_UP = 11
BUTTON_DPAD_DOWN = 12
SPEED_STEP = 0.25
SPEED_MIN = 0.25
SPEED_MAX = 3.0
# Full-stick gripper speed in Cartesian mode, metres per second, and yaw in degrees per second.
EE_SPEED_M_S = 0.06
YAW_SPEED_DEG_S = 60.0
URDF_PATH = Path(__file__).resolve().parent / "phone_to_so100" / "SO101" / "so101_new_calib.urdf"
# Full-stick speeds at --speed 1. Body joints are degrees per second. Gripper is percent per second.
SPEEDS = {
    "shoulder_pan": 35.0,
    "shoulder_lift": 35.0,
    "elbow_flex": 35.0,
    "wrist_flex": 35.0,
    "wrist_roll": 60.0,
    "gripper": 40.0,
}
PREVIEW_WINDOW = "SO-100 cameras"
PANEL_W = 640
PANEL_H = 480


def _stick(pad: gamepad.Controller, axis: int) -> float:
    value = pad.get_axis(axis) / 32767.0
    if abs(value) < DEADZONE:
        return 0.0
    return max(-1.0, min(1.0, value))


def _trigger(pad: gamepad.Controller, axis: int) -> float:
    value = pad.get_axis(axis) / 32767.0
    if value < DEADZONE:
        return 0.0
    return max(0.0, min(1.0, value))


def _limits(robot: SO100Follower, name: str) -> tuple[float, float]:
    motor = robot.bus.motors[name]
    calibration = robot.bus.calibration[name]
    if motor.norm_mode is MotorNormMode.RANGE_0_100:
        return 0.0, 100.0
    midpoint = (calibration.range_min + calibration.range_max) / 2
    resolution = robot.bus.resolution(motor.model) - 1
    low = (calibration.range_min - midpoint) * 360 / resolution
    high = (calibration.range_max - midpoint) * 360 / resolution
    return low, high


def _read_pad(pad: gamepad.Controller) -> dict[str, float]:
    return {
        "left_x": _stick(pad, AXIS_LEFT_X),
        "left_y": -_stick(pad, AXIS_LEFT_Y),
        "right_x": _stick(pad, AXIS_RIGHT_X),
        "right_y": -_stick(pad, AXIS_RIGHT_Y),
        "dpad_y": float(pad.get_button(BUTTON_DPAD_UP)) - float(pad.get_button(BUTTON_DPAD_DOWN)),
        "gripper": _trigger(pad, AXIS_R2) - _trigger(pad, AXIS_L2),
    }


def _apply_joint_mode(command: dict, rates: dict[str, float], bounds: dict, speed: float) -> None:
    for name, rate in rates.items():
        key = f"{name}.pos"
        low, high = bounds[name]
        command[key] = min(high, max(low, command[key] + rate * SPEEDS[name] * speed * LOOP_DT))


def _apply_cartesian_mode(
    command: dict,
    kinematics: ArmKinematics,
    rates: dict[str, float],
    bounds: dict,
    speed: float,
) -> None:
    joints = np.array([command[f"{name}.pos"] for name in ARM_JOINTS], dtype=float)
    target = kinematics.forward(joints)
    target[:3, 3] = target[:3, 3] + np.array(
        [rates["left_y"], -rates["left_x"], rates["right_y"]], dtype=float
    ) * (EE_SPEED_M_S * speed * LOOP_DT)
    target[:3, :3] = rot_z(rates["right_x"] * YAW_SPEED_DEG_S * speed * LOOP_DT) @ target[:3, :3]
    solved = kinematics.solve(joints, target)
    for index, name in enumerate(ARM_JOINTS):
        low, high = bounds[name]
        command[f"{name}.pos"] = float(min(high, max(low, solved[index])))
    low, high = bounds["gripper"]
    command["gripper.pos"] = min(
        high, max(low, command["gripper.pos"] + rates["gripper"] * SPEEDS["gripper"] * speed * LOOP_DT)
    )


def _peek_cameras(robot: SO100Follower, cache: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    """Read camera frames for preview without aborting on a brief USB stall."""
    frames = dict(cache)
    for key, camera in robot.cameras.items():
        try:
            frames[key] = camera.read_latest(max_age_ms=2000)
        except (TimeoutError, RuntimeError):
            # Keep the last good frame so a slow cam does not stop teleop.
            pass
    return frames


def _print_controls(mode: str, speed: float) -> None:
    print(f"Mode: {mode}    speed scale: {speed}")
    if mode == "cartesian":
        print("Left stick   slide the gripper across the table")
        print("Right stick  up/down raises it, left/right twists it")
    else:
        print("Left stick   pan / lift")
        print("Right stick  wrist roll / elbow")
        print("D-pad        wrist flex")
    print("L2 / R2      gripper close / open")
    print("L1 / R1      slower / faster")
    print("Square       switch between gripper and joint control")
    print("Triangle or Ctrl+C quits")


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return np.ascontiguousarray(frame)
    # Nearest-neighbor keeps the preview snappy without pulling in OpenCV GUI code.
    y_idx = (np.linspace(0, frame.shape[0] - 1, height)).astype(np.int32)
    x_idx = (np.linspace(0, frame.shape[1] - 1, width)).astype(np.int32)
    return np.ascontiguousarray(frame[y_idx][:, x_idx])


def _blank_panel(label: str, font: pygame.font.Font) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    surface = pygame.surfarray.make_surface(panel.swapaxes(0, 1))
    text = font.render(label, True, (200, 200, 200))
    surface.blit(text, (16, 16))
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)


def _show_cameras(
    frames: dict[str, np.ndarray | None],
    mode: str,
    speed: float,
    screen: pygame.Surface | None,
    font: pygame.font.Font,
) -> pygame.Surface:
    order = [("wrist", "wrist"), ("front", "overhead")]
    panels = []
    for key, title in order:
        frame = frames.get(key)
        if frame is None:
            panel = _blank_panel(f"{title}: no frame", font)
        else:
            panel = _resize_rgb(frame, PANEL_W, PANEL_H)
        panels.append((title, panel))

    canvas = np.zeros((PANEL_H, PANEL_W * len(panels), 3), dtype=np.uint8)
    for index, (_, panel) in enumerate(panels):
        canvas[:, index * PANEL_W : (index + 1) * PANEL_W] = panel

    width, height = canvas.shape[1], canvas.shape[0]
    if screen is None or screen.get_size() != (width, height):
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(PREVIEW_WINDOW)

    surface = pygame.image.frombuffer(np.ascontiguousarray(canvas).tobytes(), (width, height), "RGB")
    screen.blit(surface, (0, 0))
    for index, (title, _) in enumerate(panels):
        label = font.render(title, True, (40, 220, 40))
        screen.blit(label, (12 + index * PANEL_W, 10))
    status = font.render(f"{mode}  x{speed:.2f}", True, (40, 220, 40))
    screen.blit(status, (12, height - 32))
    pygame.display.flip()
    return screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--id", default="my_so100_follower")
    parser.add_argument("--mode", choices=("cartesian", "joint"), default="cartesian")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Starting speed scale. L1 and R1 change it while the arm is running.",
    )
    parser.add_argument(
        "--wrist-camera",
        type=int,
        default=1,
        help="OpenCV index of the wrist camera (gripper view).",
    )
    parser.add_argument(
        "--overhead-camera",
        type=int,
        default=2,
        help="OpenCV index of the overhead / front camera.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Drive without camera previews.")
    args = parser.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than 0.")

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("consolas", 22)
    gamepad.init()
    if gamepad.get_count() == 0:
        raise SystemExit(
            "No controller found. Plug the PS4 controller in over USB and close Steam if it has captured it."
        )

    pad = gamepad.Controller(0)
    print(f"Controller: {pad.name}")
    kinematics = None
    if args.mode == "cartesian":
        if not URDF_PATH.is_file():
            raise SystemExit(f"Arm model not found at {URDF_PATH}. Start with --mode joint instead.")
        kinematics = ArmKinematics(URDF_PATH)
    _print_controls(args.mode, args.speed)

    cameras = {}
    if not args.no_camera:
        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_camera,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            ),
            "front": OpenCVCameraConfig(
                index_or_path=args.overhead_camera,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
            ),
        }
        print(f"Wrist camera: OpenCV index {args.wrist_camera}")
        print(f"Overhead camera: OpenCV index {args.overhead_camera}")

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cameras))
    robot.connect(calibrate=False)
    preview_screen = None
    try:
        with robot.bus.torque_disabled():
            for motor in robot.bus.motors:
                robot.bus.write("Acceleration", motor, 30)
                robot.bus.write("Maximum_Acceleration", motor, 30)

        joint_obs = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
        command = {f"{name}.pos": value for name, value in joint_obs.items()}
        bounds = {name: _limits(robot, name) for name in robot.bus.motors}
        speed = args.speed
        mode = args.mode
        l1_was_down = False
        r1_was_down = False
        square_was_down = False
        camera_cache: dict[str, np.ndarray | None] = {"wrist": None, "front": None}

        if cameras:
            print("Showing wrist + overhead previews. Close the window or press Triangle to quit.")
            time.sleep(0.5)  # let auto-exposure and the read threads settle
            camera_cache = _peek_cameras(robot, camera_cache)
            preview_screen = _show_cameras(camera_cache, mode, speed, preview_screen, font)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Preview closed, stopping.")
                    return
            if pad.get_button(BUTTON_TRIANGLE):
                print("Triangle pressed, stopping.")
                break

            l1_down = bool(pad.get_button(BUTTON_L1))
            r1_down = bool(pad.get_button(BUTTON_R1))
            square_down = bool(pad.get_button(BUTTON_SQUARE))
            if l1_down and not l1_was_down:
                speed = max(SPEED_MIN, round(speed - SPEED_STEP, 2))
                print(f"Speed scale: {speed}")
            if r1_down and not r1_was_down:
                speed = min(SPEED_MAX, round(speed + SPEED_STEP, 2))
                print(f"Speed scale: {speed}")
            if square_down and not square_was_down:
                if mode == "cartesian":
                    mode = "joint"
                    _print_controls(mode, speed)
                elif not URDF_PATH.is_file():
                    print(f"Cannot switch: arm model not found at {URDF_PATH}")
                else:
                    if kinematics is None:
                        kinematics = ArmKinematics(URDF_PATH)
                    mode = "cartesian"
                    _print_controls(mode, speed)
            l1_was_down = l1_down
            r1_was_down = r1_down
            square_was_down = square_down

            rates = _read_pad(pad)
            if mode == "cartesian":
                _apply_cartesian_mode(command, kinematics, rates, bounds, speed)
            else:
                _apply_joint_mode(
                    command,
                    {
                        "shoulder_pan": rates["left_x"],
                        "shoulder_lift": rates["left_y"],
                        "elbow_flex": rates["right_y"],
                        "wrist_flex": rates["dpad_y"],
                        "wrist_roll": rates["right_x"],
                        "gripper": rates["gripper"],
                    },
                    bounds,
                    speed,
                )

            robot.send_action(command)
            if cameras:
                camera_cache = _peek_cameras(robot, camera_cache)
                preview_screen = _show_cameras(camera_cache, mode, speed, preview_screen, font)
            time.sleep(LOOP_DT)
    finally:
        robot.disconnect()
        pygame.quit()


if __name__ == "__main__":
    main()
