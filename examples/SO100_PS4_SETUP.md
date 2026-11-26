# SO-100 follower setup (Windows + PS4 + cameras)

This notes the working setup for a calibrated **SO-100 follower** on Windows, driven with a **PS4 DualShock**, with **wrist** and **overhead** OpenCV cameras.

Hardware used in this workspace:

| Piece | Detail |
| --- | --- |
| Arm | SO-100 follower, Feetech STS3215, MotorsBus on **COM5** |
| Calibration id | `my_so100_follower` (file under `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`) |
| Controller | PS4 DualShock (USB; close Steam if it captures the pad) |
| Wrist camera | Innomaker U20CAM, OpenCV index **1** |
| Overhead camera | Innomaker U20CAM, OpenCV index **2** |
| Laptop webcam | OpenCV index **0** (unused) |

Do **not** leave lens caps on the Innomaker cams — they open as black frames.

## 1. Environment

From the repo root (LeRobot + Feetech extras already installed in `.venv`):

```powershell
cd C:\Users\PHILIPPE\IdeaProjects\so-100-setup
.\.venv\Scripts\Activate.ps1
```

Optional: reset both robot cams to the **same** stock DirectShow defaults (stop teleop first):

```powershell
python examples/reset_cameras_default.py
```

Teleop does **not** change brightness / exposure.

## 2. One-time arm bring-up (already done)

Motor IDs (one motor at a time if the arm was assembled first — isolate each cable):

```powershell
lerobot-setup-motors --robot.type=so100_follower --robot.port=COM5
```

Expected IDs: shoulder_pan=1 … gripper=6, baud 1_000_000.

Calibrate (mid pose → Enter → sweep ranges → Enter):

```powershell
lerobot-calibrate --robot.type=so100_follower --robot.port=COM5 --robot.id=my_so100_follower
```

## 3. PS4 teleop

Scripts:

| File | Role |
| --- | --- |
| `examples/so100_ps4_teleop.py` | Main teleop + side-by-side camera preview |
| `examples/so100_kinematics.py` | FK/IK for Cartesian (gripper-in-space) mode |
| `examples/reset_cameras_default.py` | Identical stock defaults for wrist + overhead |
| `examples/phone_to_so100/SO101/so101_new_calib.urdf` | Kinematic model for Cartesian mode (not in git: `*.urdf` is ignored) |

Get the URDF (required for `--mode cartesian`):

```powershell
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git $env:TEMP\SO-ARM100
Copy-Item -Recurse $env:TEMP\SO-ARM100\Simulation\SO101 examples\phone_to_so100\SO101
```

Run:

```powershell
python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower
```

Useful flags:

```powershell
python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower --mode joint
python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower --no-camera
python examples/so100_ps4_teleop.py --port COM5 --id my_so100_follower --wrist-camera 1 --overhead-camera 2
```

Preview uses **pygame** (this env has `opencv-python-headless`, so `cv2.imshow` is unavailable). Left pane = wrist, right pane = overhead.

### Controls

| Input | Cartesian mode (default) | Joint mode |
| --- | --- | --- |
| Left stick | Slide gripper over the table | shoulder pan / lift |
| Right stick up/down | Raise / lower gripper | elbow |
| Right stick left/right | Twist (yaw) | wrist roll |
| D-pad up/down | — | wrist flex |
| L2 / R2 | Gripper close / open | same |
| L1 / R1 | Slower / faster (steps of 0.25, range 0.25–3) | same |
| Square | Switch Cartesian ↔ joint | same |
| Triangle / Ctrl+C / close window | Quit, torque off | same |

Cartesian mode needs the URDF next to the phone examples. Joint mode works without it.

## 4. Camera notes (Windows)

- Prefer DirectShow + `MJPG` at 640×480.
- Brief USB stalls: preview keeps the last good frame (does not abort teleop).
- If indices shuffle after replug, re-detect with a short OpenCV loop or pass `--wrist-camera` / `--overhead-camera`.
- Windows lists devices such as `Innomaker-U20CAM-720P` and `Innomaker-U20CAM-1080p-S1`.

## 5. Sensible next steps

1. Practice pick-and-place with Cartesian teleop + both camera views.
2. Optionally add recording (`lerobot-record`) once demos are clean.
3. Train a small policy (e.g. ACT) and run `lerobot-rollout` on the follower.

A matching **leader arm** is still the easiest physical teleop for long demos; the PS4 path is enough to move and record with care.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| No motors on COM5 | Power (5 V 4 A for SO-100) + USB; Waveshare jumpers on **B** |
| No controller | USB DualShock; quit Steam; confirm pygame sees a joystick |
| Black camera frames | Remove lens caps; free the device (stop other apps) |
| `cv2.imshow` not implemented | Expected with headless OpenCV — use this pygame teleop |
| `latest frame is too old` | Preview path already softens this; unplug/replug cams if persistent |
| Cartesian fails / no URDF | Use `--mode joint` or restore `examples/phone_to_so100/SO101/so101_new_calib.urdf` |
