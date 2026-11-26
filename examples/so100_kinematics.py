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

"""Forward and inverse kinematics for the SO-100 arm.

Joint angles are degrees in the same frame the calibrated follower reports.
The chain comes from the SO-101 URDF, which is the model LeRobot uses for this arm.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EE_LINK = "gripper_frame_link"


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: RotZ(yaw) @ RotY(pitch) @ RotX(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def _rotvec(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    scale = angle / (2.0 * np.sin(angle))
    return scale * skew


def rot_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


class ArmKinematics:
    """Serial chain from the base to the gripper frame."""

    def __init__(self, urdf_path: str | Path):
        self.chain = self._load_chain(Path(urdf_path))

    @staticmethod
    def _load_chain(urdf_path: Path) -> list[dict]:
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, dict] = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
            rpy = (origin.get("rpy") if origin is not None else None) or "0 0 0"
            axis_el = joint.find("axis")
            axis = (axis_el.get("xyz") if axis_el is not None else None) or "0 0 1"
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            by_child[child.get("link")] = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent.get("link"),
                "xyz": np.array([float(value) for value in xyz.split()]),
                "rpy": np.array([float(value) for value in rpy.split()]),
                "axis": np.array([float(value) for value in axis.split()]),
            }

        chain: list[dict] = []
        link = EE_LINK
        while link != "base_link":
            joint = by_child[link]
            chain.append(joint)
            link = joint["parent"]
        chain.reverse()
        names = [joint["name"] for joint in chain if joint["type"] == "revolute"]
        if tuple(names) != ARM_JOINTS:
            raise ValueError(f"URDF arm joints {names} do not match {ARM_JOINTS}.")
        return chain

    def forward(self, joints_deg: np.ndarray) -> np.ndarray:
        """4x4 pose of the gripper frame for arm joints in degrees."""
        angles = {name: np.deg2rad(float(joints_deg[index])) for index, name in enumerate(ARM_JOINTS)}
        pose = np.eye(4)
        for joint in self.chain:
            angle = angles.get(joint["name"], 0.0) if joint["type"] == "revolute" else 0.0
            rotation = _rpy_matrix(*joint["rpy"]) @ _rot_axis(joint["axis"], angle)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = joint["xyz"]
            pose = pose @ transform
        return pose

    def _jacobian(self, joints_deg: np.ndarray) -> np.ndarray:
        current = self.forward(joints_deg)
        position = current[:3, 3]
        rotation = current[:3, :3]
        jacobian = np.zeros((6, len(ARM_JOINTS)))
        step = 0.1
        for index in range(len(ARM_JOINTS)):
            nudged = joints_deg.copy()
            nudged[index] += step
            moved = self.forward(nudged)
            jacobian[:3, index] = (moved[:3, 3] - position) / step
            jacobian[3:, index] = _rotvec(moved[:3, :3] @ rotation.T) / step
        return jacobian

    def solve(
        self,
        joints_deg: np.ndarray,
        target: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.15,
        iterations: int = 8,
        damping: float = 0.001,
        max_step_deg: float = 1.5,
    ) -> np.ndarray:
        """Damped least-squares step toward a gripper pose. Joints stay in degrees."""
        joints = np.array(joints_deg, dtype=float)
        weights = np.diag([position_weight] * 3 + [orientation_weight] * 3)
        for _ in range(iterations):
            current = self.forward(joints)
            position_error = target[:3, 3] - current[:3, 3]
            orientation_error = _rotvec(target[:3, :3] @ current[:3, :3].T)
            if np.linalg.norm(position_error) < 2e-4 and np.linalg.norm(orientation_error) < 1e-3:
                break
            jacobian = weights @ self._jacobian(joints)
            error = weights @ np.concatenate([position_error, orientation_error])
            system = jacobian @ jacobian.T + (damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            delta = np.clip(delta, -max_step_deg, max_step_deg)
            joints = joints + delta
        return joints
