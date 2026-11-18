"""Minimal numpy forward kinematics for the R1Pro from its URDF (no kinematics library needed).

Used by the best-of-N critic in comet_server.py to predict where a candidate action chunk would put each hand.
Frame: the URDF root `base_link` (validated against the robot's reported eef positions in steering/NOTES.md).
"""

import xml.etree.ElementTree as ET

import numpy as np

URDF = "/home/behavior/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
TORSO = [f"torso_joint{i}" for i in range(1, 5)]
ARM = {s: [f"{s}_arm_joint{i}" for i in range(1, 8)] for s in ("left", "right")}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class R1ProFK:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        self.child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            rec = {"type": j.get("type"), "parent": j.find("parent").get("link"), "T": T, "axis": axis}
            self.joints[j.get("name")] = rec
            self.child[j.find("child").get("link")] = j.get("name")

    def chain(self, tip):
        names, link = [], tip
        while link in self.child:
            jn = self.child[link]
            names.append(jn)
            link = self.joints[jn]["parent"]
        return names[::-1]

    def pos(self, tip, q):
        """Position of link `tip` in base_link frame. q: dict joint name -> value (missing joints = 0)."""
        return self.transform(tip, q)[:3, 3]

    def transform(self, tip, q):
        """4x4 pose of link `tip` in base_link frame."""
        T = np.eye(4)
        for jn in self.chain(tip):
            j = self.joints[jn]
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], q.get(jn, 0.0))
                T = T @ R
            elif j["type"] == "prismatic":
                P = np.eye(4)
                P[:3, 3] = j["axis"] * q.get(jn, 0.0)
                T = T @ P
        return T

    # OmniGibson's R1Pro end-effector point = gripper_link pose @ (0, 0, -0.06). Recovered from the robot's reported
    # eef positions (zero spread over 97 states); with it FK matches the robot to < 1 mm.
    EEF_OFFSET = np.array([0.0, 0.0, -0.06, 1.0])

    def hands(self, torso, left_arm, right_arm):
        """End-effector positions (base_link frame) for 4 torso + 7 + 7 arm joint values."""
        q = dict(zip(TORSO, torso))
        q.update(zip(ARM["left"], left_arm))
        q.update(zip(ARM["right"], right_arm))
        return {s: (self.transform(f"{s}_gripper_link", q) @ self.EEF_OFFSET)[:3] for s in ("left", "right")}
