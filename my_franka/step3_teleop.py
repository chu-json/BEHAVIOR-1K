"""
Step 3: drive the Franka by keyboard in the same scene, and log poses we'll later turn into pick waypoints.

Run (GUI only - teleop needs a window to receive key presses):
    python my_franka/step3_teleop.py

Every press of Z appends the current end-effector pose (and the apple pose) to my_franka/waypoints.json.
"""
import json
import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import KeyboardRobotController

# Reuse the exact same world description + camera helper from step 2 (also applies its gm settings).
from step2_scene_franka_apple import CAMERA_EYE, CAMERA_TARGET, cfg, look_at

WAYPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


def main():
    env = og.Environment(configs=cfg)
    robot = env.scene.robots[0]
    apple = env.scene.object_registry("name", "apple")

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(CAMERA_EYE), orientation=look_at(CAMERA_EYE, CAMERA_TARGET)
    )

    # KeyboardRobotController reads robot.controllers and builds a key -> action-index mapping:
    # IK arm keys write into action[0:6], the binary gripper key (T) writes into action[6].
    teleop = KeyboardRobotController(robot=robot)

    waypoints = []

    def log_pose():
        # Poses relative to the robot BASE: this is the frame the IK controller works in,
        # and it stays valid if we ever move the robot somewhere else in the room.
        eef_pos, eef_quat = robot.get_relative_eef_pose()
        apple_pos, apple_quat = T.relative_pose_transform(
            *apple.get_position_orientation(), *robot.get_position_orientation()
        )
        entry = {
            "eef_pos": [round(v, 4) for v in eef_pos.tolist()],
            "eef_quat": [round(v, 4) for v in eef_quat.tolist()],
            "apple_pos": [round(v, 4) for v in apple_pos.tolist()],
            "grasping": robot.is_grasping().name,
        }
        waypoints.append(entry)
        with open(WAYPOINT_FILE, "w") as f:  # saved on every press, since ESC exits immediately
            json.dump(waypoints, f, indent=2)
        print(f"[waypoint {len(waypoints) - 1}] {entry}", flush=True)

    def print_camera():
        pos, quat = og.sim.viewer_camera.get_position_orientation()
        print(f"[camera] position={pos.tolist()} orientation={quat.tolist()}", flush=True)

    teleop.register_custom_keymapping(lazy.carb.input.KeyboardInput.Z, "log eef + apple pose", log_pose)
    teleop.register_custom_keymapping(lazy.carb.input.KeyboardInput.X, "print viewer camera pose", print_camera)
    teleop.register_custom_keymapping(lazy.carb.input.KeyboardInput.R, "reset scene", lambda: env.reset())
    teleop.print_keyboard_teleop_info()
    print("NOTE: the printout above swaps x/y. Actual: UP/DOWN = +x/-x (forward/back), LEFT/RIGHT = +y/-y.")
    print("Press ESC to quit.\n", flush=True)

    while True:
        env.step(teleop.get_teleop_action())


if __name__ == "__main__":
    main()
