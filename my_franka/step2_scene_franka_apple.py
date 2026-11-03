"""
Step 2-4: build the scene (Rs_int + Franka + apple on the table) and LOOK at it. No motion yet.

Run (GUI):       python my_franka/step2_scene_franka_apple.py
Run (headless):  OMNIGIBSON_HEADLESS=1 python my_franka/step2_scene_franka_apple.py --steps 120
"""
import argparse

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm

# ---------------------------------------------------------------------------------------------
# 1. Global settings (`gm`). These must be set BEFORE the Environment is created.
# ---------------------------------------------------------------------------------------------
gm.USE_GPU_DYNAMICS = False  # plain CPU physics; GPU dynamics is only needed for particles/fluids
gm.ENABLE_OBJECT_STATES = True  # lets us ask "is the apple OnTop of the table?" later

# ---------------------------------------------------------------------------------------------
# 2. Where things go. Numbers come from step1_inspect_scene.py:
#    breakfast_table_skczfi_0 has its top surface at z ~= 0.76 and spans x 1.08..1.87, y -0.07..0.90
# ---------------------------------------------------------------------------------------------
TABLE_NAME = "breakfast_table_skczfi_0"
TABLE_TOP_Z = 0.76
FRANKA_POS = [1.25, 0.41, TABLE_TOP_Z]  # the Franka's base: on the table, on the near (-x) side
APPLE_POS = [1.65, 0.41, TABLE_TOP_Z + 0.05]  # 40 cm in front (+x) of the base, dropped from 5 cm up

cfg = {
    # --- env: how often we act vs. how often physics runs -------------------------------------
    "env": {
        "action_frequency": 30,  # env.step() = 1/30 s of simulated time
        "physics_frequency": 120,  # 4 physics sub-steps per env.step()
    },
    # --- scene: loads the Rs_int house (walls, furniture, ...) --------------------------------
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "not_load_object_categories": ["ceilings"],  # hide ceilings so the GUI camera can see in
    },
    # --- robots: a list; each entry is the kwargs for the robot class -------------------------
    "robots": [
        {
            "type": "Franka",
            "name": "franka",
            "position": FRANKA_POS,
            "orientation": [0, 0, 0, 1],  # quaternion (x, y, z, w) = no rotation, arm faces +x
            "obs_modalities": [],  # no camera observations needed; saves compute
            # "assisted" = when the gripper closes on an object, the sim *attaches* it (forgiving).
            # Switch to "physical" later for real friction-based grasping.
            "grasping_mode": "assisted",
            # Franka's defaults (from franka.yaml) are IK for the arm and a multi-finger controller
            # for the gripper. We spell the arm one out so the action meaning is explicit:
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    # pose_delta_ori: action = [dx, dy, dz, d_rot_x, d_rot_y, d_rot_z], a *delta*
                    # of the end-effector pose relative to where it is now.
                    "mode": "pose_delta_ori",
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "mode": "binary",  # action = 1 dim: open (>0) / close (<0)
                },
            },
        }
    ],
    # --- objects: extra objects added on top of the scene --------------------------------------
    "objects": [
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",  # a specific apple model from the BEHAVIOR asset dataset
            "position": APPLE_POS,
        }
    ],
    # --- task: DummyTask = no goal, no reward. We're driving the robot ourselves ---------------
    "task": {"type": "DummyTask"},
}


# Viewer camera: placed behind and to the side of the robot, aimed at the apple.
CAMERA_EYE = [0.35, -0.2, 1.45]
CAMERA_TARGET = [1.55, 0.41, 0.80]


def look_at(eye, target):
    """
    Quaternion (x, y, z, w) that points a camera at @eye toward @target, keeping the world +z axis "up".

    Omniverse cameras look down their own -z axis, with +x to the right and +y up in the image. So we build
    the camera's three axes in world coordinates and stack them into a rotation matrix.
    """
    eye, target = th.tensor(eye, dtype=th.float32), th.tensor(target, dtype=th.float32)
    forward = target - eye
    z_axis = -forward / th.norm(forward)  # camera looks along -z, so +z points backward
    x_axis = th.linalg.cross(th.tensor([0.0, 0.0, 1.0]), z_axis)  # "right" = world-up x backward
    x_axis = x_axis / th.norm(x_axis)
    y_axis = th.linalg.cross(z_axis, x_axis)  # "up" in the image
    return T.mat2quat(th.stack([x_axis, y_axis, z_axis], dim=1))


def main(steps):
    # Building the Environment launches Isaac Sim, loads everything above, and steps once.
    env = og.Environment(configs=cfg)
    robot = env.scene.robots[0]
    apple = env.scene.object_registry("name", "apple")
    table = env.scene.object_registry("name", TABLE_NAME)

    # Point the GUI camera at the robot + apple
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(CAMERA_EYE), orientation=look_at(CAMERA_EYE, CAMERA_TARGET)
    )

    # ---- Inspect what we built -------------------------------------------------------------
    print("\n=========== INSPECTION ===========", flush=True)
    print("robot name / model :", robot.name, "/", robot.model, flush=True)
    print("arm names          :", robot.arm_names, flush=True)
    print("action space       :", robot.action_space, flush=True)
    print("action order       :", robot.controller_order, "(action vector = these, concatenated)", flush=True)
    print("eef position       :", robot.get_eef_position().tolist(), flush=True)

    # ---- Let physics run so the apple falls the last 5 cm and settles -----------------------
    zero_action = th.zeros(robot.action_dim)  # all-zero delta = "hold still"
    for i in range(steps):
        env.step(zero_action)
        if i % 30 == 0 or i == steps - 1:
            print(f"step {i:4d}  apple z={apple.get_position_orientation()[0][2].item():.3f}", flush=True)

    print("apple on table?    :", apple.states[og.object_states.OnTop].get_value(table), flush=True)
    print("robot grasping?    :", robot.is_grasping().name, flush=True)
    print("eef position       :", robot.get_eef_position().tolist(), flush=True)

    if not gm.HEADLESS:
        print("\nGUI mode: close the window or press Ctrl+C to exit.", flush=True)
        try:
            while True:
                env.step(zero_action)
        except KeyboardInterrupt:
            pass

    og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=90, help="env steps to run before inspecting")
    main(parser.parse_args().steps)
