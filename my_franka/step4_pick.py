"""
Step 4: scripted pick of the apple. No hand-tuned joint angles or recorded waypoints - every target is computed
from where the apple *actually is* right before each phase, so a nudged apple doesn't break the plan.

Run (GUI):       python my_franka/step4_pick.py
Run (headless):  OMNIGIBSON_HEADLESS=1 python my_franka/step4_pick.py

MOTION SPEC (all positions in the robot BASE frame, gripper orientation held fixed at its start value):
    0. SETTLE     open gripper, hold still until the apple stops moving
    1. PRE-GRASP  move to (apple.x, apple.y, apple_top + PREGRASP_CLEARANCE)       - safely above
    2. RE-AIM     re-read the apple pose; correct x/y if it moved                    - tolerate nudges
    3. DESCEND    straight down to (apple.x, apple.y, apple_center + GRASP_Z_OFFSET) - slow, vertical only
    4. CLOSE      close gripper, hold position for CLOSE_STEPS
    5. LIFT       straight up by LIFT_HEIGHT
    6. CHECK      success = robot.is_grasping(apple) and apple is >= SUCCESS_LIFT above the table
    On failure: open, back off upward, re-read the apple, retry from 1 (up to MAX_ATTEMPTS).
"""
import copy

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import IsGraspingState
from omnigibson.macros import gm

from step2_scene_franka_apple import CAMERA_EYE, CAMERA_TARGET, cfg, look_at

# ---------------------------------------------------------------------------------------------
# Tunable numbers (metres / env steps). Env runs at 30 steps per simulated second.
# ---------------------------------------------------------------------------------------------
PREGRASP_CLEARANCE = 0.10  # gap between gripper and the apple's top during the approach
GRASP_Z_OFFSET = 0.0  # where the eef sits relative to the apple centre when closing (eef_link ~ between fingertips)
LIFT_HEIGHT = 0.15
SUCCESS_LIFT = 0.05  # apple must rise at least this much above its resting height
MAX_SPEED = 0.15  # m/s of the commanded target; 0.15 m/s = 5 mm per step
DESCEND_SPEED = 0.05  # slower near the apple so we don't swat it
POS_TOL = 0.01  # how close the real eef must get to a target before we call the move done
MOVE_TIMEOUT = 150  # extra steps allowed to converge after the target stops moving
CLOSE_STEPS = 30
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------------------------
# Controller change vs. step 2: absolute_pose IK. The 6 arm numbers are now "go HERE" (x, y, z in the
# base frame + an axis-angle orientation) instead of "move by this much". That makes a motion spec easy
# to write: we just stream a sequence of target poses. absolute_pose requires the limits to be None.
# ---------------------------------------------------------------------------------------------
cfg = copy.deepcopy(cfg)
# Robots default to action_normalize=True, which forces every controller's input limits back to (-1, 1).
# Absolute poses are real metres/radians, so turn normalization off.
cfg["robots"][0]["action_normalize"] = False
cfg["robots"][0]["controller_config"]["arm_0"] = {
    "name": "InverseKinematicsController",
    "mode": "absolute_pose",
    "command_input_limits": None,
    "command_output_limits": None,
}

OPEN, CLOSE = 1.0, -1.0  # binary gripper command (>= 0 open, < 0 close)


class Picker:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene.robots[0]
        self.apple = env.scene.object_registry("name", "apple")
        # Hold the gripper's starting orientation for the whole pick (it already points roughly down).
        _, self.hold_quat = self.robot.get_relative_eef_pose()
        self.hold_axisangle = T.quat2axisangle(self.hold_quat)
        self.gripper = OPEN

    # ---- frames -----------------------------------------------------------------------------
    def to_base(self, world_pos):
        """World position -> robot base frame (the frame the IK controller uses)."""
        pos, _ = T.relative_pose_transform(world_pos, th.tensor([0, 0, 0, 1.0]), *self.robot.get_position_orientation())
        return pos

    def apple_in_base(self):
        """Apple centre and top (z of its bounding box) in the base frame."""
        center = self.to_base(self.apple.get_position_orientation()[0])
        top_world = self.apple.aabb[1].clone()
        top_world[:2] = self.apple.get_position_orientation()[0][:2]
        return center, self.to_base(top_world)[2]

    def eef_pos(self):
        return self.robot.get_relative_eef_pose()[0]

    # ---- primitives -------------------------------------------------------------------------
    def step(self, target_pos):
        action = th.cat([target_pos, self.hold_axisangle, th.tensor([self.gripper])])
        self.env.step(action)

    def move_to(self, goal, speed=MAX_SPEED, label=""):
        """
        Stream targets along a straight line from the current eef position to @goal at @speed, then keep
        commanding @goal until the real eef is within POS_TOL. Differential IK only works well for small
        steps, which is why we never send the far-away goal directly.
        """
        start = self.eef_pos()
        dist = th.norm(goal - start).item()
        n = max(1, int(dist / (speed / 30.0)))
        for i in range(1, n + 1):
            self.step(start + (goal - start) * (i / n))
        for _ in range(MOVE_TIMEOUT):
            if th.norm(self.eef_pos() - goal).item() < POS_TOL:
                break
            self.step(goal)
        err = th.norm(self.eef_pos() - goal).item()
        print(f"  {label:<10} goal={[round(v, 3) for v in goal.tolist()]}  final err={err * 100:.1f} cm", flush=True)
        return err < POS_TOL * 2

    def hold(self, steps):
        target = self.eef_pos()
        for _ in range(steps):
            self.step(target)

    def wait_for_apple_to_settle(self, max_steps=90):
        last = self.apple.get_position_orientation()[0]
        for _ in range(max_steps):
            self.hold(1)
            now = self.apple.get_position_orientation()[0]
            if th.norm(now - last).item() < 1e-4:
                return
            last = now

    # ---- the motion spec --------------------------------------------------------------------
    def attempt(self):
        # 1. PRE-GRASP
        center, top_z = self.apple_in_base()
        self.move_to(th.tensor([center[0], center[1], top_z + PREGRASP_CLEARANCE]), label="pre-grasp")
        # 2. RE-AIM (the apple may have been bumped since we last looked)
        center, top_z = self.apple_in_base()
        self.move_to(th.tensor([center[0], center[1], top_z + PREGRASP_CLEARANCE]), label="re-aim")
        # 3. DESCEND
        self.move_to(th.tensor([center[0], center[1], center[2] + GRASP_Z_OFFSET]), speed=DESCEND_SPEED, label="descend")
        # 4. CLOSE
        self.gripper = CLOSE
        self.hold(CLOSE_STEPS)
        # 5. LIFT
        self.move_to(self.eef_pos() + th.tensor([0, 0, LIFT_HEIGHT]), label="lift")
        self.hold(30)  # give a weak grasp time to fail
        # 6. CHECK
        return self.is_success()

    def is_success(self):
        grasping = self.robot.is_grasping(candidate_obj=self.apple) == IsGraspingState.TRUE
        lifted = self.apple.get_position_orientation()[0][2].item() - self.rest_z
        print(f"  check      grasping={grasping}  apple lifted {lifted * 100:.1f} cm", flush=True)
        return grasping and lifted > SUCCESS_LIFT

    def run(self):
        print(f"gripper z-axis in world (should be ~[0,0,-1] = pointing down): "
              f"{T.quat2mat(self.robot.eef_links[self.robot.default_arm].get_position_orientation()[1])[:, 2].tolist()}")
        # 0. SETTLE
        self.gripper = OPEN
        self.wait_for_apple_to_settle()
        self.rest_z = self.apple.get_position_orientation()[0][2].item()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"attempt {attempt}", flush=True)
            if self.attempt():
                print("SUCCESS: apple picked up", flush=True)
                return True
            # Recover: let go, back off upward, let the apple settle, try again from its new pose
            self.gripper = OPEN
            self.hold(CLOSE_STEPS)
            self.move_to(self.eef_pos() + th.tensor([0, 0, 0.10]), label="back off")
            self.wait_for_apple_to_settle()
            self.rest_z = min(self.rest_z, self.apple.get_position_orientation()[0][2].item())
        print("FAILED after all attempts", flush=True)
        return False


def main():
    env = og.Environment(configs=cfg)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(CAMERA_EYE), orientation=look_at(CAMERA_EYE, CAMERA_TARGET)
    )
    Picker(env).run()

    if not gm.HEADLESS:
        print("Done. Holding pose - Ctrl+C to exit.", flush=True)
        robot = env.scene.robots[0]
        try:
            while True:
                env.step(th.cat([robot.get_relative_eef_pose()[0], T.quat2axisangle(robot.get_relative_eef_pose()[1]),
                                 th.tensor([CLOSE])]))
        except KeyboardInterrupt:
            pass
    og.shutdown()


if __name__ == "__main__":
    main()
