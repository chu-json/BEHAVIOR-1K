"""
Shared pieces for the imitation-learning pipeline (collect -> train -> eval):

    build_cfg()            the world: Rs_int + Franka + apple, absolute-pose IK
    sample_layout(rng)     a random robot pose on the table + a random reachable apple pose
    apply_layout(...)      put robot / apple there and let things settle
    get_obs(...)           the 11-number observation the policy sees
    Expert                 a scripted feedback controller: obs -> action, used to generate demos
    to_env_action(...)     policy action (4 numbers) -> env action (7 numbers)
    check_success(...)     did we really pick the apple?

Coordinate convention: everything the policy sees or outputs is in the robot BASE frame, so a policy
trained with the robot at one spot on the table works wherever the robot is placed.
"""
import copy
import math

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import IsGraspingState

from omnigibson.macros import gm
from step2_scene_franka_apple import TABLE_NAME, TABLE_TOP_Z, cfg as base_cfg

if gm.HEADLESS:
    gm.RENDER_VIEWER_CAMERA = False  # nobody is watching; skip rendering the viewport for speed

# ---------------------------------------------------------------------------------------------
# Randomization ranges (world frame). breakfast_table_skczfi_0 top: x 1.08..1.87, y -0.07..0.90, z 0.76
# ---------------------------------------------------------------------------------------------
TABLE_X, TABLE_Y = (1.08, 1.87), (-0.07, 0.90)
ROBOT_X, ROBOT_Y = (1.15, 1.35), (0.10, 0.70)  # robot base stays near the -x edge, facing +x
ROBOT_YAW = math.radians(25)  # +/- this much rotation about z
APPLE_DIST = (0.40, 0.60)  # apple distance from the base (closer folds the elbow into its joint limit)
APPLE_BEARING = math.radians(35)  # apple within +/- this angle of where the robot faces
TABLE_MARGIN = 0.06  # keep the apple this far from the table edge
JOINT_NOISE = 0.05  # rad of noise on each arm joint at the start, so start poses vary

# ---------------------------------------------------------------------------------------------
# Expert motion parameters
# ---------------------------------------------------------------------------------------------
PREGRASP_CLEARANCE = 0.10  # hover this far above the apple's top before descending
LIFT_Z = 0.19  # lift target height in the BASE frame (base sits on the table top, so this is ~15 cm above the apple)
# The commanded target is placed at most this far AHEAD of the current eef ("carrot on a stick"). The IK /
# joint controllers move the arm in proportion to that gap, so the lead sets the speed. Too small a lead
# (a few mm) and the arm barely moves and drifts; at the goal the target is exactly the goal, so it converges.
MAX_STEP = 0.03
DESCEND_STEP = 0.012  # smaller lead = slower, gentler descent onto the apple
XY_TOL = 0.015  # start descending when the gripper is within this horizontal distance of the apple
XY_RECOVER = 0.035  # ...and only abort a descent (go back up) if it drifts further than this
CLOSE_TOL = 0.006  # close when the gripper is this close to the apple centre
LIFT_TOL = 0.01
LIFT_HOLD_STEPS = 15
SUCCESS_LIFT = 0.05
MAX_EPISODE_STEPS = 400

OPEN, CLOSE = 1.0, -1.0
# obs: eef(3) fingers(2) apple(3) apple-eef(3) prev_gripper_cmd(1) is_grasping(1); act: dpos(3) gripper(1)
OBS_DIM, ACT_DIM = 13, 4


def build_cfg():
    cfg = copy.deepcopy(base_cfg)
    cfg["robots"][0]["action_normalize"] = False  # absolute poses are real metres, not [-1, 1]
    cfg["robots"][0]["controller_config"]["arm_0"] = {
        "name": "InverseKinematicsController",
        "mode": "absolute_pose",
        "command_input_limits": None,
        "command_output_limits": None,
    }
    return cfg


# ---------------------------------------------------------------------------------------------
# Layout randomization
# ---------------------------------------------------------------------------------------------
def sample_layout(rng):
    """Returns dict(robot_pos, robot_yaw, apple_pos) in world frame, with the apple reachable and on the table."""
    while True:
        rx, ry = rng.uniform(*ROBOT_X), rng.uniform(*ROBOT_Y)
        yaw = rng.uniform(-ROBOT_YAW, ROBOT_YAW)
        dist = rng.uniform(*APPLE_DIST)
        bearing = yaw + rng.uniform(-APPLE_BEARING, APPLE_BEARING)
        ax, ay = rx + dist * math.cos(bearing), ry + dist * math.sin(bearing)
        on_table = (TABLE_X[0] + TABLE_MARGIN < ax < TABLE_X[1] - TABLE_MARGIN) and (
            TABLE_Y[0] + TABLE_MARGIN < ay < TABLE_Y[1] - TABLE_MARGIN
        )
        if on_table:
            return dict(
                robot_pos=[rx, ry, TABLE_TOP_Z],
                robot_yaw=yaw,
                apple_pos=[ax, ay, TABLE_TOP_Z + 0.05],
                joint_noise=rng.uniform(-JOINT_NOISE, JOINT_NOISE, size=7).tolist(),
            )


def apply_layout(env, layout, settle_steps=40):
    """Reset the scene, then place the robot and apple according to @layout and let the apple settle."""
    robot, apple = env.scene.robots[0], env.scene.object_registry("name", "apple")
    env.reset()
    quat = T.euler2quat(th.tensor([0.0, 0.0, layout["robot_yaw"]]))
    robot.set_position_orientation(position=th.tensor(layout["robot_pos"]), orientation=quat)
    robot.reset()  # arm joints back to the default pose
    arm_idx = robot.arm_control_idx[robot.default_arm]
    q = robot.get_joint_positions()
    q[arm_idx] += th.tensor(layout["joint_noise"])
    robot.set_joint_positions(q)
    robot.keep_still()
    apple.set_position_orientation(position=th.tensor(layout["apple_pos"]), orientation=th.tensor([0, 0, 0, 1.0]))
    apple.keep_still()
    hold = None
    for _ in range(settle_steps):
        # hold the arm where it is, gripper open, while the apple drops the last few cm
        pos, quat_rel = robot.get_relative_eef_pose()
        hold = hold if hold is not None else (pos, T.quat2axisangle(quat_rel))
        env.step(th.cat([hold[0], hold[1], th.tensor([OPEN])]))


# ---------------------------------------------------------------------------------------------
# Observation / action
# ---------------------------------------------------------------------------------------------
def to_base(robot, world_pos):
    pos, _ = T.relative_pose_transform(world_pos, th.tensor([0, 0, 0, 1.0]), *robot.get_position_orientation())
    return pos


def get_state(env, prev_grip):
    """
    Raw quantities in the base frame (used by both the expert and get_obs).
    @prev_grip is the gripper command sent on the previous step (+1 open / -1 close): the policy's own memory
    of whether it already decided to close.
    """
    robot, apple = env.scene.robots[0], env.scene.object_registry("name", "apple")
    eef = robot.get_relative_eef_pose()[0]
    fingers = robot.get_joint_positions()[robot.gripper_control_idx[robot.default_arm]]
    apple_world = apple.get_position_orientation()[0]
    apple_c = to_base(robot, apple_world)
    top_world = apple.aabb[1].clone()
    top_world[:2] = apple_world[:2]
    apple_top_z = to_base(robot, top_world)[2]
    grasping = float(robot.is_grasping(candidate_obj=apple) == IsGraspingState.TRUE)
    return dict(eef=eef, fingers=fingers, apple=apple_c, apple_top_z=apple_top_z, prev_grip=float(prev_grip),
                grasping=grasping)


def get_obs(state):
    return th.cat([
        state["eef"], state["fingers"], state["apple"], state["apple"] - state["eef"],
        th.tensor([state["prev_grip"], state["grasping"]]),
    ]).float()


# Fixed gripper orientation (base frame, axis-angle) used for every step. Captured from the default arm pose
# the first time it's needed, so expert and policy always command the same orientation.
_HOLD_ORI = None


def hold_orientation(env):
    global _HOLD_ORI
    if _HOLD_ORI is None:
        robot = env.scene.robots[0]
        saved = robot.get_joint_positions()
        robot.reset()
        _HOLD_ORI = T.quat2axisangle(robot.get_relative_eef_pose()[1])
        robot.set_joint_positions(saved)
    return _HOLD_ORI


def to_env_action(env, state, act):
    """act = [dx, dy, dz, gripper] -> 7-dim absolute-pose env action."""
    dpos = th.as_tensor(act[:3], dtype=th.float32).clamp(-MAX_STEP, MAX_STEP)
    grip = OPEN if float(act[3]) >= 0 else CLOSE
    return th.cat([state["eef"] + dpos, hold_orientation(env), th.tensor([grip])])


# ---------------------------------------------------------------------------------------------
# Expert: a MARKOV feedback rule. Every decision depends only on what the policy can also observe (gripper
# and apple positions, last gripper command, is_grasping) - no hidden phase variable or timers. That matters
# for imitation: if the expert acts on information the policy can't see, the policy can only learn the
# average of the expert's possible actions (e.g. "hover" = average of "hold still" and "lift").
#
#   gripper open, not above the apple   -> fly to the hover point above the apple
#   gripper open, above the apple       -> descend onto the apple centre (keep correcting x/y)
#   gripper open, at the apple centre   -> close
#   gripper closed, not yet grasping    -> hold still while the fingers close
#   gripper closed, grasping            -> lift straight up to LIFT_Z, then hold
# ---------------------------------------------------------------------------------------------
class Expert:
    def __init__(self):
        self.hold_count = 0  # only used to decide when the demo episode is over, never to choose actions

    @staticmethod
    def _toward(eef, goal, max_step):
        d = goal - eef
        n = th.norm(d).item()
        return d if n <= max_step else d * (max_step / n), n

    def act(self, s):
        eef, apple = s["eef"], s["apple"]
        if s["prev_grip"] < 0:  # already closed
            if s["grasping"] > 0:
                d, err = self._toward(eef, th.stack([eef[0], eef[1], th.tensor(LIFT_Z)]), MAX_STEP)
                self.hold_count += err < LIFT_TOL
                return th.cat([d, th.tensor([CLOSE])])
            return th.tensor([0.0, 0.0, 0.0, CLOSE])
        hover_z = s["apple_top_z"] + PREGRASP_CLEARANCE
        xy_err = th.norm(apple[:2] - eef[:2]).item()
        # Start descending only when well centred, but once below the hover height keep descending (correcting
        # x/y on the way) unless we're far off. Without that slack, a small IK jump mid-descent sends the arm
        # back up and it can loop forever. Still Markov: it depends only on the current eef height.
        descending = xy_err < XY_TOL or (eef[2] < hover_z - 0.02 and xy_err < XY_RECOVER)
        if not descending:
            d, _ = self._toward(eef, th.stack([apple[0], apple[1], hover_z]), MAX_STEP)
            return th.cat([d, th.tensor([OPEN])])
        d, err = self._toward(eef, apple, DESCEND_STEP)
        if err < CLOSE_TOL:
            return th.tensor([0.0, 0.0, 0.0, CLOSE])
        return th.cat([d, th.tensor([OPEN])])

    @property
    def done(self):
        return self.hold_count >= LIFT_HOLD_STEPS

    @property
    def phase(self):
        return "hold" if self.hold_count > 0 else "running"


def check_success(env, apple_rest_z):
    """Picked = robot reports grasping the apple AND the apple is at least SUCCESS_LIFT above where it rested."""
    robot, apple = env.scene.robots[0], env.scene.object_registry("name", "apple")
    grasping = robot.is_grasping(candidate_obj=apple) == IsGraspingState.TRUE
    lifted = apple.get_position_orientation()[0][2].item() - apple_rest_z
    return grasping and lifted > SUCCESS_LIFT, lifted


def apple_z(env):
    return env.scene.object_registry("name", "apple").get_position_orientation()[0][2].item()

