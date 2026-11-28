"""Last-mile press assist (shared autonomy): the policy drives and reaches; a scripted IK press finishes the job.

Motivation (NOTES 2026-11-27): the organisers' released pi0.5 radio checkpoint reliably gets its hand onto the radio
but rarely triggers the toggle button. PressAssist wraps the evaluator's policy.forward. While the policy runs it
watches the closest hand-to-button distance (privileged: true button position from the sim, like RecoveryAssist).
When a hand is within `near_m` of the button and has made no progress for `stall_steps` steps (radio not yet on),
it takes over:
  pre    move the nearer arm so the end-effector point sits `pre_m` in front of the button along the radio's front
         face normal (radio local +x, NOTES "Button geometry follow-up"), gripper closed, base still, torso held
  press  move the end-effector onto the button marker center (+ `press_m` past it)
  hold   hold up to `hold_steps` steps or until the radio turns on
  back   return to the pre point, then hand back to the policy (action queue reset)
At most `max_presses` per episode. IK: damped least squares on the 7 arm joints with steering/urdf_fk.py (base_link
frame, validated to < 1 mm), joint limits from the URDF, position only.

Also usable without a policy (scripted demos): PressAssist(..., always=True) triggers as soon as a hand is within
`near_m`, regardless of stall.

Used via run_rollouts.py --press-assist.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import torch as th

from steering.urdf_fk import ARM, TORSO, URDF, R1ProFK

RADIO_INST = "radio_receiver.n.01_1"


def _limits(urdf=URDF):
    lim = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        L = j.find("limit")
        if L is not None and L.get("lower") is not None:
            lim[j.get("name")] = (float(L.get("lower")), float(L.get("upper")))
    return lim


class ArmIK:
    def __init__(self):
        self.fk = R1ProFK()
        lim = _limits()
        self.lim = {s: np.array([lim.get(n, (-np.pi, np.pi)) for n in ARM[s]]) for s in ARM}

    def eef(self, side, torso, arm):
        q = dict(zip(TORSO, torso))
        q.update(zip(ARM[side], arm))
        return (self.fk.transform(f"{side}_gripper_link", q) @ self.fk.EEF_OFFSET)[:3]

    def solve(self, side, torso, arm0, target, iters=150, damp=0.05, step=0.5):
        q = np.array(arm0, dtype=np.float64)
        lo, hi = self.lim[side][:, 0], self.lim[side][:, 1]
        for _ in range(iters):
            p = self.eef(side, torso, q)
            e = target - p
            if np.linalg.norm(e) < 2e-3:
                break
            J = np.zeros((3, 7))
            for k in range(7):
                dq = q.copy()
                dq[k] += 1e-4
                J[:, k] = (self.eef(side, torso, dq) - p) / 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), e)
            q = np.clip(q + step * dq, lo, hi)
        return q, float(np.linalg.norm(target - self.eef(side, torso, q)))


class PressAssist:
    def __init__(
        self,
        evaluator,
        near_m=0.15,
        stall_steps=150,
        progress_m=0.01,
        min_policy_steps=200,
        pre_m=0.06,
        press_m=0.01,
        move_steps=40,
        hold_steps=60,
        back_steps=25,
        max_presses=3,
        always=False,
    ):
        self.ev = evaluator
        self.p = dict(locals())
        del self.p["self"], self.p["evaluator"]
        self.ik = ArmIK()
        self.reset()

    def reset(self):
        self.state = "policy"
        self.step = 0
        self.count = 0
        self.log = []
        self._segment()

    def _segment(self):
        self.seg_start = self.step
        self.best = float("inf")
        self.best_step = self.step

    @property
    def active(self):
        return int(self.state != "policy")

    def _world(self):
        from omnigibson.object_states import ToggledOn

        acc = self.ev.instance_eval_states[0].env_accessor
        robot = acc.robot
        entity = acc.object_scope[RADIO_INST]
        radio = getattr(entity, "wrapped_obj", entity)
        button = radio.states[ToggledOn].marker.get_position_orientation()[0]
        on = bool(radio.states[ToggledOn].get_value())
        d = {arm: float(th.linalg.norm(robot.get_eef_position(arm) - button)) for arm in robot.arm_names}
        return robot, radio, button, on, d

    def _to_base(self, robot, p_world):
        import omnigibson.utils.transform_utils as T

        return (
            T.relative_pose_transform(p_world, th.tensor([0.0, 0.0, 0.0, 1.0]), *robot.get_position_orientation())[0]
            .cpu()
            .double()
            .numpy()
        )

    def _plan(self, robot, radio, button, d):
        import omnigibson.utils.transform_utils as T

        arm_name = min(d, key=d.get)  # e.g. "left" / "right"
        side = "left" if "left" in arm_name else "right"
        jidx = robot.controller_joint_idx
        q = robot.get_joint_positions().cpu().double().numpy()
        torso = q[jidx["trunk"]]
        arm0 = q[jidx[f"arm_{side}"]]
        # radio front-face normal (local +x) in world, flattened to horizontal
        R = T.quat2mat(radio.get_position_orientation()[1].cpu().float())
        n = R[:, 0].clone()
        n[2] = 0.0
        n = n / (th.linalg.norm(n) + 1e-9)
        b = button.cpu().float()
        to_robot = robot.get_position_orientation()[0].cpu().float() - b
        facing = float((n[:2] @ to_robot[:2]) / (th.linalg.norm(to_robot[:2]) + 1e-9))
        pre_w = b + self.p["pre_m"] * n
        press_w = b - self.p["press_m"] * n
        q_pre, r_pre = self.ik.solve(side, torso, arm0, self._to_base(robot, pre_w))
        q_press, r_press = self.ik.solve(side, torso, q_pre, self._to_base(robot, press_w))
        self.plan = dict(side=side, arm0=arm0, q_pre=q_pre, q_press=q_press, torso=torso)
        self.log.append(
            f"t{self.step}:{side},d={d[arm_name]:.3f},face={facing:.2f},ik_pre={r_pre:.3f},ik_press={r_press:.3f}"
        )

    def _action(self, robot, arm_q, gripper=-1.0):
        a = th.zeros((1, robot.action_dim))
        q = robot.get_joint_positions()
        jidx = robot.controller_joint_idx
        for name, aidx in robot.controller_action_idx.items():
            if name.startswith(("arm", "trunk")):
                a[0, aidx] = q[jidx[name]]
            elif name.startswith("gripper"):
                a[0, aidx] = 1.0
        a[0, robot.controller_action_idx["trunk"]] = th.tensor(self.plan["torso"], dtype=a.dtype)
        s = self.plan["side"]
        a[0, robot.controller_action_idx[f"arm_{s}"]] = th.tensor(arm_q, dtype=a.dtype)
        a[0, robot.controller_action_idx[f"gripper_{s}"]] = gripper
        a[0, robot.controller_action_idx["base"]] = 0.0
        return a

    def wrap(self, forward):
        def assisted_forward(*args, **kwargs):
            self.step += 1
            robot, radio, button, on, d = self._world()
            p = self.p
            if self.state == "policy":
                dmin = min(d.values())
                if dmin < self.best - p["progress_m"]:
                    self.best, self.best_step = dmin, self.step
                stalled = self.step - self.best_step >= p["stall_steps"] and self.step - self.seg_start >= p[
                    "min_policy_steps"
                ]
                if not on and self.count < p["max_presses"] and dmin < p["near_m"] and (p["always"] or stalled):
                    self.count += 1
                    self._plan(robot, radio, button, d)
                    self.state, self.t = "pre", 0
                else:
                    return forward(*args, **kwargs)
            self.t += 1
            P, m = self.plan, p["move_steps"]
            if self.state == "pre":
                a = min(1.0, self.t / m)
                q = (1 - a) * P["arm0"] + a * P["q_pre"]
                if self.t >= m + 5:
                    self.state, self.t = "press", 0
                return self._action(robot, q)
            if self.state == "press":
                a = min(1.0, self.t / m)
                q = (1 - a) * P["q_pre"] + a * P["q_press"]
                if on:
                    self.log.append(f"t{self.step}:ON(press)")
                    self.state, self.t = "back", 0
                elif self.t >= m:
                    self.state, self.t = "hold", 0
                return self._action(robot, q)
            if self.state == "hold":
                if on or self.t >= p["hold_steps"]:
                    self.log.append(f"t{self.step}:{'ON' if on else 'no-toggle'}(hold)")
                    self.state, self.t = "back", 0
                return self._action(robot, P["q_press"])
            if self.state == "back":
                a = min(1.0, self.t / p["back_steps"])
                q = (1 - a) * P["q_press"] + a * P["q_pre"]
                if self.t >= p["back_steps"]:
                    self.state = "policy"
                    self._segment()
                    self.ev.policy.reset()
                return self._action(robot, q)
            return forward(*args, **kwargs)

        return assisted_forward

    def summary(self):
        return {"presses": self.count, "press_log": ";".join(self.log)}


if __name__ == "__main__":  # IK self-test (no sim): reach points around the home pose
    ik = ArmIK()
    torso = np.zeros(4)
    for side in ("left", "right"):
        arm0 = np.zeros(7)
        p0 = ik.eef(side, torso, arm0)
        for dx in (0.05, 0.1):
            q, r = ik.solve(side, torso, arm0, p0 + np.array([dx, 0.0, 0.05]))
            print(side, dx, "residual", round(r, 4))
    _ = math
