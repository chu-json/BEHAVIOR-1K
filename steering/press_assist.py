"""Last-mile press assist (shared autonomy): the policy drives and reaches; a scripted IK press finishes the job.

Motivation (NOTES 2026-11-27): the organisers' released pi0.5 radio checkpoint gets its hand onto the radio but
rarely triggers the toggle button (failing rollouts stall 12-22 cm from it; both successes got within 5 cm).
PressAssist wraps the evaluator's policy.forward. While the policy runs it watches the closest hand-to-button
distance (privileged: true button position from the sim, like RecoveryAssist). When a hand is within `near_m` of
the button and has made no progress for `stall_steps` steps (radio not yet on), it takes over:
  base   holonomic base move so the robot squarely faces the radio's front face (radio local +x, NOTES "Button
         geometry follow-up") with the button `reach_x` ahead and in line with the pressing arm; arms/torso held
  pre    move that arm so the end-effector point sits `pre_m` in front of the button along the face normal,
         gripper closed
  press  move the end-effector onto the button marker center (+ `press_m` past it)
  hold   hold up to `hold_steps` steps or until the radio turns on
  back   return to the pre point, then hand back to the policy (action queue reset)
At most `max_presses` per episode. IK: damped least squares on the 7 arm joints with steering/urdf_fk.py (base_link
frame, validated to < 1 mm), joint limits from the URDF, position only, multi-start. At the policy's typical torso
pose the arm reaches about 0.6-0.7 m forward at table height.

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


def _yaw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ArmIK:
    def __init__(self):
        self.fk = R1ProFK()
        lim = _limits()
        self.lim = {s: np.array([lim.get(n, (-np.pi, np.pi)) for n in ARM[s]]) for s in ARM}

    def eef(self, side, torso, arm):
        q = dict(zip(TORSO, torso))
        q.update(zip(ARM[side], arm))
        return (self.fk.transform(f"{side}_gripper_link", q) @ self.fk.EEF_OFFSET)[:3]

    def _solve1(self, side, torso, q, target, iters, damp, step):
        lo, hi = self.lim[side][:, 0], self.lim[side][:, 1]
        q = np.clip(np.array(q, dtype=np.float64), lo, hi)
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

    def solve(self, side, torso, arm0, target, iters=150, damp=0.05, step=0.5, restarts=4, seed=0):
        """Multi-start: from arm0, from zeros, then random starts; prefer the closest solution to arm0 among good ones."""
        rng = np.random.default_rng(seed)
        lo, hi = self.lim[side][:, 0], self.lim[side][:, 1]
        starts = [np.asarray(arm0, dtype=np.float64), np.zeros(7)]
        starts += [lo + (hi - lo) * rng.uniform(0.3, 0.7, 7) for _ in range(restarts)]
        sols = [self._solve1(side, torso, s, target, iters, damp, step) for s in starts]
        good = [s for s in sols if s[1] < 5e-3]
        if good:
            return min(good, key=lambda s: np.linalg.norm(s[0] - arm0))
        return min(sols, key=lambda s: s[1])


class PressAssist:
    def __init__(
        self,
        evaluator,
        near_m=0.30,
        stall_steps=150,
        progress_m=0.01,
        min_policy_steps=300,
        reach_x=0.55,
        arm_y=0.2,
        base_steps=300,
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

    def _normal(self, radio):
        import omnigibson.utils.transform_utils as T

        R = T.quat2mat(radio.get_position_orientation()[1].cpu().float())
        n = R[:, 0].clone()
        n[2] = 0.0
        return n / (th.linalg.norm(n) + 1e-9)

    def _start(self, robot, radio, button, d):
        """Choose the arm, freeze the current arm/torso pose, and set the base goal facing the front face."""
        side = min(d, key=d.get)
        side = "left" if "left" in side else "right"
        jidx = robot.controller_joint_idx
        q = robot.get_joint_positions().cpu().double().numpy()
        n = self._normal(radio)
        b = button.cpu().float()
        bxy = robot.get_position_orientation()[0].cpu().float()[:2]
        to_robot = bxy - b[:2]
        facing = float((n[:2] @ to_robot) / (th.linalg.norm(to_robot) + 1e-9))
        yaw_goal = math.atan2(-float(n[1]), -float(n[0]))  # look at the front face
        ys = self.p["arm_y"] if side == "left" else -self.p["arm_y"]
        c, s = math.cos(yaw_goal), math.sin(yaw_goal)
        off = (c * self.p["reach_x"] - s * ys, s * self.p["reach_x"] + c * ys)
        goal_xy = (float(b[0]) - off[0], float(b[1]) - off[1])
        self.plan = dict(
            side=side,
            torso=q[jidx["trunk"]],
            arm_hold={k: q[jidx[f"arm_{k}"]] for k in ("left", "right")},
            goal=(goal_xy[0], goal_xy[1], yaw_goal),
        )
        self.log.append(f"t{self.step}:{side},d={d[side]:.3f},face={facing:.2f}")

    def _ik(self, robot, radio, button):
        n, b = self._normal(radio), button.cpu().float()
        P, side = self.plan, self.plan["side"]
        arm0 = P["arm_hold"][side]
        q_pre, r_pre = self.ik.solve(side, P["torso"], arm0, self._to_base(robot, b + self.p["pre_m"] * n))
        q_press, r_press = self.ik.solve(side, P["torso"], q_pre, self._to_base(robot, b - self.p["press_m"] * n))
        P.update(arm0=arm0, q_pre=q_pre, q_press=q_press)
        self.log.append(f"t{self.step}:ik_pre={r_pre:.3f},ik_press={r_press:.3f}")

    def _action(self, robot, arm_q=None, gripper=-1.0, base=(0.0, 0.0, 0.0)):
        a = th.zeros((1, robot.action_dim))
        P, idx = self.plan, robot.controller_action_idx
        a[0, idx["trunk"]] = th.tensor(P["torso"], dtype=a.dtype)
        for k in ("left", "right"):
            a[0, idx[f"arm_{k}"]] = th.tensor(P["arm_hold"][k], dtype=a.dtype)
            a[0, idx[f"gripper_{k}"]] = 1.0
        if arm_q is not None:
            s = P["side"]
            a[0, idx[f"arm_{s}"]] = th.tensor(arm_q, dtype=a.dtype)
            a[0, idx[f"gripper_{s}"]] = gripper
        a[0, idx["base"]] = th.tensor(base, dtype=a.dtype)
        return a

    def _base_cmd(self, robot):
        """Robot-frame holonomic velocity toward the goal pose; returns (cmd, done)."""
        gx, gy, gyaw = self.plan["goal"]
        pos, quat = robot.get_position_orientation()
        x, y, yaw = float(pos[0]), float(pos[1]), _yaw(quat)
        ex, ey = gx - x, gy - y
        c, s = math.cos(yaw), math.sin(yaw)
        fx, fy = c * ex + s * ey, -s * ex + c * ey  # error in the robot frame
        eyaw = _wrap(gyaw - yaw)
        done = math.hypot(ex, ey) < 0.03 and abs(eyaw) < math.radians(4)
        clip = lambda v, m: max(-m, min(m, v))  # noqa: E731
        return (clip(1.5 * fx, 0.25), clip(1.5 * fy, 0.25), clip(2.0 * eyaw, 0.5)), done

    def wrap(self, forward):
        def assisted_forward(*args, **kwargs):
            self.step += 1
            robot, radio, button, on, d = self._world()
            p = self.p
            if self.state == "policy":
                dmin = min(d.values())
                if dmin < self.best - p["progress_m"]:
                    self.best, self.best_step = dmin, self.step
                stalled = (
                    self.step - self.best_step >= p["stall_steps"]
                    and self.step - self.seg_start >= p["min_policy_steps"]
                )
                if not on and self.count < p["max_presses"] and dmin < p["near_m"] and (p["always"] or stalled):
                    self.count += 1
                    self._start(robot, radio, button, d)
                    self.state, self.t = "base", 0
                else:
                    return forward(*args, **kwargs)
            self.t += 1
            P, m = self.plan, p["move_steps"]
            if self.state == "base":
                cmd, done = self._base_cmd(robot)
                if done or self.t >= p["base_steps"]:
                    self.log.append(f"t{self.step}:base_{'ok' if done else 'timeout'}")
                    self._ik(robot, radio, button)
                    self.state, self.t = "pre", 0
                    return self._action(robot)
                return self._action(robot, base=cmd)
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
                    self.log.append(f"t{self.step}:{'ON' if on else 'no-toggle'}(hold),d={min(d.values()):.3f}")
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


if __name__ == "__main__":  # IK self-test (no sim): reach points in front of the robot at the policy's torso pose
    ik = ArmIK()
    torso = np.array([1.08, -1.548, -0.531, 0.0])
    for side, y in (("left", 0.2), ("right", -0.2)):
        for x in (0.5, 0.55, 0.61):
            q, r = ik.solve(side, torso, np.zeros(7), np.array([x, y, 0.6]))
            print(side, x, "residual", round(r, 4))
