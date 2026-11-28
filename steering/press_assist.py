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
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch as th

from steering.urdf_fk import ARM, TORSO, URDF, R1ProFK

RADIO_INST = "radio_receiver.n.01_1"
# Torso pose during the press: the robot's reset pose (button reachable 0.4-0.6 m ahead at table height with it).
TORSO_PRESS = np.array([1.02, -1.45, -0.47, 0.0])


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
        self.tlim = {n: lim.get(n, (-np.pi, np.pi)) for n in TORSO}

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


    def solve_full(self, side, x0, target, iters=200, damp=0.05, step=0.5, restarts=6, seed=0):
        """Torso (4) + arm (7) IK. x0 = r_[torso, arm]. Prefers the good solution closest to x0 (torso weighted x2)."""
        tl = np.array([self.tlim[n] for n in TORSO])
        lo, hi = np.r_[tl[:, 0], self.lim[side][:, 0]], np.r_[tl[:, 1], self.lim[side][:, 1]]
        rng = np.random.default_rng(seed)
        f = lambda x: self.eef(side, x[:4], x[4:])  # noqa: E731
        sols = []
        for s0 in [np.asarray(x0, dtype=np.float64)] + [lo + (hi - lo) * rng.uniform(0.3, 0.7, 11) for _ in range(restarts)]:
            x = np.clip(s0.copy(), lo, hi)
            for _ in range(iters):
                p = f(x)
                e = target - p
                if np.linalg.norm(e) < 2e-3:
                    break
                J = np.zeros((3, 11))
                for k in range(11):
                    d = x.copy()
                    d[k] += 1e-4
                    J[:, k] = (f(d) - p) / 1e-4
                x = np.clip(x + step * J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), e), lo, hi)
            sols.append((x, float(np.linalg.norm(target - f(x)))))
        w = np.r_[2 * np.ones(4), np.ones(7)]
        good = [s for s in sols if s[1] < 5e-3]
        if good:
            return min(good, key=lambda s: np.linalg.norm(w * (s[0] - x0)))
        return min(sols, key=lambda s: s[1])


class PressAssist:
    def __init__(
        self,
        evaluator,
        near_m=0.30,
        stall_steps=150,
        progress_m=0.01,
        min_policy_steps=300,
        reach_x=0.45,
        arm_y=0.2,
        base_steps=300,
        pre_m=0.06,
        press_m=0.01,
        move_steps=40,
        hold_steps=60,
        back_steps=25,
        max_presses=3,
        always=False,
        mode="torso",
        approach_max=0.85,
        radio_abort_m=0.03,
    ):
        self.ev = evaluator
        self.p = dict(locals())
        del self.p["self"], self.p["evaluator"]
        if os.environ.get("STEER_PRESS_SCRIPTED"):  # debugging / scripted demos: take over at once, no policy
            self.p.update(always=True, near_m=10.0, min_policy_steps=0, base_steps=1500)
        if os.environ.get("STEER_PRESS_MODE"):
            self.p["mode"] = os.environ["STEER_PRESS_MODE"]
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
        """Choose the arm, freeze the current pose, set the base goal.
        mode "front": base goes to squarely face the radio's front face with the button at (reach_x, +-arm_y), torso to
                      TORSO_PRESS (v3; can drive into the table when the front face points away).
        mode "torso": base only turns to face the button and drives straight toward it if it is farther than
                      approach_max; the reach is done by torso + arm IK (11 joints)."""
        side = min(d, key=d.get)
        side = "left" if "left" in side else "right"
        jidx = robot.controller_joint_idx
        q = robot.get_joint_positions().cpu().double().numpy()
        n = self._normal(radio)
        b = button.cpu().float()
        bxy = robot.get_position_orientation()[0].cpu().float()[:2]
        to_robot = bxy - b[:2]
        facing = float((n[:2] @ to_robot) / (th.linalg.norm(to_robot) + 1e-9))
        torso0 = q[jidx["trunk"]]
        P = dict(side=side, torso0=torso0, arm_hold={k: q[jidx[f"arm_{k}"]] for k in ("left", "right")})
        P["radio0"] = radio.get_position_orientation()[0].cpu().float().clone()
        if self.p["mode"] == "front":
            yaw_goal = math.atan2(-float(n[1]), -float(n[0]))  # look at the front face
            P["torso"] = TORSO_PRESS.copy()
        else:
            yaw_goal = math.atan2(-float(to_robot[1]), -float(to_robot[0]))  # look at the button
            P["torso"] = torso0.copy()
        P["goal"] = (None, None, yaw_goal)
        self.plan = P
        self.log.append(f"t{self.step}:{side},{self.p['mode']},d={d[side]:.3f},face={facing:.2f}")

    def _ik(self, robot, radio, button):
        n, b = self._normal(radio), button.cpu().float()
        P, side = self.plan, self.plan["side"]
        x0 = np.r_[P["torso"], P["arm_hold"][side]]
        tb = self._to_base(robot, b)
        self.log.append(
            f"t{self.step}:button_base=({tb[0]:.3f},{tb[1]:.3f},{tb[2]:.3f}),torso={np.round(P['torso'], 2).tolist()},"
            f"eef_fk={np.round(self.ik.eef(side, x0[:4], x0[4:]), 3).tolist()},"
            f"eef_sim={np.round(self._to_base(robot, robot.get_eef_position(side)), 3).tolist()}"
        )
        pre_b = self._to_base(robot, b + self.p["pre_m"] * n)
        press_b = self._to_base(robot, b - self.p["press_m"] * n)
        if self.p["mode"] == "front":
            q_pre, r_pre = self.ik.solve(side, P["torso"], x0[4:], pre_b)
            q_press, r_press = self.ik.solve(side, P["torso"], q_pre, press_b)
            x_pre, x_press = np.r_[P["torso"], q_pre], np.r_[P["torso"], q_press]
        else:
            x_pre, r_pre = self.ik.solve_full(side, x0, pre_b)
            x_press, r_press = self.ik.solve_full(side, x_pre, press_b)
        P.update(x0=x0, x_pre=x_pre, x_press=x_press)
        self.log.append(
            f"t{self.step}:ik_pre={r_pre:.3f},ik_press={r_press:.3f},torso_pre={np.round(x_pre[:4], 2).tolist()}"
        )

    def _action(self, robot, x=None, gripper=-1.0, base=(0.0, 0.0, 0.0)):
        """x = r_[torso(4), arm(7)] for the pressing arm; None = hold (torso per phase)."""
        a = th.zeros((1, robot.action_dim))
        P, idx = self.plan, robot.controller_action_idx
        torso = P["torso"]
        if self.state == "base":  # front mode: move the torso to TORSO_PRESS during the base phase
            w = min(1.0, self.t / self.p["move_steps"])
            torso = (1 - w) * P["torso0"] + w * P["torso"]
        for k in ("left", "right"):
            a[0, idx[f"arm_{k}"]] = th.tensor(P["arm_hold"][k], dtype=a.dtype)
            a[0, idx[f"gripper_{k}"]] = 1.0
        if x is not None:
            torso = x[:4]
            s = P["side"]
            a[0, idx[f"arm_{s}"]] = th.tensor(x[4:], dtype=a.dtype)
            a[0, idx[f"gripper_{s}"]] = gripper
        a[0, idx["trunk"]] = th.tensor(np.asarray(torso), dtype=a.dtype)
        a[0, idx["base"]] = th.tensor(base, dtype=a.dtype)
        return a

    def _base_cmd(self, robot, radio, button):
        """Robot-frame holonomic base velocity; returns (cmd, done). Closed loop on the button position in the base
        frame. front: put the button at (reach_x, +-arm_y) and face the front face. torso: face the button, approach
        straight to approach_max if farther."""
        tb = self._to_base(robot, button)
        clip = lambda v, m: max(-m, min(m, v))  # noqa: E731
        if self.p["mode"] == "front":
            ys = self.p["arm_y"] if self.plan["side"] == "left" else -self.p["arm_y"]
            fx, fy = tb[0] - self.p["reach_x"], tb[1] - ys
            eyaw = _wrap(self.plan["goal"][2] - _yaw(robot.get_position_orientation()[1]))
            done = math.hypot(fx, fy) < 0.02 and abs(eyaw) < math.radians(4)
            return (clip(1.5 * fx, 0.25), clip(1.5 * fy, 0.25), clip(2.0 * eyaw, 0.5)), done
        eyaw = math.atan2(tb[1], tb[0])
        dist = math.hypot(tb[0], tb[1])
        far = dist - (self.p["approach_max"] - 0.1)
        v = clip(1.5 * far, 0.2) if dist > self.p["approach_max"] and abs(eyaw) < math.radians(20) else 0.0
        done = abs(eyaw) < math.radians(5) and dist <= self.p["approach_max"]
        return (v, 0.0, clip(2.0 * eyaw, 0.5)), done

    def _radio_moved(self, radio):
        return float(th.linalg.norm(radio.get_position_orientation()[0].cpu().float() - self.plan["radio0"]))

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
                    self.state, self.t, self._done_n = "base", 0, 0
                else:
                    return forward(*args, **kwargs)
            self.t += 1
            P, m = self.plan, p["move_steps"]
            if self.state in ("base", "pre") and self._radio_moved(radio) > p["radio_abort_m"]:
                self.log.append(f"t{self.step}:abort_radio_moved({self._radio_moved(radio):.3f})")
                self.state = "policy"
                self._segment()
                self.ev.policy.reset()
                return forward(*args, **kwargs)
            if self.state == "base":
                cmd, done = self._base_cmd(robot, radio, button)
                self._done_n = self._done_n + 1 if done else 0
                if self._done_n >= 10 or self.t >= p["base_steps"]:
                    done = self._done_n >= 10
                    self.log.append(f"t{self.step}:base_{'ok' if done else 'timeout'}")
                    self._ik(robot, radio, button)
                    self.state, self.t = "pre", 0
                    return self._action(robot)
                return self._action(robot, base=cmd)
            if self.state == "pre":
                a = min(1.0, self.t / m)
                if self.t >= m + 5:
                    self.state, self.t = "press", 0
                return self._action(robot, (1 - a) * P["x0"] + a * P["x_pre"])
            if self.state == "press":
                a = min(1.0, self.t / m)
                x = (1 - a) * P["x_pre"] + a * P["x_press"]
                if on:
                    self.log.append(f"t{self.step}:ON(press)")
                    self.state, self.t = "back", 0
                elif self.t >= m:
                    self.state, self.t = "hold", 0
                return self._action(robot, x)
            if self.state == "hold":
                if on or self.t >= p["hold_steps"]:
                    self.log.append(f"t{self.step}:{'ON' if on else 'no-toggle'}(hold),d={min(d.values()):.3f}")
                    self.state, self.t = "back", 0
                return self._action(robot, P["x_press"])
            if self.state == "back":
                a = min(1.0, self.t / p["back_steps"])
                x = (1 - a) * P["x_press"] + a * P["x_pre"]
                if self.t >= p["back_steps"]:
                    self.state = "policy"
                    self._segment()
                    self.ev.policy.reset()
                return self._action(robot, x)
            return forward(*args, **kwargs)

        return assisted_forward

    def summary(self):
        return {"presses": self.count, "press_log": ";".join(self.log)}


if __name__ == "__main__":  # IK self-test (no sim): reach points in front of the robot at the policy's torso pose
    ik = ArmIK()
    for side, y in (("left", 0.2), ("right", -0.2)):
        for x, z in ((0.45, 0.52), (0.5, 0.5), (0.55, 0.52), (0.45, 0.62)):
            q, r = ik.solve(side, TORSO_PRESS, np.zeros(7), np.array([x, y, z]))
            x11, r11 = ik.solve_full(side, np.r_[TORSO_PRESS, np.zeros(7)], np.array([x + 0.35, y, z]))
            print(side, x, z, "arm-only residual", round(r, 4), "| torso+arm at x+0.35:", round(r11, 4))
