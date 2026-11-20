"""Causal test: does fixing base alignment fix the hard (radio-on-the-right) instances?

AlignAssist wraps the evaluator's policy.forward. When the base has moved, then stopped near the radio while
more than `threshold_deg` off facing it (and is not holding the radio), it pauses the policy, rotates the base in
place to face the radio's true position (from the sim), holds still briefly, resets the policy's action queue
(queued actions were planned for the old heading) and resumes. At most `max_interventions` per episode.

Used via run_rollouts.py --align-assist.
"""

import math
import os

import torch as th

RADIO_INST = "radio_receiver.n.01_1"


def _yaw_from_quat(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class AlignAssist:
    def __init__(
        self,
        evaluator,
        threshold_deg=15.0,
        near_m=1.5,
        stop_window=30,
        stop_pos_m=0.03,
        stop_yaw_deg=3.0,
        min_moved_m=0.2,
        min_turned_deg=20.0,
        done_deg=3.0,
        settle_steps=10,
        max_rotate_steps=300,
        max_interventions=1,
        gain=2.0,
        max_wz=0.6,
        min_wz=0.1,
    ):
        self.ev = evaluator
        self.p = dict(locals())
        del self.p["self"], self.p["evaluator"]
        self.reset()

    def reset(self):
        self.state = "policy"
        self.hist = []
        self.start = None
        self.last_action = None
        self.count = 0
        self.log = []  # dicts: start, end, err_before, err_after, rotate_steps
        self.step = 0
        self._cur = None

    # --- sim state ---------------------------------------------------------------------------------------------
    def _pose(self):
        acc = self.ev.instance_eval_states[0].env_accessor
        robot = acc.robot
        pos, quat = robot.get_position_orientation()
        entity = acc.object_scope[RADIO_INST]
        radio = getattr(entity, "wrapped_obj", entity)
        rp = radio.get_position_orientation()[0]
        x, y, yaw = float(pos[0]), float(pos[1]), _yaw_from_quat(quat)
        err = _wrap(math.atan2(float(rp[1]) - y, float(rp[0]) - x) - yaw)
        dist = math.hypot(float(rp[0]) - x, float(rp[1]) - y)
        grasping = any(robot.is_grasping(arm=arm, candidate_obj=radio).value > 0 for arm in robot.arm_names)
        return robot, x, y, yaw, err, dist, grasping

    def _should_trigger(self, x, y, yaw, err, dist, grasping):
        p = self.p
        if self.count >= p["max_interventions"] or grasping:
            return False
        x0, y0, yaw0 = self.start
        moved = (
            math.hypot(x - x0, y - y0) > p["min_moved_m"] or abs(math.degrees(_wrap(yaw - yaw0))) > p["min_turned_deg"]
        )
        if not moved or len(self.hist) < p["stop_window"]:
            return False
        hx, hy, hyaw = self.hist[-p["stop_window"]]
        stopped = (
            math.hypot(x - hx, y - hy) < p["stop_pos_m"] and abs(math.degrees(_wrap(yaw - hyaw))) < p["stop_yaw_deg"]
        )
        return stopped and dist < p["near_m"] and abs(math.degrees(err)) > p["threshold_deg"]

    # --- actions -----------------------------------------------------------------------------------------------
    def _action(self, robot, wz):
        a = self.last_action.clone() if self.last_action is not None else th.zeros((1, robot.action_dim))
        base = robot.controller_action_idx["base"]
        a[..., base] = 0.0
        a[..., base[2]] = wz
        return a

    def wrap(self, forward):
        def assisted_forward(*args, **kwargs):
            self.step += 1
            robot, x, y, yaw, err, dist, grasping = self._pose()
            if self.start is None:
                self.start = (x, y, yaw)
            self.hist.append((x, y, yaw))
            p = self.p

            if self.state == "policy" and self._should_trigger(x, y, yaw, err, dist, grasping):
                self.state = "rotate"
                self._cur = {"start": self.step, "err_before": round(math.degrees(err), 1), "dist": round(dist, 2)}
            if self.state == "rotate":
                rot_steps = self.step - self._cur["start"]
                if abs(math.degrees(err)) < p["done_deg"] or rot_steps >= p["max_rotate_steps"]:
                    self.state, self._settle = "settle", 0
                    self._cur["rotate_steps"] = rot_steps
                else:
                    wz = max(-p["max_wz"], min(p["max_wz"], p["gain"] * err))
                    wz = math.copysign(max(abs(wz), p["min_wz"]), wz)
                    return self._action(robot, wz)
            if self.state == "settle":
                self._settle += 1
                if self._settle <= p["settle_steps"]:
                    return self._action(robot, 0.0)
                self._cur.update(end=self.step, err_after=round(math.degrees(err), 1))
                self.log.append(self._cur)
                self.count += 1
                self.state = "policy"
                self.hist = []
                self.ev.policy.reset()  # drop Comet's queued (now stale) action chunk

            out = forward(*args, **kwargs)
            self.last_action = out.detach().clone()
            return out

        return assisted_forward

    @property
    def active(self):
        return int(self.state != "policy")

    def summary(self):
        return {
            "assist_interventions": self.count,
            "assist_log": ";".join(
                f"{d['start']}-{d.get('end', '?')}:{d['err_before']}->{d.get('err_after', '?')}deg@{d['dist']}m"
                for d in self.log + ([self._cur] if self.state != "policy" and self._cur else [])
            ),
        }


class ApproachAssist:
    """Shared-autonomy start: before the policy acts, drive the base to `standoff_m` from the radio's true position and
    face it, holding torso/arms at their current positions with grippers open; then hand over to the policy.
    Used via run_rollouts.py --approach-assist <standoff_m>.
    """

    def __init__(self, evaluator, standoff_m=0.8, max_steps=600, max_v=0.3, max_wz=0.6, done_deg=4.0, settle_steps=10):
        self.ev = evaluator
        self.p = dict(
            standoff_m=standoff_m,
            max_steps=max_steps,
            max_v=max_v,
            max_wz=max_wz,
            done_deg=done_deg,
            settle_steps=settle_steps,
        )
        self.reset()

    def reset(self):
        self.state = "approach"
        self.step = 0
        self.hold = None
        self.log = {}

    @property
    def active(self):
        return int(self.state != "policy")

    def _hold_action(self, robot):
        a = th.zeros((1, robot.action_dim))
        q = robot.get_joint_positions()
        jidx = robot.controller_joint_idx
        for name, aidx in robot.controller_action_idx.items():
            if name.startswith(("arm", "trunk")):
                a[0, aidx] = q[jidx[name]]
            elif name.startswith("gripper"):
                a[0, aidx] = 1.0
        return a

    def wrap(self, forward):
        align = AlignAssist(self.ev)

        def assisted_forward(*args, **kwargs):
            self.step += 1
            if self.state == "policy":
                return forward(*args, **kwargs)
            robot, x, y, yaw, err, dist, _ = align._pose()
            if self.hold is None:
                self.hold = self._hold_action(robot)
                self.log["start_dist"] = round(dist, 2)
                self.log["start_err"] = round(math.degrees(err), 1)
            p = self.p
            a = self.hold.clone()
            base = robot.controller_action_idx["base"]
            if self.state == "approach":
                done = dist <= p["standoff_m"] + 0.05 and abs(math.degrees(err)) < p["done_deg"]
                if done or self.step >= p["max_steps"]:
                    self.state, self._settle = "settle", 0
                else:
                    # robot-frame command: turn toward the radio; drive forward only when roughly facing it
                    wz = max(-p["max_wz"], min(p["max_wz"], 2.0 * err))
                    v = 0.0
                    if abs(err) < math.radians(25) and dist > p["standoff_m"]:
                        v = max(0.05, min(p["max_v"], 1.5 * (dist - p["standoff_m"])))  # floor avoids creeping
                    a[0, base] = th.tensor([v, 0.0, wz], dtype=a.dtype)
                    return a
            if self.state == "settle":
                self._settle += 1
                if self._settle <= p["settle_steps"]:
                    return a
                self.log.update(end_step=self.step, end_dist=round(dist, 2), end_err=round(math.degrees(err), 1))
                self.state = "policy"
                self.ev.policy.reset()
            return forward(*args, **kwargs)

        return assisted_forward

    def summary(self):
        return {"approach_log": ";".join(f"{k}={v}" for k, v in self.log.items())}


class RecoveryAssist:
    """Failure monitor + retry (oracle-triggered for now). While the policy runs, watch two failure signals:
      stall:  no new closest hand-to-button distance (by > `progress_m`) for `stall_steps` steps
      pushed: radio moved > `pushed_m` from where it was at the last (re)approach while no hand grasps it
    On a trigger: open the grippers (lets go of the radio) holding the arms where they are, back the base off to
    `backoff_m`, then arms/torso back to their episode-start pose, re-approach to `standoff_m` facing the radio, settle, reset the policy's action
    queue and hand over again. At most `max_recoveries` per episode. Used via run_rollouts.py --recovery.
    """

    def __init__(
        self,
        evaluator,
        stall_steps=500,
        progress_m=0.02,
        pushed_m=0.25,
        min_policy_steps=300,
        standoff_m=0.8,
        backoff_m=1.2,
        max_recoveries=2,
        home_steps=60,
        max_phase_steps=400,
    ):
        self.ev = evaluator
        self.p = dict(locals())
        del self.p["self"], self.p["evaluator"]
        if os.environ.get("STEER_RECOVERY_STALL"):  # debugging: force early triggers
            self.p["stall_steps"] = self.p["min_policy_steps"] = int(os.environ["STEER_RECOVERY_STALL"])
        self.align = AlignAssist(evaluator)
        self.reset()

    def reset(self):
        self.state = "policy"
        self.step = 0
        self.home = None
        self.count = 0
        self.log = []
        self._new_policy_segment()

    def _new_policy_segment(self):
        self.seg_start = self.step
        self.best = float("inf")
        self.best_step = self.step
        self.radio0 = None

    @property
    def active(self):
        return int(self.state != "policy")

    def _signals(self):
        from omnigibson.object_states import ToggledOn

        acc = self.ev.instance_eval_states[0].env_accessor
        robot = acc.robot
        entity = acc.object_scope[RADIO_INST]
        radio = getattr(entity, "wrapped_obj", entity)
        button = radio.states[ToggledOn].marker.get_position_orientation()[0]
        d = min(float(th.linalg.norm(robot.get_eef_position(arm) - button)) for arm in robot.arm_names)
        grasping = any(robot.is_grasping(arm=arm, candidate_obj=radio).value > 0 for arm in robot.arm_names)
        return robot, d, radio.get_position_orientation()[0], grasping

    def _pose_action(self, robot):
        a = (self.freeze if self.state in ("open", "backoff") else self.home).clone()
        a[0, robot.controller_action_idx["base"]] = 0.0
        return a

    def wrap(self, forward):
        def assisted_forward(*args, **kwargs):
            self.step += 1
            p = self.p
            robot, d, rpos, grasping = self._signals()
            if self.home is None:
                self.home = ApproachAssist._hold_action(None, robot)  # episode-start arms/torso, grippers open
            if self.state == "policy":
                if self.radio0 is None:
                    self.radio0 = rpos.clone()
                if d < self.best - p["progress_m"]:
                    self.best, self.best_step = d, self.step
                ran = self.step - self.seg_start
                stall = ran >= p["min_policy_steps"] and self.step - self.best_step >= p["stall_steps"]
                pushed = not grasping and float(th.linalg.norm(rpos[:2] - self.radio0[:2])) > p["pushed_m"]
                if self.count < p["max_recoveries"] and (stall or pushed):
                    self.state, self.phase_start = "open", self.step
                    self.freeze = ApproachAssist._hold_action(None, robot)  # current arms/torso, grippers open
                    self._cur = {
                        "start": self.step,
                        "why": "stall" if stall else "pushed",
                        "best_cm": round(100 * self.best, 1),
                    }
                else:
                    return forward(*args, **kwargs)
            _, x, y, yaw, err, dist, _ = self.align._pose()
            a = self._pose_action(robot)
            base = robot.controller_action_idx["base"]
            n = self.step - self.phase_start
            if self.state == "open":  # let go (grippers open), arms held where they are, base still
                if n >= 20:
                    self.state, self.phase_start = "backoff", self.step
                return a
            if self.state == "backoff":  # arms held; face the radio and reverse until backoff_m away
                if dist >= p["backoff_m"] or n >= p["max_phase_steps"]:
                    self.state, self.phase_start = "home", self.step
                    return self._pose_action(robot)
                else:
                    wz = max(-0.6, min(0.6, 2.0 * err))
                    a[0, base] = th.tensor([-0.25 if abs(err) < math.radians(45) else 0.0, 0.0, wz], dtype=a.dtype)
                    return a
            if self.state == "home":  # clear of the radio now: arms/torso back to the episode-start pose
                if n >= p["home_steps"]:
                    self.state, self.phase_start = "approach", self.step
                return a
            if self.state == "approach":
                done = dist <= p["standoff_m"] + 0.05 and abs(math.degrees(err)) < 4.0
                if done or n >= p["max_phase_steps"]:
                    self.state, self.phase_start = "settle", self.step
                else:
                    wz = max(-0.6, min(0.6, 2.0 * err))
                    v = 0.0
                    if abs(err) < math.radians(25) and dist > p["standoff_m"]:
                        v = max(0.05, min(0.3, 1.5 * (dist - p["standoff_m"])))
                    a[0, base] = th.tensor([v, 0.0, wz], dtype=a.dtype)
                    return a
            if self.state == "settle":
                if n < 10:
                    return a
                self._cur.update(end=self.step, dist=round(dist, 2), err=round(math.degrees(err), 1))
                self.log.append(self._cur)
                self.count += 1
                self.state = "policy"
                self._new_policy_segment()
                self.ev.policy.reset()
            return forward(*args, **kwargs)

        return assisted_forward

    def summary(self):
        cur = [self._cur] if self.state != "policy" else []  # include one still in progress at episode end
        return {
            "recoveries": self.count,
            "recovery_log": ";".join(
                f"{c['start']}-{c.get('end', '?')}:{c['why']}(best {c['best_cm']}cm)" for c in self.log + cur
            ),
        }
