"""Causal test: does fixing base alignment fix the hard (radio-on-the-right) instances?

AlignAssist wraps the evaluator's policy.forward. When the base has moved, then stopped near the radio while
more than `threshold_deg` off facing it (and is not holding the radio), it pauses the policy, rotates the base in
place to face the radio's true position (from the sim), holds still briefly, resets the policy's action queue
(queued actions were planned for the old heading) and resumes. At most `max_interventions` per episode.

Used via run_rollouts.py --align-assist.
"""

import math

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
