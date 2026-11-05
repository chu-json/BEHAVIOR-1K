"""Per-step diagnostic trace for turning_on_radio rollouts (used by run_rollouts.py --trace).

The radio's only goal predicate is (toggled_on radio). ToggledOn flips when a robot finger link overlaps the
radio's toggle-button marker for m.CAN_TOGGLE_SECONDS (object_states/toggle.py). So the useful questions for a
failure are: did the robot get there (base), did a finger reach the button (distance), did it knock or pick
up the radio instead (radio displacement / grasp), and did it stall.
"""

import csv
import math
import os

import torch as th

from omnigibson.object_states import ToggledOn

TRACE_FIELDS = [
    "step",
    "toggled_on",
    "base_x",
    "base_y",
    "base_yaw",
    "finger_button_dist_left",
    "finger_button_dist_right",
    "grasp_radio_left",
    "grasp_radio_right",
    "radio_disp",
    "radio_z",
    "finger_contact_radio",  # fork ToggledOn: a finger link is in contact with the radio
    "fork_toggle_timer_s",  # fork ToggledOn: seconds of finger-overlapping-marker so far (flips at 0.15 s)
    "overlap_2025",  # 2025 rule's sphere test (PhysX overlap_sphere at the marker vs finger links)
]
SUMMARY_FIELDS = [
    "min_finger_button_dist",
    "first_step_within_5cm",
    "steps_within_5cm",
    "grasped_radio_steps",
    "max_radio_disp",
    "final_radio_disp",
    "base_dist_traveled",
    "last_progress_step",
    "contact_steps",
    "overlap_2025_steps",
    "max_overlap_2025_run",  # longest run of consecutive overlap_2025 steps (2025 toggles at 5 while in contact)
    "max_fork_timer_s",
]
RADIO_INST = "radio_receiver.n.01_1"
NEAR_BUTTON_M = 0.05


class RolloutTracer:
    def __init__(self, evaluator, out_dir):
        self.evaluator = evaluator
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []
        self._radio = None

    def reset(self):
        self.rows = []
        self._radio = None

    def _setup(self):
        acc = self.evaluator.instance_eval_states[0].env_accessor
        self.robot = acc.robot
        entity = acc.object_scope[RADIO_INST]
        self._radio = getattr(entity, "wrapped_obj", entity)
        self._toggle = self._radio.states[ToggledOn]
        self._obj_idx = next(o for o, x in enumerate(ToggledOn.IDX_OBJS[0]) if x is self._radio)
        self._radio_p0 = self._radio.get_position_orientation()[0].clone()

    def record(self, step):
        if self._radio is None:
            self._setup()
        robot, radio = self.robot, self._radio
        button = self._toggle.marker.get_position_orientation()[0]
        pos, quat = robot.get_position_orientation()
        yaw = math.atan2(
            2 * (quat[3] * quat[2] + quat[0] * quat[1]).item(), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2).item()
        )
        row = {"step": step, "toggled_on": int(bool(self._toggle.get_value()))}
        row.update(base_x=round(pos[0].item(), 4), base_y=round(pos[1].item(), 4), base_yaw=round(yaw, 4))
        for arm, side in zip(robot.arm_names, ("left", "right")):
            fingers = th.stack([link.get_position_orientation()[0] for link in robot.finger_links[arm]])
            row[f"finger_button_dist_{side}"] = round(th.linalg.norm(fingers - button, dim=-1).min().item(), 4)
            row[f"grasp_radio_{side}"] = int(robot.is_grasping(arm=arm, candidate_obj=radio).value > 0)
        rp = radio.get_position_orientation()[0]
        row["radio_disp"] = round(th.linalg.norm(rp - self._radio_p0).item(), 4)
        row["radio_z"] = round(rp[2].item(), 4)
        import warp as wp

        n_obj = len(ToggledOn.IDX_OBJS[0])
        mask = ToggledOn._mask_can_toggle_flat
        row["finger_contact_radio"] = int(wp.to_torch(mask)[self._obj_idx].item() >= 1) if mask is not None else ""
        timer = ToggledOn._robots_can_toggle_time
        row["fork_toggle_timer_s"] = (
            round(wp.to_torch(timer).reshape(-1)[self._obj_idx].item(), 4) if timer is not None and n_obj else ""
        )
        row["overlap_2025"] = int(bool(ToggledOn._check_overlap(0, self._obj_idx)))
        self.rows.append(row)

    def finish(self, stem):
        path = os.path.join(self.out_dir, stem + "_trace.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        return path, self.summary()

    def summary(self):
        if not self.rows:
            return {k: "" for k in SUMMARY_FIELDS}
        d = [min(r["finger_button_dist_left"], r["finger_button_dist_right"]) for r in self.rows]
        near = [r["step"] for r, x in zip(self.rows, d) if x < NEAR_BUTTON_M]
        base = sum(
            math.hypot(b["base_x"] - a["base_x"], b["base_y"] - a["base_y"]) for a, b in zip(self.rows, self.rows[1:])
        )
        # Last step at which the closest-finger distance hit a new minimum (by >1 cm): a crude "stalled since" marker.
        best, last_progress = float("inf"), 0
        for r, x in zip(self.rows, d):
            if x < best - 0.01:
                best, last_progress = x, r["step"]
        return {
            "min_finger_button_dist": round(min(d), 4),
            "first_step_within_5cm": near[0] if near else "",
            "steps_within_5cm": len(near),
            "grasped_radio_steps": sum(r["grasp_radio_left"] or r["grasp_radio_right"] for r in self.rows),
            "max_radio_disp": max(r["radio_disp"] for r in self.rows),
            "final_radio_disp": self.rows[-1]["radio_disp"],
            "base_dist_traveled": round(base, 3),
            "last_progress_step": last_progress,
            "contact_steps": sum(int(r["finger_contact_radio"] or 0) for r in self.rows),
            "overlap_2025_steps": sum(r["overlap_2025"] for r in self.rows),
            "max_overlap_2025_run": max_run([r["overlap_2025"] for r in self.rows]),
            "max_fork_timer_s": max(float(r["fork_toggle_timer_s"] or 0) for r in self.rows),
        }


def max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best
