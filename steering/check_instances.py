"""Check whether each task instance's initial state survives loading in this fork (no policy needed).

For each instance: load it exactly as the evaluator does (TRO state -> settle 25 physics steps -> reset), then
report where the radio is vs. the TRO file, whether it is still on the table (bbox overlap + BDDL OnTop), and
whether the init conditions of the BDDL activity hold. Uses a zero-action LocalPolicy, so it never talks to
(or disturbs) a running policy server.

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/check_instances.py \
        --instances 242 295 211 203 109 181 197 187 214 139 --out steering/runs/instance_check.csv
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from steering.run_rollouts import RADIO_2025_VAL  # noqa: E402

RADIO, TABLE = "radio_receiver.n.01_1", "table.n.02_1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="turning_on_radio")
    ap.add_argument("--instances", type=int, nargs="+", default=RADIO_2025_VAL)
    ap.add_argument("--settle-steps", type=int, default=60, help="Extra zero-action env steps after reset.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch as th
    from omegaconf import OmegaConf

    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.macros import gm
    from omnigibson.object_states import OnTop
    from omnigibson.utils.asset_utils import get_task_instance_path

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "local",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": 10_000,
            "write_video": False,
            "mode": "train",
            "seed": 0,
            "num_envs": 1,
            "task": {"name": args.task},
            "robot": None,
        }
    )
    unwrap = lambda e: getattr(e, "wrapped_obj", e)  # noqa: E731
    rows = []
    with BatchedEvaluator(cfg) as ev:
        acc = ev.instance_eval_states[0].env_accessor
        scene_model = ev.env.task.scene_name
        for inst in args.instances:
            ev.load_batch({0: inst})
            radio, table = unwrap(acc.object_scope[RADIO]), unwrap(acc.object_scope[TABLE])
            after_load = radio.get_position_orientation()[0].clone()
            # Hold the robot still: zero base velocity, every other controller at its current joint targets.
            robot = acc.robot
            hold = th.zeros(robot.action_dim)
            q = robot.get_joint_positions()
            joint_idx = robot.controller_joint_idx
            for name, action_idx in robot.controller_action_idx.items():
                if name.startswith(("arm", "trunk")):
                    hold[action_idx] = q[joint_idx[name]]
                elif name.startswith("gripper"):
                    hold[action_idx] = 1.0  # open
            for _ in range(args.settle_steps):
                ev.env.step(hold.unsqueeze(0))
            final = radio.get_position_orientation()[0]

            tro_name = ev.env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=ev.env.task.activity_name,
                activity_definition_id=ev.env.task.activity_definition_id,
                activity_instance_id=inst,
            )
            tro_path = get_task_instance_path(
                scene_model, f"{scene_model}_task_{ev.env.task.activity_name}_instances/{tro_name}-tro_state"
            )
            tro = json.load(open(tro_path))
            tro_pos = th.tensor(tro[RADIO]["root_link"]["pos"])
            lo, hi = table.aabb
            rows.append(
                {
                    "instance_id": inst,
                    "tro_radio_xyz": [round(x, 3) for x in tro_pos.tolist()],
                    "after_load_xyz": [round(x, 3) for x in after_load.tolist()],
                    "after_settle_xyz": [round(x, 3) for x in final.tolist()],
                    "drift_from_tro_m": round(th.linalg.norm(final - tro_pos).item(), 3),
                    "radio_ontop_table": int(bool(radio.states[OnTop].get_value(table))),
                    "radio_xy_inside_table_aabb": int(bool(lo[0] <= final[0] <= hi[0] and lo[1] <= final[1] <= hi[1])),
                    "table_aabb_lo": [round(x, 3) for x in lo.tolist()],
                    "table_aabb_hi": [round(x, 3) for x in hi.tolist()],
                    "table_pos_xyz": [round(x, 3) for x in table.get_position_orientation()[0].tolist()],
                    "tro_table_xyz": [round(x, 3) for x in tro[TABLE]["root_link"]["pos"]],
                }
            )
            print("INSTANCE_CHECK", rows[-1], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
