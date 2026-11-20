"""Scene perturbations for steerability tests on turning_on_radio (steering/RESEARCH.md, types S1/S2/S4).

  clutter N     N distractor objects (BEHAVIOR assets) placed on the radio's table around the radio
  occluder      a grey panel between the robot's start position and the radio, offset sideways so it hides about
                half of the radio from the start view
  radio_yaw D   rotate the radio by D degrees about z at episode start (S4: button facing elsewhere)

The extra objects are added to the env config at creation (adding objects to a running sim can break cached
per-object buffers, e.g. ToggledOn's marker arrays) and are kinematic: they do not fall or drift between episodes
and the robot cannot push them (v1 limitation). Each episode they are re-placed after the task reset, from a
per-episode RNG (perturb_seed + rollout seed), so pairs of (setup, seed) see the same layout across conditions.
Used via run_rollouts.py --clutter N / --occluder / --radio-yaw-deg D.
"""

import math

import torch as th

RADIO = "radio_receiver.n.01_1"
# small tabletop objects (category, model) available in datasets/behavior-1k-assets
CLUTTER_POOL = [
    ("mug", "dhnxww"),
    ("bowl", "adciys"),
    ("apple", "agveuv"),
    ("bottle_of_water", "atvnqy"),
    ("notebook", "aanuhi"),
    ("mug", "ehnmxj"),
    ("pen", "enfumt"),
    ("bowl", "ajzltc"),
    ("notebook", "bnsvvw"),
    ("pen", "gqwdor"),
]
PARK = (60.0, 60.0, -20.0)  # far away and below the house while unused


def extra_object_configs(clutter=0, occluder=False):
    cfgs = []
    for i in range(clutter):
        cat, model = CLUTTER_POOL[i % len(CLUTTER_POOL)]
        cfgs.append(
            {
                "type": "DatasetObject",
                "name": f"steer_clutter_{i}",
                "category": cat,
                "model": model,
                "kinematic_only": True,
                "position": [PARK[0] + i, PARK[1], PARK[2]],
            }
        )
    if occluder:
        cfgs.append(
            {
                "type": "PrimitiveObject",
                "name": "steer_occluder",
                "primitive_type": "Cube",
                "rgba": [0.55, 0.55, 0.58, 1.0],
                "scale": [0.03, 0.22, 0.24],  # thin panel: 3 cm deep, 22 cm wide, 24 cm tall
                "kinematic_only": True,
                "position": [PARK[0], PARK[1] + 5, PARK[2]],
            }
        )
    return cfgs


def _yaw_quat(yaw):
    return th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


class ScenePerturber:
    def __init__(self, clutter=0, occluder=False, radio_yaw_deg=0.0, perturb_seed=0):
        self.n_clutter, self.occluder, self.radio_yaw = clutter, occluder, math.radians(radio_yaw_deg)
        self.perturb_seed = perturb_seed
        self.log = {}

    @property
    def active(self):
        return self.n_clutter > 0 or self.occluder or self.radio_yaw != 0.0

    def apply(self, env, env_idx, rollout_seed):
        """Call right after the task reset. Moves the radio / extra objects; returns a dict for the CSV."""
        import omnigibson as og

        scene = env.scenes[env_idx]
        robot = scene.robots[0]
        entity = env.task.object_scopes[env_idx][RADIO]
        radio = getattr(entity, "wrapped_obj", entity)
        g = th.Generator().manual_seed(int(self.perturb_seed) * 100003 + int(rollout_seed))
        log = {}

        rpos, rquat = radio.get_position_orientation()
        if self.radio_yaw != 0.0:
            import omnigibson.utils.transform_utils as T

            q = T.quat_multiply(_yaw_quat(self.radio_yaw).to(rquat.device), rquat)
            radio.set_position_orientation(position=rpos, orientation=q)
            log["radio_yaw_deg"] = round(math.degrees(self.radio_yaw), 1)

        lo, hi = (v.cpu().float() for v in radio.aabb)
        center = (lo + hi) / 2
        top = float(lo[2])  # table surface ~ bottom of the radio
        bxy = robot.get_position_orientation()[0][:2].cpu().float()
        to_robot = bxy - center[:2]
        toward = math.atan2(float(to_robot[1]), float(to_robot[0]))

        placed = []
        for i in range(self.n_clutter):
            obj = scene.object_registry("name", f"steer_clutter_{i}")
            olo, ohi = (v.cpu().float() for v in obj.aabb)
            half_h = float(ohi[2] - olo[2]) / 2
            for _ in range(50):  # rejection-sample a free spot 12-30 cm from the radio center
                ang = float(th.rand(1, generator=g)) * 2 * math.pi
                d = 0.12 + 0.18 * float(th.rand(1, generator=g))
                xy = center[:2] + d * th.tensor([math.cos(ang), math.sin(ang)])
                if all(float(th.linalg.norm(xy - p)) > 0.09 for p in placed):
                    break
            placed.append(xy)
            yaw = float(th.rand(1, generator=g)) * 2 * math.pi
            pos = th.tensor([float(xy[0]), float(xy[1]), top + half_h + 0.005])
            obj.set_position_orientation(position=pos, orientation=_yaw_quat(yaw))
        if self.n_clutter:
            log["clutter"] = self.n_clutter

        if self.occluder:
            occ = scene.object_registry("name", "steer_occluder")
            side = 1.0 if float(th.rand(1, generator=g)) < 0.5 else -1.0
            fwd = th.tensor([math.cos(toward), math.sin(toward)])
            perp = th.tensor([-math.sin(toward), math.cos(toward)])
            xy = center[:2] + 0.16 * fwd + side * 0.08 * perp  # 16 cm toward the robot, 8 cm to one side
            pos = th.tensor([float(xy[0]), float(xy[1]), top + 0.12])
            occ.set_position_orientation(position=pos, orientation=_yaw_quat(toward))
            log["occluder_side"] = int(side)

        for _ in range(5):  # let the scene settle and refresh poses/renders
            og.sim.step()
        self.log = log
        return log
