"""Step 1: load Rs_int (no robot yet) and print where the tables are."""
import omnigibson as og
from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True

cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "not_load_object_categories": ["ceilings"],
    },
    "robots": [],
    "task": {"type": "DummyTask"},
}

env = og.Environment(configs=cfg)

print("NUM_OBJECTS", len(env.scene.objects), flush=True)
for obj in env.scene.objects:
    if "table" in obj.name:
        lo, hi = obj.aabb
        print(f"{obj.name}: pos={obj.get_position_orientation()[0].tolist()} aabb_lo={lo.tolist()} aabb_hi={hi.tolist()}", flush=True)

og.shutdown()
