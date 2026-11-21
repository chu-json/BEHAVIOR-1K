"""Paced download of the radio task's per-episode metadata (meta/episodes/task-0000/*.json: instance-id mapping for
the seg_instance_id videos) from behavior-1k/2025-challenge-demos. Run with openpi-comet's venv (has huggingface_hub).
"""
import os
import time

from huggingface_hub import hf_hub_download

REPO, DEST = "behavior-1k/2025-challenge-demos", "/home/behavior/ext/data/2025-challenge-demos"
eps = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(f"{DEST}/data/task-0000"))
todo = [f"meta/episodes/task-0000/episode_{e:08d}.json" for e in eps]
todo = [p for p in todo if not os.path.exists(os.path.join(DEST, p))]
print(len(todo), "to download", flush=True)
for i, p in enumerate(todo):
    for attempt in range(30):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=DEST)
            break
        except Exception as e:
            w = 30 if "429" in str(e) else 5
            print(f"retry {p} in {w}s: {str(e)[:80]}", flush=True)
            time.sleep(w)
    time.sleep(1.0)
    if i % 50 == 0:
        print(i, "done", flush=True)
print("DONE", flush=True)
