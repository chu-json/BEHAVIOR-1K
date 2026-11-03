#!/bin/bash
# Unattended vision pipeline: for each data size N in $SIZES, wait until N successful demos exist in FINISHED
# collection chunks, convert them to a LeRobot dataset, train a diffusion policy, then evaluate checkpoints.
# Status lines go to data/vision_pipeline.log.
cd "$(dirname "$0")"
PY="$HOME/miniconda3/condabin/conda run --no-capture-output -n behavior"
LOG=data/vision_pipeline.log
SIZES=${SIZES:-"500 1000"}
STEPS=${STEPS:-50000}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

closed_successes() {  # successes in chunk files that are no longer being written (HDF5 locking skips open ones)
  $PY python -c "
import glob, h5py
from step10_to_lerobot import open_by_another_process
ok = 0
for p in glob.glob('data/vision/demos_w*_c*.hdf5'):
    if open_by_another_process(p):
        continue
    try:
        with h5py.File(p, 'r') as f:
            ok += sum(bool(f[k].attrs.get('success', False)) for k in f.keys())
    except OSError:
        pass
print(ok)" 2>/dev/null | tail -1
}

for N in $SIZES; do
  log "waiting for $N successful demos in finished chunks"
  until [ "$(closed_successes)" -ge "$N" ] 2>/dev/null; do sleep 60; done
  NAME=franka_apple_wrist_$N
  if [ -f data/lerobot/$NAME/meta/info.json ]; then
    log "dataset data/lerobot/$NAME already exists, reusing it"
  else
    log "converting $(closed_successes) successful demos (capped at $N) -> data/lerobot/$NAME"
    $PY python step10_to_lerobot.py --name $NAME --max $N > data/convert_$N.log 2>&1
    log "$(tail -1 data/convert_$N.log)"
  fi

  OUT=data/train/diffusion_$N
  rm -rf $OUT
  log "training diffusion policy on $N demos for $STEPS steps -> $OUT"
  $PY lerobot-train \
    --dataset.repo_id=local/$NAME --dataset.root=data/lerobot/$NAME \
    --policy.type=diffusion --policy.push_to_hub=false --policy.device=cuda \
    --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \
    --policy.crop_shape="[120,120]" --policy.down_dims="[256,512,1024]" \
    --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
    --steps=$STEPS --batch_size=64 --log_freq=500 --save_freq=10000 --num_workers=10 \
    --output_dir=$OUT --wandb.enable=false > data/train_$N.log 2>&1
  log "training done (exit $?): $(grep -a 'step:' data/train_$N.log | tail -1 | grep -o 'step:.*loss:[0-9.]*')"

  for CK in 020000 040000 last; do
    [ -d $OUT/checkpoints/$CK/pretrained_model ] || continue
    log "evaluating $N-demo policy, checkpoint $CK, on 50 unseen layouts"
    OMNIGIBSON_HEADLESS=1 $PY python step11_eval_vision.py --ckpt $OUT/checkpoints/$CK/pretrained_model --episodes 50 \
      > data/eval_vision_${N}_$CK.log 2>&1
    log "$N demos, ckpt $CK: $(grep -a FINAL data/eval_vision_${N}_$CK.log)"
  done
done
log "VISION PIPELINE DONE"
