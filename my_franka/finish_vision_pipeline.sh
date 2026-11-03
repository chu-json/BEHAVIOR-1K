#!/bin/bash
# Second half of the vision pipeline (collection was stopped at 962 successful demos instead of 1000):
#   1. wait for the running 500-demo diffusion training to finish, evaluate its 40k and final checkpoints
#   2. convert ALL successful demos, train a diffusion policy on them, evaluate 20k / 40k / final
# Status lines go to data/vision_pipeline.log (same log as run_vision_pipeline.sh).
cd "$(dirname "$0")"
PY="$HOME/miniconda3/condabin/conda run --no-capture-output -n behavior"
LOG=data/vision_pipeline.log
STEPS=${STEPS:-50000}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

evaluate() {  # $1 = run name (500 / all), $2 = checkpoint dir name
  local CKPT=data/train/diffusion_$1/checkpoints/$2/pretrained_model
  [ -d $CKPT ] || { log "no checkpoint $CKPT"; return; }
  log "evaluating $1-demo policy, checkpoint $2, on 50 unseen layouts"
  OMNIGIBSON_HEADLESS=1 $PY python step11_eval_vision.py --ckpt $CKPT --episodes 50 > data/eval_vision_${1}_$2.log 2>&1
  log "$1 demos, ckpt $2: $(grep -a FINAL data/eval_vision_${1}_$2.log)"
}

log "waiting for the 500-demo training to finish"
while ps -eo cmd | grep -v grep | grep -q -- "--output_dir=data/train/diffusion_500"; do sleep 60; done
log "500-demo training finished: $(grep -ao 'step:[0-9K]* smpl.*loss:[0-9.]*' data/train_500.log | tail -1)"
# wait for the separately-launched early eval of the 20k checkpoint, so two sims never run at once
while ps -eo cmd | grep -v grep | grep -q "step11_eval_vision.py"; do sleep 30; done
evaluate 500 040000
evaluate 500 last

NAME=franka_apple_wrist_all
log "converting all successful demos -> data/lerobot/$NAME"
$PY python step10_to_lerobot.py --name $NAME > data/convert_all.log 2>&1
log "$(tail -1 data/convert_all.log)"

OUT=data/train/diffusion_all
rm -rf $OUT
log "training diffusion policy on all demos for $STEPS steps -> $OUT"
$PY lerobot-train \
  --dataset.repo_id=local/$NAME --dataset.root=data/lerobot/$NAME \
  --policy.type=diffusion --policy.push_to_hub=false --policy.device=cuda \
  --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape="[120,120]" --policy.down_dims="[256,512,1024]" \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
  --steps=$STEPS --batch_size=64 --log_freq=500 --save_freq=10000 --num_workers=10 \
  --output_dir=$OUT --wandb.enable=false > data/train_all.log 2>&1
log "training done (exit $?): $(grep -ao 'step:[0-9K]* smpl.*loss:[0-9.]*' data/train_all.log | tail -1)"
evaluate all 020000
evaluate all 040000
evaluate all last
log "VISION PIPELINE DONE"
