#!/bin/bash
# Unattended: wait for TARGET successful demos -> stop collector -> train -> eval policy -> eval expert.
# Everything logs to my_franka/data/; a one-line status per stage goes to data/pipeline.log.
TARGET=${TARGET:-200}
cd "$(dirname "$0")"
PY="$HOME/miniconda3/condabin/conda run --no-capture-output -n behavior python"
LOG=data/pipeline.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "waiting for $TARGET successful episodes"
while true; do
  ok=$(grep -a "^\[w0\] ep" data/collect_w0.log | tail -1 | sed -n 's/.*| ok \([0-9]*\)\/.*/\1/p')
  if [ -n "$ok" ] && [ "$ok" -ge "$TARGET" ]; then break; fi
  if ! pgrep -f "python .*step5_collect.py --worker-id 0" >/dev/null; then log "collector exited early (ok=$ok)"; break; fi
  sleep 2
done
# Kill right after an episode line was written (the file was just flushed), to avoid a half-written episode.
pkill -f "python .*step5_collect.py --worker-id 0"
sleep 10
log "collection stopped: $(grep -a '^\[w0\] ep' data/collect_w0.log | tail -1 | sed 's/.*| ok/ok/')"

log "training"
$PY step6_train.py > data/train.log 2>&1
log "training done: $(tail -1 data/train.log)"

log "evaluating policy on 50 unseen layouts"
OMNIGIBSON_HEADLESS=1 $PY step7_eval.py --episodes 50 > data/eval_policy.log 2>&1
log "$(grep -a 'FINAL' data/eval_policy.log)"

log "evaluating expert on the same 50 layouts"
OMNIGIBSON_HEADLESS=1 $PY step7_eval.py --episodes 50 --expert > data/eval_expert.log 2>&1
log "$(grep -a 'FINAL' data/eval_expert.log)"
log "PIPELINE DONE"
