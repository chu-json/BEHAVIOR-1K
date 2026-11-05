#!/usr/bin/env bash
# Unattended pipeline: wait until the GPU is free of other jobs, then run the fixed N=10 Comet batch.
# Run detached, holding a sleep inhibitor for its whole lifetime:
#   setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="BEHAVIOR Comet eval" --mode=block \
#       bash steering/overnight.sh > /home/behavior/ext/logs/overnight.log 2>&1 &
set -u
REPO=/home/behavior/BEHAVIOR-1K
EXT=/home/behavior/ext
LOGS=$EXT/logs
OUT=${OUT:-$EXT/runs/val10_full_fixed}
QUIET_MIN=${QUIET_MIN:-10}
cd "$REPO"
log() { echo "$(date '+%F %T') $*"; }

log "overnight start; waiting for ${QUIET_MIN} quiet GPU minutes"
bash steering/wait_for_gpu.sh "" "$QUIET_MIN" "$LOGS/wait_for_gpu.log" -- true

bash steering/gpu_watchdog.sh 1000 "$LOGS/gpu_watchdog.log" &
WATCHDOG=$!
trap 'kill $WATCHDOG $SERVER 2>/dev/null' EXIT

log "starting policy server"
(cd "$EXT/openpi-comet" && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 \
    exec .venv/bin/python "$REPO/steering/comet_server.py" \
    --ckpt "$EXT/ckpt/pi05-turning_on_radio-sft" --port 8000) > "$LOGS/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 1 120); do
    curl -s -m 2 localhost:8000/healthz > /dev/null && break
    kill -0 $SERVER 2> /dev/null || { log "server died during startup; see $LOGS/server.log"; exit 1; }
    sleep 5
done
log "server healthy (pid $SERVER)"

log "running batch -> $OUT"
OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/condabin/conda run --no-capture-output -n behavior \
    python steering/run_rollouts.py --seeds 0 1 2 3 4 5 6 7 8 9 --max-steps 4299 --resolution full --trace \
    --output-dir "$OUT" > "$LOGS/val10_full_fixed.log" 2>&1
log "batch exit $?"
[ -f "$OUT/rollouts.csv" ] && cut -d, -f3-8 "$OUT/rollouts.csv"
log "done; stopping server and watchdog"
