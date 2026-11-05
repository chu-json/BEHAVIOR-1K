#!/usr/bin/env bash
# Unattended experiment campaign: run several rollout batches back to back against one policy server.
# Each line of the RUNS array is "<output dir name>|<extra run_rollouts.py args>". Detach with:
#   setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="BEHAVIOR campaign" --mode=block \
#       bash steering/campaign.sh > /home/behavior/ext/logs/campaign.log 2>&1 &
set -u
REPO=/home/behavior/BEHAVIOR-1K
EXT=/home/behavior/ext
LOGS=$EXT/logs
cd "$REPO"
log() { echo "$(date '+%F %T') $*"; }

COMMON="--max-steps 3000 --resolution full --trace"
RUNS=(
    "massA_default_s0|--base-mass default --seeds 0 1 2 3 4 5 6 7 8 9"
    "massB_eval_s10|--base-mass eval --seeds 10 11 12 13 14 15 16 17 18 19"
    "massC_default_s10|--base-mass default --seeds 10 11 12 13 14 15 16 17 18 19"
)

bash steering/gpu_watchdog.sh 1000 "$LOGS/gpu_watchdog.log" &
WATCHDOG=$!
SERVER=""
trap 'kill $WATCHDOG ${SERVER} 2>/dev/null' EXIT

if ! curl -s -m 2 localhost:8000/healthz > /dev/null; then
    log "starting policy server"
    (cd "$EXT/openpi-comet" && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 \
        exec .venv/bin/python "$REPO/steering/comet_server.py" \
        --ckpt "$EXT/ckpt/pi05-turning_on_radio-sft" --port 8000) > "$LOGS/server_campaign.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 120); do
        curl -s -m 2 localhost:8000/healthz > /dev/null && break
        sleep 5
    done
fi
log "server healthy"

for spec in "${RUNS[@]}"; do
    name=${spec%%|*}
    extra=${spec#*|}
    out=$EXT/runs/$name
    log "=== $name: $extra"
    # shellcheck disable=SC2086
    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/condabin/conda run --no-capture-output -n behavior \
        python steering/run_rollouts.py $COMMON $extra --output-dir "$out" > "$LOGS/$name.log" 2>&1
    log "$name exit $?: $(awk -F, 'NR>1{n++; s+=$5} END{print s"/"n" success"}' "$out/rollouts.csv" 2>/dev/null)"
done
log "campaign done"
