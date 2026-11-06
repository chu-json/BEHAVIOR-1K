#!/usr/bin/env bash
# Align-assist causal test (steering/align_assist.py). Unattended experiment campaign: run several rollout batches back to back against one policy server.
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
    "assist_right|--base-mass default --align-assist --instances 211 197 214 139 211 197 214 139 211 197 214 139 --seeds 2 6 8 9 12 16 18 19 22 26 28 29"
    "assist_left_control|--base-mass default --align-assist --instances 242 295 203 109 181 187 --seeds 0 1 3 4 5 7"
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
