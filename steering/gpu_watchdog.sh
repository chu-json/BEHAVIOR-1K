#!/usr/bin/env bash
# Protect another GPU job: if free GPU memory drops below a threshold, kill steering/run_rollouts.py.
# Also logs total and per-process GPU usage every few seconds, for measuring peaks.
#
#   bash steering/gpu_watchdog.sh <min_free_mib> <log_file>
# Never touches processes other than run_rollouts.py (the Isaac Sim client).
set -u
MIN_FREE_MIB=${1:-700}
LOG=${2:-gpu_watchdog.log}
INTERVAL=${INTERVAL:-2}
echo "$(date +%T) watchdog start: min_free=${MIN_FREE_MIB}MiB" >> "$LOG"
while true; do
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
    free=$((total - used))
    procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' ')
    echo "$(date +%T) used=${used} free=${free} procs=[${procs}]" >> "$LOG"
    if [ "$free" -lt "$MIN_FREE_MIB" ]; then
        pids=$(pgrep -f "python steering/run_rollouts.py")
        if [ -n "$pids" ]; then
            echo "$(date +%T) FREE ${free}MiB < ${MIN_FREE_MIB}MiB: killing run_rollouts.py pids ${pids}" | tee -a "$LOG"
            kill $pids
        fi
    fi
    sleep "$INTERVAL"
done
