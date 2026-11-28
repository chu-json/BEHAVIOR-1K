#!/usr/bin/env bash
# Continued fine-tune of the released pi0.5 radio checkpoint (steering/ft/mentor_ft.py train), GPU-protocol aware:
# waits while /home/behavior/ext/GPU_LOCK exists, holds /home/behavior/ext/GPU_TRAINING ("ft-cont") while training,
# removes it on exit. Extra args go to mentor_ft.py train.
#
#   systemd-run --user --unit ft-cont --property=MemoryHigh=24G -- systemd-inhibit --what=sleep:idle \
#       --why="ft-cont training" bash /home/behavior/BEHAVIOR-1K/steering/ft/ft_cont.sh --exp ft_cont ...
set -u
LOCK=/home/behavior/ext/GPU_LOCK
FLAG=/home/behavior/ext/GPU_TRAINING
LOG=${FT_LOG:-/home/behavior/ext/logs/ft_cont.log}
mkdir -p "$(dirname "$LOG")"
while [ -e "$LOCK" ]; do
    echo "$(date +%T) waiting for $LOCK" >> "$LOG"
    sleep 60
done
echo "ft-cont" > "$FLAG"
trap 'rm -f "$FLAG"' EXIT
echo "$(date '+%F %T') start: mentor_ft.py train $*" >> "$LOG"
cd /home/behavior/ext/mentor/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python -u /home/behavior/BEHAVIOR-1K/steering/ft/mentor_ft.py train \
    "$@" >> "$LOG" 2>&1
rc=$?
echo "$(date '+%F %T') exit code $rc" >> "$LOG"
exit $rc
