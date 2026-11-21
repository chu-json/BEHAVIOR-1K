#!/usr/bin/env bash
# Speed/memory probe: bash probe.sh <variant> <batch> [steps]  -> /home/behavior/ext/logs/ft_probe_<variant>_bs<batch>.log
V=$1; B=$2; N=${3:-30}; W=${4:-1}
EXT=/home/behavior/ext; P=$EXT/logs/ft_probe_${V}_bs$B.log
cd $EXT/openpi-comet
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 timeout 2400 .venv/bin/python -u /home/behavior/BEHAVIOR-1K/steering/ft/train_radio.py \
    --exp probe_${V}_bs$B --variant $V --batch-size $B --steps $N --log-interval 5 --save-interval 100000 --workers $W --overwrite 2>&1 \
    | while IFS= read -r line; do echo "$(date +%s.%N) $line"; done > $P
grep "Step \|peak_bytes\|RESOURCE_EXHAUSTED" $P | cut -c1-160
rm -rf $EXT/ckpt/ft/probe_${V}_bs$B
