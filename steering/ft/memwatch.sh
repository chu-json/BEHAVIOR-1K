#!/usr/bin/env bash
# Log host memory (unit cgroup + system available) and GPU memory every 2 s: memwatch.sh <unit> <out>
while true; do
  c=$(systemctl --user show -p MemoryCurrent --value "$1" 2>/dev/null)
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$(date +%T) unit_MiB=$(( ${c:-0} / 1048576 )) avail_MiB=$a gpu_MiB=$g"
  sleep 2
done >> "$2"
