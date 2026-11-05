#!/usr/bin/env bash
# Block until no GPU compute process other than the allowed PIDs has existed for QUIET_MIN minutes, then exec "$@".
#
#   bash steering/wait_for_gpu.sh "<allowed pid list>" <quiet_min> <log> -- <command...>
set -u
ALLOWED=" $1 "
QUIET_MIN=$2
LOG=$3
shift 3
[ "${1:-}" = "--" ] && shift
INTERVAL=30
need=$((QUIET_MIN * 60 / INTERVAL))
quiet=0
echo "$(date +%T) waiting for ${QUIET_MIN} quiet minutes (allowed: ${ALLOWED})" >> "$LOG"
while [ "$quiet" -lt "$need" ]; do
    others=""
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' '); do
        case "$ALLOWED" in *" $pid "*) ;; *) others="$others $pid" ;; esac
    done
    if [ -n "$others" ]; then
        [ "$quiet" -gt 0 ] && echo "$(date +%T) GPU busy again (pids:${others}); resetting" >> "$LOG"
        quiet=0
    else
        quiet=$((quiet + 1))
    fi
    sleep "$INTERVAL"
done
echo "$(date +%T) GPU quiet for ${QUIET_MIN} min; starting: $*" >> "$LOG"
exec "$@"
