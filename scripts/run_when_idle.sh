#!/usr/bin/env bash
# Wait for a sustained idle machine, then run the trend sweep and its gates.
#
# Tail measurements are invalid on a busy machine (docs/decisions.md #10), so
# this waits for load to stay low rather than firing on a momentary dip. It
# never touches other processes -- it only waits for them to finish.
set -uo pipefail
cd "$(dirname "$0")/.."

THRESHOLD=${1:-1.5}
SUSTAIN=${2:-90}     # seconds load must stay below threshold

load() { uptime | sed 's/.*load averages*: *//' | awk '{print $1}'; }
below() { awk -v a="$(load)" -v b="$THRESHOLD" 'BEGIN{exit !(a<b)}'; }

echo "[$(date +%H:%M:%S)] waiting for load < $THRESHOLD sustained ${SUSTAIN}s"
while true; do
    if below; then
        ok=1
        for _ in $(seq 1 $((SUSTAIN / 10))); do
            sleep 10
            below || { ok=0; break; }
        done
        [ "$ok" = 1 ] && break
    fi
    sleep 20
done

echo "[$(date +%H:%M:%S)] idle (load $(load)) - running trend sweep"
python3 -m dpt.bench.sweep --config experiments/noise_floor_trend.yaml 2>&1 | tail -2

echo
echo "[$(date +%H:%M:%S)] gate results:"
for P in 2 4 5 6; do
    printf "  p=%s: " "$P"
    if python3 -m dpt.analysis.noise \
         --critpath results/noise_floor_trend_critpath.csv \
         --require-usable-at $P --gate-beta 0.4 >/dev/null 2>&1; then
        echo "PASSED"
    else
        echo "failed"
    fi
done
echo "[$(date +%H:%M:%S)] done. load now $(load)"
