#!/usr/bin/env bash
# Wait for a sustained idle machine, run the trend sweep, and verify the machine
# STAYED idle throughout.
#
# Tail measurements are invalid on a busy machine (docs/decisions.md #10).
# Checking load only before starting is not enough: a run that began at load
# 0.96 and ended at 16.87 produced four failed gates that looked like a finding
# but were contamination. Load is now sampled throughout and the results are
# refused if it rose during the sweep.
set -uo pipefail
cd "$(dirname "$0")/.."

THRESHOLD=${1:-1.5}
SUSTAIN=${2:-90}
CEILING=$(awk -v t="$THRESHOLD" 'BEGIN{print t*2}')

load() { uptime | sed 's/.*load averages*: *//' | awk '{print $1}'; }
below() { awk -v a="$(load)" -v b="$THRESHOLD" 'BEGIN{exit !(a<b)}'; }

echo "[$(date +%H:%M:%S)] waiting for load < $THRESHOLD sustained ${SUSTAIN}s"
while true; do
    if below; then
        ok=1
        for _ in $(seq 1 $((SUSTAIN / 10))); do
            sleep 10; below || { ok=0; break; }
        done
        [ "$ok" = 1 ] && break
    fi
    sleep 20
done

START_LOAD=$(load)
echo "[$(date +%H:%M:%S)] idle (load $START_LOAD) - running sweep"

# Sample load in the background for the duration of the sweep.
PEAK_FILE=$(mktemp)
echo "$START_LOAD" > "$PEAK_FILE"
( while :; do
    cur=$(load)
    prev=$(cat "$PEAK_FILE")
    awk -v a="$cur" -v b="$prev" 'BEGIN{exit !(a>b)}' && echo "$cur" > "$PEAK_FILE"
    sleep 15
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

python3 -m dpt.bench.sweep --config experiments/noise_floor_trend.yaml 2>&1 | tail -2

kill $SAMPLER 2>/dev/null
PEAK_LOAD=$(cat "$PEAK_FILE"); END_LOAD=$(load); rm -f "$PEAK_FILE"

echo
echo "[$(date +%H:%M:%S)] load: start $START_LOAD  peak $PEAK_LOAD  end $END_LOAD"
if awk -v p="$PEAK_LOAD" -v c="$CEILING" 'BEGIN{exit !(p>c)}'; then
    echo
    echo "REFUSED: load reached $PEAK_LOAD during the sweep (ceiling $CEILING)."
    echo "Other work started while this ran, so the numbers measure that, not the"
    echo "apparatus. Results written but NOT to be interpreted. Re-run on a quiet"
    echo "machine."
    exit 2
fi

echo "[$(date +%H:%M:%S)] machine stayed quiet. gate results:"
for P in 2 4 5 6; do
    printf "  p=%s: " "$P"
    python3 -m dpt.analysis.noise \
        --critpath results/noise_floor_trend_critpath.csv \
        --require-usable-at $P --gate-beta 0.4 >/dev/null 2>&1 \
        && echo "PASSED" || echo "failed"
done
