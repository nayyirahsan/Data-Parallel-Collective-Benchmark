#!/usr/bin/env bash
# Wait for a genuinely idle machine, run the trend sweep, and verify it STAYED
# idle -- measuring only CPU consumed by *other* processes.
#
# Two earlier versions of this guard were wrong:
#   1. Checked load only before starting. A run that began at load 0.96 and
#      ended at 16.87 produced four failed gates that looked like a finding but
#      were contamination.
#   2. Sampled total load average during the run. That counts this script's own
#      workers -- p=6 spinning workers raise load by ~3 on their own -- so the
#      guard refused its own benchmark and could not distinguish that from real
#      interference.
# Foreign CPU (everything outside our process group) attributes correctly.
set -uo pipefail
cd "$(dirname "$0")/.."

THRESHOLD=${1:-50}    # foreign CPU %, i.e. half a core
SUSTAIN=${2:-90}
CEILING=${3:-120}     # refuse if foreign CPU exceeds ~1.2 cores mid-run

MYPG=$(ps -o pgid= -p $$ | tr -d ' ')
foreign() { ps -A -o pgid=,pcpu= 2>/dev/null | awk -v m="$MYPG" '$1!=m{s+=$2} END{printf "%.1f", s+0}'; }
below() { awk -v a="$(foreign)" -v b="$THRESHOLD" 'BEGIN{exit !(a<b)}'; }

echo "[$(date +%H:%M:%S)] waiting for foreign CPU < ${THRESHOLD}% sustained ${SUSTAIN}s"
echo "               (currently $(foreign)%)"
while true; do
    if below; then
        ok=1
        for _ in $(seq 1 $((SUSTAIN / 10))); do sleep 10; below || { ok=0; break; }; done
        [ "$ok" = 1 ] && break
    fi
    sleep 20
done

START=$(foreign)
echo "[$(date +%H:%M:%S)] idle (foreign CPU ${START}%) - running sweep"

PEAK_FILE=$(mktemp); echo "$START" > "$PEAK_FILE"
( while :; do
    c=$(foreign); p=$(cat "$PEAK_FILE")
    awk -v a="$c" -v b="$p" 'BEGIN{exit !(a>b)}' && echo "$c" > "$PEAK_FILE"
    sleep 10
  done ) 2>/dev/null & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

python3 -m dpt.bench.sweep --config experiments/noise_floor_trend.yaml 2>&1 | tail -2

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
PEAK=$(cat "$PEAK_FILE"); END=$(foreign); rm -f "$PEAK_FILE"

echo
echo "[$(date +%H:%M:%S)] foreign CPU: start ${START}%  peak ${PEAK}%  end ${END}%"
if awk -v p="$PEAK" -v c="$CEILING" 'BEGIN{exit !(p>c)}'; then
    echo
    echo "REFUSED: other processes reached ${PEAK}% CPU during the sweep (ceiling ${CEILING}%)."
    echo "Results written but NOT to be interpreted."
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
