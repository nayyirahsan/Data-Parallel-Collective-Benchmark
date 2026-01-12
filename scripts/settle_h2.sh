#!/usr/bin/env bash
# Settle H2 on a machine with enough cores.
#
# H2's blocker on the 12-core development machine was not the absence of a real
# network -- it was that p spin-waiting workers need cores that are not also
# running the workers. At p=8 on 12 cores the scheduler starves and the
# apparatus's own p99/p50 reaches 2-10x, swamping any injected effect.
#
# This script re-establishes the measurement gates on the new machine, and only
# runs the experiment if they pass. Running H2 on a machine that fails the gate
# measures the machine, not the hypothesis.
set -euo pipefail
cd "$(dirname "$0")/.."

P="${1:-8}"
CORES=$(python3 -c "import os; print(os.cpu_count())")
echo "target world size: p=$P    cores available: $CORES"
if [ "$CORES" -lt $((P * 3)) ]; then
    echo "WARNING: fewer than 3x cores per worker. The gate will likely fail."
    echo "         Rule of thumb from the development machine: p=8 needs ~24-32 cores."
fi

echo
echo "=== 1/4  timer fidelity under this core count ==="
python3 -m dpt.bench.calibrate --workers 1 "$P" --targets-ns 50000 200000 1000000

echo
echo "=== 2/4  apparatus tail-noise floor ==="
python3 -m dpt.bench.sweep --config experiments/noise_floor.yaml

echo
echo "=== 3/4  gate: can this machine support a p99 claim at p=$P? ==="
# Scoped to the operating point experiments/h2_jitter.yaml actually uses:
# beta=0.4 and messages of 1 MB and up.
if ! python3 -m dpt.analysis.noise --require-usable-at "$P" \
        --gate-beta 0.4 --gate-min-bytes 1048576; then
    echo
    echo "Stopping. The apparatus cannot support the measurement at p=$P."
    echo "Either use a machine with more cores, or lower p."
    exit 1
fi

echo
echo "=== 4/4  H2 jitter experiment ==="
python3 -m dpt.bench.sweep --config experiments/h2_jitter.yaml
python3 -m dpt.analysis.jitter_impact
echo
echo "=== simulator validation against these measurements ==="
python3 -m dpt.analysis.validate

echo
echo "Done. The ring/ps column in the jitter_impact output is the H2 answer."
echo "With the gate passed at p=$P, that number is a measurement rather than"
echo "a model prediction -- update docs/findings-h2.md accordingly."
