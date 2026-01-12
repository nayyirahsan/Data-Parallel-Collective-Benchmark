#!/usr/bin/env bash
# Report whether this machine can support the H2 measurement, before spending
# time or credits on it.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== machine ==="
python3 - <<'PY'
import os, platform
print(f"  platform : {platform.system()} {platform.machine()}")
print(f"  cores    : {os.cpu_count()}")
try:
    print(f"  load     : {os.getloadavg()[0]:.2f}")
except OSError:
    pass
PY

echo
echo "=== timer ==="
python3 - <<'PY'
from dpt.timing import sleep_granularity_ns, SLEEP_SAFETY_MARGIN, SLEEP_FRACTION
import os

g = sleep_granularity_ns()
engages = g * SLEEP_SAFETY_MARGIN / SLEEP_FRACTION
cores = os.cpu_count() or 1
yielding = engages <= 200_000  # the alpha the experiments use

print(f"  sleep granularity : {g/1000:.1f} us")
print(f"  yielding above    : {engages/1000:.0f} us delay")
print("  " + ("fine timer: core-yielding ACTIVE at alpha=200us"
              if yielding else
              "coarse timer: core-yielding INACTIVE, each worker spins a full core"))
print()

# Calibrated against measured gate outcomes on the reference machine
# (M4 Pro, 12 cores, fine timer):
#   pure spin     -> p=4 FAILED, p=2 passed   => about cores/6
#   core-yielding -> p=4 PASSED, p=8 FAILED   => about cores/3
# These are anchors from observed gate results, not a model of contention. An
# earlier version divided by spin CPU share and predicted "p up to 19" on the
# very machine where p=8 had been measured to fail.
divisor = 3 if yielding else 6
budget = max(2, cores // divisor)
print(f"  Extrapolating from measured gate outcomes (cores/{divisor}):")
print(f"  => expect p up to ~{budget} on {cores} cores")
print()
if budget >= 8:
    print("     p=8 looks reachable. The gate decides, not this estimate.")
else:
    print(f"     p=8 likely out of reach; p={budget} is the gated fallback.")
    print(f"     p=8 would want ~{8 * divisor} cores on a machine with this timer.")
PY

echo
echo "=== next ==="
echo "  ./scripts/settle_h2.sh 8     # runs the gate, refuses if unsupported"
echo "  ./scripts/settle_h2.sh 4     # the configuration that passes on 12 cores"
