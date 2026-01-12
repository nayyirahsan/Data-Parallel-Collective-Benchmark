#!/usr/bin/env bash
# Run the tc netem validation in a Linux container.
#
# Requires Docker to be running (on macOS: open -a Docker). NET_ADMIN is needed
# because the experiment attaches a qdisc to the loopback interface; it is
# scoped to the container's own network namespace and does not touch the host's.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "building image..."
docker build -q -t dpt-netem -f Dockerfile . >/dev/null

mkdir -p results
echo "running validation (a few minutes)..."
docker run --rm \
    --cap-add=NET_ADMIN \
    -v "$PWD/results:/work/results" \
    dpt-netem "$@"

echo
echo "result: results/netem_validation.json"
