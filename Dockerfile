# Linux environment for the tc netem validation, which needs a real kernel
# network stack and NET_ADMIN. CPU-only torch keeps the image ~1 GB instead of
# the ~6 GB a CUDA build would pull in; this experiment does no GPU work.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir numpy pyyaml

WORKDIR /work
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . --no-deps

ENTRYPOINT ["python", "-m", "dpt.bench.netem"]
