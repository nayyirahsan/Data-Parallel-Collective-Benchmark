"""Collective algorithms, implemented from scratch over a p2p transport."""
from .param_server import ps_allreduce
from .recursive import is_supported as recursive_supported
from .recursive import recursive_allreduce, recursive_allreduce_fresh
from .reference import reference_allreduce
from .ring import ring_allreduce

ALGORITHMS = {
    "ring": ring_allreduce,
    "ps": ps_allreduce,
    "recursive": recursive_allreduce,
    "recursive_fresh": recursive_allreduce_fresh,
    "reference": reference_allreduce,
}

__all__ = [
    "ALGORITHMS",
    "ring_allreduce",
    "ps_allreduce",
    "recursive_allreduce",
    "reference_allreduce",
    "recursive_supported",
]
