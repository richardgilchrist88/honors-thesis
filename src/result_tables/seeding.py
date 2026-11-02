"""Stable experiment identities, independent of loop order or sweep length."""

import hashlib
import json
from . import config


def experiment_seed(master_seed, problem_type, row_size, *, epsilon=None, replication=None):
    """Return a uint64 seed for an instance or a particular projection.

    Diet variants deliberately omit variant from the key to share the same T.
    Float.hex encodes epsilon exactly; Python's process-randomized hash is not used.
    """
    key = [config.SEED_POLICY, int(master_seed), problem_type, int(row_size)]
    if epsilon is None and replication is None:
        key += ["instance"]
    elif epsilon is not None and replication is not None:
        key += ["projection", float(epsilon).hex(), int(replication)]
    else:
        raise ValueError("epsilon and replication must both be supplied for a projection")
    encoded = json.dumps(key, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
