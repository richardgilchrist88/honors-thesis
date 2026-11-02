"""Seeded adaptation of rp4lp.sparse_gaussian and the reference notebook."""

import hashlib
import math
import numpy as np
from scipy import sparse
from .config import integer


def projection_dimension(epsilon, number_of_lp_variables):
    n = integer(number_of_lp_variables, "number_of_lp_variables", 2)
    if isinstance(epsilon, bool) or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between 0 and 1")
    try:
        k = int(round((1.0 / float(epsilon)) ** 2) * math.log(float(n)))
    except (OverflowError, ValueError) as exc:
        raise ValueError("epsilon produces an unrepresentable projection dimension") from exc
    if k < 1:
        raise ValueError("projection dimension must be positive")
    return k  # Deliberately no k <= m restriction.


def sparse_gaussian(A, k, seed):
    """Fixed-count support (authors' scipy.sparse.random semantics), scaled normals.

    Unlike the paper's Bernoulli description, the nonzero count is deterministic.
    This preserves the authors' construction and the reference notebook's sampler.
    """
    integer(k, "k")
    integer(seed, "projection_seed", 0)
    m, n = A.shape
    sigma = 0.5 * A.count_nonzero() / (m * n)
    if not 0 < sigma <= 1:
        raise ValueError("A must have positive density")
    rng = np.random.default_rng(seed)
    nonzeros = int(round(sigma * k * m))
    positions = rng.choice(k * m, size=nonzeros, replace=False)
    values = rng.normal(0.0, 1.0 / math.sqrt(k * sigma), nonzeros)
    T = sparse.csr_matrix((values, (positions // m, positions % m)), shape=(k, m))
    T.sort_indices()
    return T, sigma


def projection_fingerprint(T):
    """Portable digest for verifying paired Diet realizations, outside CPU timing."""
    digest = hashlib.sha256()
    for array in (np.asarray(T.shape, dtype="<i8"), T.indptr.astype("<i8"),
                  T.indices.astype("<i8"), T.data.astype("<f8")):
        digest.update(array.tobytes())
    return digest.hexdigest()
