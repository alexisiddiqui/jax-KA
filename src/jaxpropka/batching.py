"""Host-side padding for shared, fixed-shape multi-structure evaluation."""
from __future__ import annotations

import numpy as np

from .cache import StructureCache


_ENV_FIELDS = frozenset({"env_neighbors", "env_mask", "volume", "mass", "hbond"})
_PAIR_FIELDS = frozenset({"neighbors", "pair_mask", "coulomb_geometry", "hb_donor", "hb_reverse"})


def pack_inputs(cache: StructureCache, probabilities, capacities=None, *,
                bucket_multiple=(64, 16, 16)):
    """Return (array dict, padded probabilities, original N), all on the host.

    Capacities are (N, Ke, Kc). By default each dimension is rounded upward to
    its bucket_multiple; explicit capacities must fit every existing slot.
    No neighbors are truncated. Labels/metadata remain in the original cache.
    Structural floats use the probability dtype and indices use int32, so
    input representation does not create extra specializations within a bucket.
    """
    cache.validate()
    p = np.asarray(probabilities)
    n = cache.n_residues
    if p.shape != (n, 20) or p.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"expected float32/float64 P[{n},20]")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-5):
        raise ValueError("P must be finite, nonnegative and sum to one in each row")
    needed = (n, cache.env_neighbors.shape[1], cache.neighbors.shape[1])
    sizes = bucket_multiple if capacities is None else capacities
    if len(sizes) != 3 or any(isinstance(x, (bool, np.bool_)) or
                             not isinstance(x, (int, np.integer)) or x < 1 for x in sizes):
        raise ValueError("capacities/bucket_multiple must contain three positive integers")
    if capacities is None:
        capacities = tuple(((size + multiple - 1) // multiple) * multiple
                           for size, multiple in zip(needed, sizes))
    if any(size > capacity for size, capacity in zip(needed, capacities)):
        raise ValueError(f"bucket overflow: need {needed}, requested {tuple(capacities)}; truncation is forbidden")
    padded_n, ke, kc = map(int, capacities)
    arrays = {}
    for name, value in vars(cache).items():
        if not isinstance(value, np.ndarray) or name == "chain_index":
            continue
        shape = list(value.shape)
        shape[0] = padded_n
        if name in _ENV_FIELDS:
            shape[1] = ke
        elif name in _PAIR_FIELDS:
            shape[1] = kc
        dtype = value.dtype
        if np.issubdtype(dtype, np.floating):
            dtype = p.dtype
        elif np.issubdtype(dtype, np.integer):
            dtype = np.int32
        padded = np.zeros(shape, dtype=dtype)
        padded[tuple(slice(0, size) for size in value.shape)] = value
        arrays[name] = padded
    padded_p = np.zeros((padded_n, 20), dtype=p.dtype)
    padded_p[:, 0] = 1
    padded_p[:n] = p
    return arrays, padded_p, n
