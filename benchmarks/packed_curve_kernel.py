"""Compatibility wrappers for the packed-kernel performance experiment."""
from jaxpropka.model import pack_interaction_edges, packed_curve_kernel


def pack_edges(arrays):
    return pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
