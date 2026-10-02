"""Frozen-structure sequence-differentiable titration.

This is a PROPKA-3.0-parameterized mean-field surrogate, not a numerically
identical reimplementation of PROPKA's coupled determinant algorithm.
"""
from .cache import ResidueKey, StructureCache
from .model import TitrationModel, CurveResult, PkaResult, GridPkaResult, one_hot
from .parameters import ALPHABET, GROUPS, ModelConfig
from .differentiation import DifferentiationConfig, EquilibriumConfig, SolverDiagnostics
from .audit import EquilibriumAudit, audit_equilibrium
from .selectivity import SelectivityObjective, SelectivityResult, SelectivityGradient

__version__ = "0.1.0"


def prepare(path_or_atoms, *, topology_options=None, geometry_options=None,
            kernel_options=None):
    """Use Biotite once, outside JIT, to construct a reusable structural cache."""
    from .topology import load_topology
    from .geometry import build_candidates
    from .precompute import build_cache
    topology = load_topology(path_or_atoms, **(topology_options or {}))
    candidates = build_candidates(topology, **(geometry_options or {}))
    return build_cache(topology,candidates,**(kernel_options or {}))
