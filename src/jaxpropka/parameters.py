"""Numerical parameters and conventions.

PROPKA 3.0 / Nov30 numerical values are from upstream parameters_new.py and
version.py. The mean-field H-bond state model is OUR surrogate, not PROPKA's
iterative determinant assignment. See docs/MODEL.md and SOURCES.md.
"""
from dataclasses import dataclass
import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREE = ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
         "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR")
AA_TO_INDEX = {x: i for i, x in enumerate(ALPHABET)}
THREE_TO_INDEX = {x: i for i, x in enumerate(THREE)}
GROUPS = ("ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTERM", "CTERM")
GROUP_AA = np.array([AA_TO_INDEX[x] for x in "DEHCYKR"], dtype=np.int32)
MODEL_PKA = np.array([3.8, 4.5, 6.5, 9.0, 10.0, 10.5, 12.5, 8.0, 3.2])
Q_DEPROT = np.array([-1., -1., 0., -1., -1., 0., 0., 0., -1.])
FORMAL_CHARGE = 2 * Q_DEPROT + 1  # acid -1, base +1
CLASSES = ("COO", "COO", "HIS", "CYS", "TYR", "LYS", "ARG", "N+", "COO")
AA_CLASSES = (None, "CYS", "COO", "COO", None, None, "HIS", None, "LYS", None,
              None, "AMD", None, "AMD", "ARG", "ROH", "ROH", None, "TRP", "TYR")
CENTERS = {
    "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "CYS": ("SG",), "TYR": ("OH",), "LYS": ("NZ",),
    "ARG": ("CZ",),
}
DONORS = {
    "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"), "CYS": ("SG",), "TYR": ("OH",),
    "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"),
    "ASN": ("ND2",), "GLN": ("NE2",), "SER": ("OG",), "THR": ("OG1",),
    "TRP": ("NE1",),
}
ACCEPTORS = {
    "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"), "CYS": ("SG",), "TYR": ("OH",),
    "ASN": ("OD1",), "GLN": ("OE1",), "SER": ("OG",), "THR": ("OG1",),
}
NEUTRAL_AA = [AA_TO_INDEX[x] for x in "NQSTW"]

# Symmetric distance ranges. Nov30 overrides nonzero amplitudes to 0.85.
_HB = {}
for _a, _values in {
    "COO": {"COO": (2.5,3.5), "CYS": (3.,4.), "TYR": (2.65,3.65),
            "HIS": (2.,3.), "N+": (2.85,3.85), "LYS": (2.85,3.85),
            "ARG": (1.85,2.85), "ROH": (2.65,3.65), "AMD": (2.,3.), "TRP": (2.,3.)},
    "CYS": {"CYS": (3.,5.), "TYR": (3.5,4.5), "HIS": (3.,4.),
            "N+": (3.,4.5), "LYS": (3.,4.), "ARG": (2.5,4.),
            "ROH": (3.5,4.5), "AMD": (2.5,3.5), "TRP": (2.5,3.5)},
    "TYR": {"TYR": (3.5,4.5), "HIS": (2.,3.), "N+": (3.,4.5),
            "LYS": (3.,4.), "ARG": (2.5,4.), "ROH": (3.5,4.5),
            "AMD": (2.5,3.5), "TRP": (2.5,3.5)},
    "HIS": {"AMD": (2.,3.)},
}.items():
    for _b, _v in _values.items():
        _HB[_a, _b] = _HB[_b, _a] = _v
BB_RANGES = np.array([(2.,3.), (2.,3.), (2.,3.), (3.,4.), (2.2,3.2),
                      (2.8,3.8), (2.,3.), (2.8,3.8), (2.,3.)])


def hbond_range(a, b):
    return _HB.get((a, b))


def atom_volume(name: str, element: str) -> float:
    if name in ("C", "CA"):
        return 1.4
    return {"N": 1.06, "O": 1., "S": 1.66}.get(element, 2.64)


@dataclass(frozen=True)
class ModelConfig:
    steps: int = 64
    damping: float = 0.35
    gate_width: float = 20.0  # smooth approximation of burial eligibility
    nmin: float = 280.0
    nmax: float = 560.0
    desolv_prefactor: float = -13.0
    surface_scale: float = 0.25
    dielectric_buried: float = 30.0
    dielectric_surface: float = 160.0
    coulomb_scale: float = 1.0
    hbond_scale: float = 1.0
    desolv_scale: float = 1.0
    root_steps: int = 28
    root_batch_size: int = 4
    ph_min: float = -10.0
    ph_max: float = 24.0
    residual_tolerance: float = 2e-5
    root_tolerance: float = 2e-5
    slope_min: float = 1e-5

    def __post_init__(self):
        for name in ("steps", "root_steps", "root_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive Python int")
        for name, value in vars(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.damping <= 1 or self.nmax <= self.nmin:
            raise ValueError("invalid damping or burial bounds")
        if self.ph_max <= self.ph_min or self.gate_width < 0:
            raise ValueError("invalid pH bounds or gate width")
        if min(self.dielectric_buried, self.dielectric_surface) <= 0:
            raise ValueError("dielectrics must be positive")
        if min(self.coulomb_scale, self.hbond_scale, self.desolv_scale) < 0:
            raise ValueError("scales must be nonnegative")
        if min(self.residual_tolerance, self.root_tolerance, self.slope_min) <= 0:
            raise ValueError("numerical tolerances must be positive")
