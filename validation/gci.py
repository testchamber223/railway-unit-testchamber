"""Grid Convergence Index (GCI), three-grid procedure of Celik et al. (2008).

Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H., Raad,
P. E. (2008). "Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications", J. Fluids Eng. 130(7), 078001
(evidence E-116; page to confirm).

With cell sizes h1 < h2 < h3 (fine, medium, coarse) and solutions f1, f2, f3:

    r21 = h2 / h1,  r32 = h3 / h2
    e21 = f2 - f1,  e32 = f3 - f2,  s = sign(e32 / e21)
    p   = | ln|e32 / e21| + q(p) | / ln r21,   q(p) = ln((r21^p - s) / (r32^p - s))
          (fixed-point iteration; q = 0 for constant refinement ratio)
    f_ext   = (r21^p f1 - f2) / (r21^p - 1)
    e_a     = |(f1 - f2) / f1|,   e_ext = |(f_ext - f1) / f_ext|
    GCI_fine = Fs * e_a / (r21^p - 1),   Fs = 1.25 (E-116)

s < 0 means oscillatory convergence: it is flagged, and p is computed from
|e32 / e21| as in the paper, but the result must be treated with caution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from evidence import registry as _evidence
from evidence.registry import Registry

FS_ID = "E-116"


@dataclass(frozen=True)
class GCIResult:
    p: float                  # apparent order
    f_extrapolated: float
    e_a21: float              # approximate relative error, fine-medium
    e_ext21: float            # extrapolated relative error
    gci_fine: float           # relative (multiply by |f1| for an absolute band)
    r21: float
    r32: float
    oscillatory: bool
    converged: bool           # fixed-point iteration converged
    iterations: int

    @property
    def band(self) -> float:
        """Absolute uncertainty band on the fine-grid value."""
        return self.gci_fine * abs(self.f_fine)

    f_fine: float = math.nan


def representative_cell_size(cell_count: int, size: float, dimension: int = 3) -> float:
    """h = (size / N)^(1/dim), size = volume (3D) or area (2D) of the domain."""
    if cell_count <= 0 or size <= 0 or dimension not in (2, 3):
        raise ValueError("need cell_count > 0, size > 0, dimension 2 or 3")
    return (size / cell_count) ** (1.0 / dimension)


def gci(f1: float, f2: float, f3: float, h1: float, h2: float, h3: float,
        evidence: Registry | None = None, tol: float = 1e-12,
        max_iter: int = 500) -> GCIResult:
    """Three-grid GCI (Celik et al. 2008); f1 fine, f3 coarse, h1 < h2 < h3."""
    if not 0 < h1 < h2 < h3:
        raise ValueError("cell sizes must satisfy 0 < h1 < h2 < h3 (fine to coarse)")
    number = evidence.number if evidence is not None else _evidence.number
    Fs = number(FS_ID, "-")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        raise ValueError("identical solutions on two grids: the order cannot be estimated")
    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0
    oscillatory = s < 0
    ln_ratio = math.log(abs(ratio))

    p = abs(ln_ratio) / math.log(r21)          # start: constant-ratio formula
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        num, den = r21 ** p - s, r32 ** p - s
        if num <= 0 or den <= 0:
            break
        p_new = abs(ln_ratio + math.log(num / den)) / math.log(r21)
        if abs(p_new - p) <= tol * max(1.0, p):
            p, converged = p_new, True
            break
        p = p_new
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    e_a = abs((f1 - f2) / f1)
    e_ext = abs((f_ext - f1) / f_ext)
    return GCIResult(p=p, f_extrapolated=f_ext, e_a21=e_a, e_ext21=e_ext,
                     gci_fine=Fs * e_a / (r21 ** p - 1), r21=r21, r32=r32,
                     oscillatory=oscillatory, converged=converged, iterations=it, f_fine=f1)
