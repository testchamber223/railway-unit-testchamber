"""Verification of validation/gci.py (Celik et al. 2008) with MANUFACTURED sequences.

f(h) = f0 + C h^q are manufactured solutions (verification-only), not CFD results.
"""

import math

import pytest

from validation.gci import gci, representative_cell_size

F0, C = 3.0, 5.0


def seq(q, hs):
    return [F0 + C * h ** q for h in hs]


@pytest.mark.parametrize("hs", [(0.01, 0.02, 0.04), (1.0, 2.0, 4.0)])
def test_second_order_sequence(hs):
    r = gci(*seq(2, hs), *hs)
    assert r.p == pytest.approx(2.0, abs=1e-6)
    assert r.f_extrapolated == pytest.approx(F0, rel=1e-9)
    assert not r.oscillatory and r.converged


def test_first_order_sequence():
    hs = (0.01, 0.02, 0.04)
    r = gci(*seq(1, hs), *hs)
    assert r.p == pytest.approx(1.0, abs=1e-6)
    assert r.f_extrapolated == pytest.approx(F0, rel=1e-9)


def test_non_constant_refinement_ratio_needs_iteration():
    hs = (1.0, 1.3, 2.0)                       # r21 = 1.3, r32 = 1.54
    r = gci(*seq(2, hs), *hs)
    assert r.iterations > 1
    assert r.p == pytest.approx(2.0, abs=1e-6)
    assert r.f_extrapolated == pytest.approx(F0, rel=1e-9)


def test_gci_formula_uses_safety_factor_e116():
    hs = (0.01, 0.02, 0.04)
    f1, f2, _ = seq(2, hs)
    r = gci(*seq(2, hs), *hs)
    e_a = abs((f1 - f2) / f1)
    assert r.gci_fine == pytest.approx(1.25 * e_a / (2.0 ** 2 - 1), rel=1e-9)
    assert r.band == pytest.approx(r.gci_fine * abs(f1))


def test_oscillating_sequence_is_flagged():
    r = gci(1.00, 1.10, 0.95, 1.0, 2.0, 4.0)
    assert r.oscillatory


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gci(1, 2, 3, 0.04, 0.02, 0.01)         # coarse first
    with pytest.raises(ValueError):
        gci(1, 1, 2, 1, 2, 4)                  # identical solutions


def test_representative_cell_size():
    assert representative_cell_size(1000, 1.0, 3) == pytest.approx(0.1)
    assert representative_cell_size(100, 1.0, 2) == pytest.approx(0.1)
    assert representative_cell_size(500_000, 137.5) == pytest.approx(
        (137.5 / 5e5) ** (1 / 3))
    assert math.isclose(representative_cell_size(8, 8.0), 1.0)
