"""Verification of the multi-zone coach (core.coach.MultiZoneCoach).

VERIFICATION-ONLY PARAMETERS: coach A = 300 m2, k = 1.2 W/m2K, C = 5e6 J/K,
V = 150 m3; verification unit card; 80 passengers at 70 / 50 W. Not real data.
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from core import loads, psychro
from core.coach import (FixedSupply, LoadTerm, MultiZoneCoach, SingleZoneCoach, UnitOperation,
                        simulate, simulate_multizone)
from core.unit import DXUnit, load_unit_card
from evidence.registry import MissingEvidenceError, Registry
from tests.conftest import DATA

BASE = SingleZoneCoach("verification", 300.0, 1.2, 5e6, air_volume_m3=150.0)
UNIT = DXUnit.from_card(load_unit_card(DATA / "verification_unit.yaml"))
RH_OUT = 0.25
W0 = psychro.humidity_ratio(45.0, RH_OUT)
PAX = LoadTerm("passengers", lambda t, T, W: loads.passengers(80, 70.0, 50.0, T_in=T))
OP = UnitOperation(UNIT, "cooling", 1.0, RH_OUT)
UNEVEN = dict(supply_fraction=(0.5, 0.3, 0.2), return_weight=(0.0, 0.0, 1.0),
              solar_share=(0.2, 0.3, 0.5), passenger_share=(0.4, 0.4, 0.2))


def test_large_mixing_reproduces_the_single_zone_model():
    sz = simulate(BASE, 45.0, 45.0, 0.0, 0.0, 4 * 3600.0, 60.0, W0=W0, loads=[PAX], unit=OP)
    mz = MultiZoneCoach(BASE, G_mix_W_K=1e7, **UNEVEN)
    md = simulate_multizone(mz, 45.0, 45.0, 0.0, 0.0, 4 * 3600.0, 60.0, W0=W0,
                            loads=[(PAX, "passengers")], unit=OP)
    for z in mz.zone_names:
        assert np.max(np.abs(md[f"T_C_{z}"] - sz["T_C"])) < 0.01
    assert np.max(np.abs(md["W_mean"] - sz["W_kg_kg"])) < 1e-5
    assert md["T_C_front"].iloc[-1] < 30                      # it is a pull-down


def test_large_mixing_reproduces_single_zone_without_humidity():
    sz = simulate(BASE, 30.0, 45.0, 2000.0, 12000.0, 6 * 3600.0, 60.0)
    mz = MultiZoneCoach(BASE, G_mix_W_K=1e7, supply_fraction=(0.6, 0.3, 0.1))
    md = simulate_multizone(mz, 30.0, 45.0, 2000.0, 12000.0, 6 * 3600.0, 60.0)
    assert np.max(np.abs(md["T_mean"] - sz["T_C"])) < 0.01


def test_total_enthalpy_closes_with_advection_and_mixing():
    mz = MultiZoneCoach(BASE, G_mix_W_K=150.0, **UNEVEN)
    md = simulate_multizone(mz, 45.0, 45.0, 0.0, 0.0, 3 * 3600.0, 30.0, W0=W0,
                            loads=[(PAX, "passengers")], unit=OP)
    t = md["t_s"].to_numpy()
    inflow = simpson((md["Q_env_W"] + md["Q_load_passengers_W"] + 80 * 50.0
                      + md["supply_Q_total_W"]).to_numpy(), x=t)
    stored = 0.0
    for i, z in enumerate(mz.zone_names):
        T, W = md[f"T_C_{z}"].to_numpy(), md[f"W_{z}"].to_numpy()
        h_g = np.array([loads.vapour_enthalpy(x) for x in T])
        stored += md.attrs["C_zone"][i] * (T[-1] - T[0]) + md.attrs["M_zone"][i] * np.sum(
            0.5 * (h_g[1:] + h_g[:-1]) * np.diff(W))
    assert md["spread"].iloc[-1] > 0.5                        # zones really differ
    assert inflow == pytest.approx(stored, rel=1e-3)


def test_steady_state_matches_the_long_dynamic_run():
    W_s = psychro.humidity_ratio(14.0, 0.8)
    mz = MultiZoneCoach(BASE, G_mix_W_K=250.0, **UNEVEN)
    sup = FixedSupply(1.6, 14.0, W_s)
    gains = np.array([1500.0, 1000.0, 2500.0])
    pax_like = [LoadTerm("gain", lambda t, T, W: (gains.sum(), 0.0))]
    # gains split by passenger share (0.4, 0.4, 0.2) -> use matching steady gains
    steady = mz.steady_state(40.0, sup, np.array(mz.passenger_share) * gains.sum())
    md = simulate_multizone(mz, 25.0, 40.0, 0.0, 0.0, 40 * 3600.0, 600.0, W0=W_s,
                            loads=[(pax_like[0], "passengers")], supply=sup)
    final = md[[f"T_C_{z}" for z in mz.zone_names]].iloc[-1].to_numpy()
    assert np.max(np.abs(final - steady)) < 1e-3


def test_reported_means_and_spread():
    mz = MultiZoneCoach(BASE, G_mix_W_K=100.0, measuring_points=(0.1, 0.5, 0.9), **UNEVEN)
    md = simulate_multizone(mz, 45.0, 45.0, 0.0, 0.0, 3600.0, 600.0, W0=W0,
                            loads=[(PAX, "passengers")], unit=OP)
    e = md.iloc[-1]
    T = np.array([e[f"T_C_{z}"] for z in mz.zone_names])
    assert e["uic_mean"] == pytest.approx(T.mean())              # one point per zone here
    assert e["spread"] == pytest.approx(T.max() - T.min())
    assert e["front_to_back"] == pytest.approx(T[0] - T[-1])
    assert e["T_mean"] == pytest.approx(np.dot(mz.capacity_share, T))


def test_measuring_points_come_from_evidence_e124():
    with pytest.raises(MissingEvidenceError, match="E-124"):
        MultiZoneCoach.measuring_points_from_evidence()
    reg = Registry().with_overrides({"E-124": "0.25, 0.5, 0.75"}, label="verification-only")
    assert MultiZoneCoach.measuring_points_from_evidence(reg) == (0.25, 0.5, 0.75)


@pytest.mark.parametrize("kw", [dict(supply_fraction=(0.5, 0.5, 0.5)),
                                dict(return_weight=(1.0, 0.0)), dict(G_mix_W_K=-1.0)])
def test_invalid_zone_settings(kw):
    with pytest.raises(ValueError):
        MultiZoneCoach(BASE, **kw)
