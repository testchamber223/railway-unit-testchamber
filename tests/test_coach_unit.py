"""Verification of the coach in "with unit" mode (core.coach + core.unit).

VERIFICATION-ONLY PARAMETERS: coach A = 300 m2, k = 1.2 W/m2K, C = 5e6 J/K,
V = 150 m3; unit tests/data/verification_unit.yaml; 80 passengers at
70 W sensible / 50 W latent; outside 45 degC / 25 % RH. Round numbers chosen
to exercise the maths -- NOT data for any real coach or unit.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from core import loads, psychro
from core.coach import LoadTerm, SingleZoneCoach, UnitOperation, simulate
from core.unit import DXUnit, load_unit_card

UNIT = DXUnit.from_card(load_unit_card(Path(__file__).parent / "data" / "verification_unit.yaml"))
COACH = SingleZoneCoach("verification", 300.0, 1.2, 5e6, air_volume_m3=150.0)
T_OUT, RH_OUT = 45.0, 0.25
W_OUT = psychro.humidity_ratio(T_OUT, RH_OUT)
HOURS = 3600.0


def passengers():
    return LoadTerm("passengers", lambda t, T, W: loads.passengers(80, 70.0, 50.0))


def run(load_terms, mode="cooling", part_load=1.0, T0=45.0, W0=W_OUT, t_end=12 * HOURS,
        T_out=T_OUT):
    return simulate(COACH, T0=T0, T_out=T_out, Q_int=0.0, Q_hvac=0.0, t_end_s=t_end,
                    dt_out_s=30.0, W0=W0, loads=load_terms,
                    unit=UnitOperation(UNIT, mode, part_load, RH_OUT))


@pytest.fixture(scope="module")
def pulldown():
    return run([passengers()])


@pytest.fixture(scope="module")
def no_internal_loads():
    return run([])


def integral(df, y):
    return simpson(np.asarray(y, dtype=float), x=df["t_s"].to_numpy())


# --------------------------------------------------------------------------
# Pull-down
# --------------------------------------------------------------------------

def test_pulldown_cools_the_coach(pulldown):
    T = pulldown["T_C"]
    assert T.iloc[0] == 45.0
    assert T.iloc[-1] < 30.0
    assert (T.diff().iloc[1:] <= 1e-9).all()          # monotonic cool-down
    assert (pulldown["unit_T_supply_C"] < T).all()
    assert pulldown["unit_coil_wet"].iloc[-1]
    assert pulldown["RH"].max() <= 1.0


def test_zone_energy_balance_closes(pulldown):
    # envelope + passengers + supply air (which carries coil, fan and fresh air)
    net = pulldown["Q_env_W"] + pulldown["Q_load_passengers_W"] + pulldown["Q_load_unit_supply_W"]
    stored = COACH.heat_capacity_J_K * (pulldown["T_C"].iloc[-1] - pulldown["T_C"].iloc[0])
    assert integral(pulldown, net) == pytest.approx(stored, rel=1e-3)


def test_unit_energy_identity_includes_fan(pulldown):
    # m_s (h_supply - h_zone) = m_f (h_out - h_zone) - Q_coil_total + P_fan, at every step
    rhs = (pulldown["unit_Q_fresh_enthalpy_W"] - pulldown["unit_Q_coil_total_W"]
           + pulldown["unit_Q_fan_W"])
    np.testing.assert_allclose(pulldown["unit_Q_supply_total_W"], rhs, rtol=1e-9, atol=1e-6)


def test_zone_moisture_balance_closes(pulldown):
    net = pulldown["m_load_passengers_kg_s"] + pulldown["m_load_unit_supply_kg_s"]
    stored = pulldown.attrs["M_da_kg"] * (pulldown["W_kg_kg"].iloc[-1] - pulldown["W_kg_kg"].iloc[0])
    assert integral(pulldown, net) == pytest.approx(stored, rel=1e-3)


def test_unit_moisture_identity(pulldown):
    # m_s (W_supply - W_zone) = m_f (W_out - W_zone) - condensate
    W = pulldown["W_kg_kg"]
    lhs = pulldown["unit_m_supply_kg_s"] * (pulldown["unit_W_supply_kg_kg"] - W)
    rhs = pulldown["unit_m_fresh_kg_s"] * (W_OUT - W) - pulldown["unit_m_condensate_kg_s"]
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-12)


def test_steady_state_coil_sensible_equals_loads_plus_fan(pulldown):
    e = pulldown.iloc[-1]
    assert abs(e["Q_env_W"] + e["Q_load_passengers_W"] + e["Q_load_unit_supply_W"]) \
        < 1e-3 * e["unit_Q_coil_sensible_W"]          # steady: dT/dt ~ 0
    Tz, Wz = e["T_C"], e["W_kg_kg"]
    fresh_sensible = e["unit_m_fresh_kg_s"] * (psychro.enthalpy(T_OUT, Wz)
                                               - psychro.enthalpy(Tz, Wz))
    sensible_loads = e["Q_env_W"] + e["Q_load_passengers_W"] + fresh_sensible
    assert e["unit_Q_coil_sensible_W"] == pytest.approx(sensible_loads + e["unit_Q_fan_W"],
                                                        rel=5e-3)


# --------------------------------------------------------------------------
# Fresh air only via the mixing box
# --------------------------------------------------------------------------

def test_fresh_air_is_not_double_counted(no_internal_loads):
    # No passengers or sun: at steady state the coil handles exactly
    # envelope + fresh-air enthalpy difference + fan heat.
    e = no_internal_loads.iloc[-1]
    expected = e["Q_env_W"] + e["unit_Q_fresh_enthalpy_W"] + e["unit_Q_fan_W"]
    assert e["unit_Q_coil_total_W"] == pytest.approx(expected, rel=5e-3)
    assert e["unit_m_fresh_kg_s"] > 0


def test_separate_fresh_air_load_is_rejected():
    def fresh(t, T, W):
        q, _, m = loads.fresh_air(1200.0, T_OUT, RH_OUT, T, W)
        return q, m

    for term in (LoadTerm("fresh_air", fresh), LoadTerm("outside_air", fresh, is_fresh_air=True)):
        with pytest.raises(ValueError, match="mixing box"):
            run([term], t_end=600.0)


def test_unit_mode_requires_humidity_state():
    with pytest.raises(ValueError, match="W0"):
        simulate(COACH, 45.0, T_OUT, 0.0, 0.0, 600.0, 60.0,
                 unit=UnitOperation(UNIT, "cooling", 1.0, RH_OUT))


def test_part_load_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="part_load"):
        UnitOperation(UNIT, "cooling", 1.2, RH_OUT)


# --------------------------------------------------------------------------
# Other modes
# --------------------------------------------------------------------------

def test_heating_mode_warms_the_coach_and_balances():
    W0 = psychro.humidity_ratio(-10.0, 0.8)
    df = run([], mode="heating", part_load=1.0, T0=-10.0, W0=W0, t_end=6 * HOURS,
             T_out=-10.0)
    assert df["T_C"].iloc[-1] > 10.0
    assert (df["unit_Q_coil_total_W"] == 0).all()
    rhs = df["unit_Q_fresh_enthalpy_W"] + df["unit_Q_heater_W"] + df["unit_Q_fan_W"]
    np.testing.assert_allclose(df["unit_Q_supply_total_W"], rhs, rtol=1e-9, atol=1e-6)
    net = df["Q_env_W"] + df["Q_load_unit_supply_W"]
    stored = COACH.heat_capacity_J_K * (df["T_C"].iloc[-1] - df["T_C"].iloc[0])
    assert integral(df, net) == pytest.approx(stored, rel=1e-3)
