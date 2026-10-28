"""Verification of core.coach and core.procedures.k_value_test.

VERIFICATION-ONLY PARAMETERS: the coach below (A = 300 m2, k = 1.2 W/m2K,
C = 5e6 J/K) and all loads are round numbers chosen to exercise the maths.
They are NOT physical data for any real coach and must not be used as such;
real parameters come from the evidence registry.
"""

import math

import numpy as np
import pytest
from scipy.integrate import simpson

from core.coach import SingleZoneCoach, simulate, steady_state_T, time_constant_s
from core.procedures import k_value_test

A, K, C = 300.0, 1.2, 5e6          # m2, W/(m2.K), J/K  -- maths check only
T_OUT, Q_INT, Q_HVAC = 45.0, 5000.0, 20000.0  # degC, W, W


@pytest.fixture
def coach():
    return SingleZoneCoach("verification", A, K, C)


def energy_closure(coach, df):
    """Return (integral of net heat flow dt [J], C * (T_end - T0) [J])."""
    net_W = df["Q_env_W"] + df["Q_int_W"] - df["Q_hvac_W"]
    integral = simpson(net_W.to_numpy(), x=df["t_s"].to_numpy())
    stored = coach.heat_capacity_J_K * (df["T_C"].iloc[-1] - df["T_C"].iloc[0])
    return integral, stored


def test_ua_and_time_constant(coach):
    assert coach.UA == pytest.approx(A * K)
    assert time_constant_s(coach) == pytest.approx(C / (A * K))


# a) analytical solution ----------------------------------------------------

def test_matches_analytical_solution(coach):
    T0 = 45.0
    tau = time_constant_s(coach)
    T_ss = steady_state_T(coach, T_OUT, Q_INT, Q_HVAC)
    df = simulate(coach, T0, T_OUT, Q_INT, Q_HVAC, t_end_s=5 * tau, dt_out_s=60.0)

    T_exact = T_ss + (T0 - T_ss) * np.exp(-df["t_s"] / tau)
    assert np.max(np.abs(df["T_C"] - T_exact)) < 1e-3
    assert df["t_s"].iloc[-1] == pytest.approx(5 * tau)


# b) energy balance ---------------------------------------------------------

def test_energy_balance_closes(coach):
    df = simulate(coach, 45.0, T_OUT, Q_INT, Q_HVAC, t_end_s=72_000, dt_out_s=60.0)
    integral, stored = energy_closure(coach, df)
    assert integral == pytest.approx(stored, rel=1e-3)


# c) solver independence ----------------------------------------------------

def test_solver_tolerance_independence(coach):
    args = (coach, 45.0, T_OUT, Q_INT, Q_HVAC, 72_000, 60.0)
    loose = simulate(*args, rtol=1e-6, atol=1e-6)
    tight = simulate(*args, rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(loose["T_C"] - tight["T_C"])) < 1e-3


# d) steady state -----------------------------------------------------------

def test_reaches_steady_state_after_10_tau(coach):
    # T0 = 20 degC: offset to T_ss is 16.7 K, so exp(-10) leaves 0.76 mK.
    # (From T0 = 45 degC the 41.7 K offset would leave 1.9 mK > 1 mK.)
    T0 = 20.0
    tau = time_constant_s(coach)
    df = simulate(coach, T0, T_OUT, Q_INT, Q_HVAC, t_end_s=10 * tau, dt_out_s=600.0)
    assert df["T_C"].iloc[-1] == pytest.approx(
        steady_state_T(coach, T_OUT, Q_INT, Q_HVAC), abs=1e-3
    )


def test_steady_state_rejects_time_varying_input(coach):
    with pytest.raises(TypeError):
        steady_state_T(coach, lambda t: 20.0, Q_INT, Q_HVAC)


# e) k-value test -----------------------------------------------------------

def test_k_value_test_recovers_k(coach):
    tau = time_constant_s(coach)
    res = k_value_test(coach, T_out=0.0, Q_heater_W=10_000.0, t_end_s=10 * tau)
    assert res.k_measured_W_m2K == pytest.approx(K, rel=5e-3)
    assert 0 < res.t_steady_s < 10 * tau
    assert abs(res.dTdt_end_K_h) < 0.1


def test_k_value_test_raises_if_not_steady(coach):
    with pytest.raises(ValueError, match="not steady"):
        k_value_test(coach, T_out=0.0, Q_heater_W=10_000.0, t_end_s=3600.0)


# f) time-varying input -----------------------------------------------------

def test_sinusoidal_outside_temperature(coach):
    day = 86_400.0

    def T_out(t):
        return 30.0 + 10.0 * math.sin(2 * math.pi * t / day)

    df = simulate(coach, 45.0, T_out, Q_INT, Q_HVAC, t_end_s=2 * day, dt_out_s=60.0)
    assert np.all(np.isfinite(df["T_C"]))
    assert df["Q_env_W"].iloc[100] == pytest.approx(
        coach.UA * (T_out(df["t_s"].iloc[100]) - df["T_C"].iloc[100])
    )
    integral, stored = energy_closure(coach, df)
    assert integral == pytest.approx(stored, rel=1e-3)


# g) invalid inputs ---------------------------------------------------------

@pytest.mark.parametrize("area,k,cap", [
    (A, 0.0, C), (A, -1.2, C), (A, K, 0.0), (A, K, -5e6), (0.0, K, C), (A, math.nan, C),
])
def test_invalid_parameters_raise(area, k, cap):
    with pytest.raises(ValueError):
        SingleZoneCoach("bad", area, k, cap)
