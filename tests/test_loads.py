"""Verification of core.loads against hand calculations.

VERIFICATION-ONLY INPUTS: every number below is a round value chosen to check
the arithmetic. None of them is physical data for a real coach; real values
come from the evidence registry.
"""

import pytest

from core import loads, psychro

REL = 1e-9


def test_envelope_hand_calc():
    # k = 1.2 W/m2K, Ae = 300 m2, T_out = 35 degC, T_in = 27 degC
    # Q = 1.2 * 300 * (35 - 27) = 360 * 8 = 2880 W
    assert loads.envelope(1.2, 300.0, 35.0, 27.0) == pytest.approx(2880.0, rel=REL)


def test_envelope_heat_loss_is_negative():
    # T_out < T_in: Q = 1.2 * 300 * (0 - 25) = -9000 W
    assert loads.envelope(1.2, 300.0, 0.0, 25.0) == pytest.approx(-9000.0, rel=REL)


def test_solar_glazing_hand_calc():
    # I = 700 W/m2 * 0.866 (vertical factor) = 606.2 W/m2, A_glass = 10 m2, g = 0.5
    # Q = 606.2 * 10 * 0.5 = 3031 W
    assert loads.solar_glazing(606.2, 10.0, 0.5) == pytest.approx(3031.0, rel=REL)


def test_solar_opaque_sol_air_hand_calc():
    # k = 1.2, A_opaque = 60 m2, alpha = 0.6, I = 700 W/m2, h_o = 17 W/m2K,
    # T_out = 35, T_in = 27
    # T_sol_air = 35 + 0.6 * 700 / 17 = 35 + 24.705882... = 59.705882... degC
    # Q = 1.2 * 60 * (59.705882... - 27) = 72 * 32.705882... = 2354.823529... W
    expected = 72.0 * (35.0 + 420.0 / 17.0 - 27.0)
    assert expected == pytest.approx(2354.8235294117646, rel=REL)
    assert loads.solar_opaque_sol_air(1.2, 60.0, 0.6, 700.0, 17.0, 35.0, 27.0) == \
        pytest.approx(expected, rel=REL)


def test_solar_opaque_increment_over_envelope():
    # The solar increment alone is k * A * alpha * I / h_o = 72 * 420 / 17 = 1778.82... W
    total = loads.solar_opaque_sol_air(1.2, 60.0, 0.6, 700.0, 17.0, 35.0, 27.0)
    increment = total - loads.envelope(1.2, 60.0, 35.0, 27.0)
    assert increment == pytest.approx(72.0 * 420.0 / 17.0, rel=REL)


def test_passengers_hand_calc():
    # n = 80, sensible 70 W/person, latent 50 W/person, h_g0 = 2501 kJ/kg (E-006)
    # Q_sens = 80 * 70 = 5600 W
    # m_w    = 80 * 50 / 2 501 000 = 4000 / 2 501 000 = 1.59936...e-3 kg/s
    Q_sens, m_w = loads.passengers(80, 70.0, 50.0)
    assert Q_sens == pytest.approx(5600.0, rel=REL)
    assert m_w == pytest.approx(4000.0 / 2_501_000.0, rel=REL)


def test_fresh_air_hand_calc():
    # Inputs: 1200 m3/h of outside air at 35 degC / 40 % RH; interior 27 degC,
    # W_in = 0.010 kg/kg; p = 101325 Pa.
    V, T_out, RH_out, T_in, W_in, p = 1200.0, 35.0, 0.40, 27.0, 0.010, 101325.0

    # Saturation pressure from core.psychro (verified in piece 2); the rest by hand.
    p_w = RH_out * psychro.sat_vapor_pressure(T_out)               # Pa
    W_out = 0.621945 * p_w / (p - p_w)                              # ASHRAE Eq. 20
    v_out = 287.042 * (T_out + 273.15) * (1 + 1.607858 * W_out) / p  # ASHRAE Eq. 26
    m_da = V / 3600.0 / v_out                                       # kg dry air / s

    def h(T, W):  # J/kg dry air; 1.006, 2501, 1.86 kJ/kg(K) = E-005, E-006, E-007
        return 1006.0 * T + W * (2_501_000.0 + 1860.0 * T)

    Q_sens = m_da * (h(T_out, W_in) - h(T_in, W_in))
    Q_lat = m_da * (h(T_out, W_out) - h(T_out, W_in))
    m_w = m_da * (W_out - W_in)

    got = loads.fresh_air(V, T_out, RH_out, T_in, W_in, p)
    assert got[0] == pytest.approx(Q_sens, rel=REL)
    assert got[1] == pytest.approx(Q_lat, rel=REL)
    assert got[2] == pytest.approx(m_w, rel=REL)
    assert Q_sens > 0 and Q_lat > 0 and m_w > 0  # hot, humid outside air


def test_fresh_air_at_interior_state_gives_zero_load():
    T, RH = 27.0, 0.50
    W_in = psychro.humidity_ratio(T, RH)          # W_out == W_in, T_out == T_in
    Q_sens, Q_lat, m_w = loads.fresh_air(1200.0, T, RH, T, W_in)
    assert Q_sens == pytest.approx(0.0, abs=1e-9)
    assert Q_lat == pytest.approx(0.0, abs=1e-9)
    assert m_w == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("call", [
    lambda: loads.envelope(-1.2, 300.0, 35.0, 27.0),
    lambda: loads.solar_glazing(606.2, 10.0, 1.5),
    lambda: loads.solar_glazing(-1.0, 10.0, 0.5),
    lambda: loads.solar_opaque_sol_air(1.2, 60.0, 0.6, 700.0, 0.0, 35.0, 27.0),
    lambda: loads.solar_opaque_sol_air(1.2, 60.0, 1.2, 700.0, 17.0, 35.0, 27.0),
    lambda: loads.passengers(-1, 70.0, 50.0),
    lambda: loads.fresh_air(-10.0, 35.0, 0.4, 27.0, 0.01),
    lambda: loads.fresh_air(1200.0, 35.0, 40, 27.0, 0.01),   # RH as percent
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
