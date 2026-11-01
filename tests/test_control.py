"""Verification of core.control and the piece-7 sensor model.

VERIFICATION-ONLY PARAMETERS: coach A = 300 m2, k = 1.2 W/m2K, C = 5e6 J/K
(V = 150 m3 where humidity is needed); constant-capacity "verification mode"
plant (no coil curve); controller settings, sensor properties, setpoint-curve
points and the DX unit come from tests/data. Round numbers chosen to check the
maths -- NOT data for any real coach, unit, controller or sensor.
"""

import math

import numpy as np
import pytest
from scipy.integrate import simpson

from core import loads, psychro
from core.chamber import UnitTestRoom
from core.coach import LoadTerm, SingleZoneCoach, time_constant_s
from core.control import (ConstantCapacityPlant, Controller, ControlParams, ControlSensor,
                          FixedSetpoint, SetpointCurve, UnitPlant, simulate_controlled)
from core.sensors import SensorSet, SensorType
from core.unit import DXUnit, UnitCard, load_unit_card
from evidence.registry import MissingEvidenceError, Registry
from tests.conftest import DATA, verification_registry

COACH = SingleZoneCoach("verification", 300.0, 1.2, 5e6, air_volume_m3=150.0)
TAU = time_constant_s(COACH)                      # 13 889 s
T_OUT = 45.0
Q_COOL = 12600.0                                  # -> T_ss(on) = 45 - 12600/360 = 10 degC
SP = 24.0


def params(type_="on_off", db=2.0, min_on=0.0, min_off=0.0, sample=1.0, **kw):
    return ControlParams(type_, "return_air", db, 1.0, 5.0, min_on, min_off, sample, **kw)


def run(p, T0=24.0, hours=4.0, sensor=None, plant=None, T_out=T_OUT, setpoint=None, **kw):
    return simulate_controlled(COACH, T0, T_out, Controller(p), setpoint or FixedSetpoint(SP),
                               plant or ConstantCapacityPlant(Q_COOL), sensor or
                               ControlSensor.ideal(), hours * 3600.0, 60.0, **kw)


def periods(r, state):
    return [b - a for s, a, b in r.onoff_periods() if s == state]


# --------------------------------------------------------------------------
# Analytic on/off cycling
# --------------------------------------------------------------------------

def test_on_off_cycle_times_match_analytic_solution():
    r = run(params(db=2.0))                        # thresholds 23 / 25 degC
    T_high, T_low = SP + 1.0, SP - 1.0
    T_ss_on, T_ss_off = T_OUT - Q_COOL / COACH.UA, T_OUT
    t_on = TAU * math.log((T_high - T_ss_on) / (T_low - T_ss_on))      # 1987.5 s
    t_off = TAU * math.log((T_low - T_ss_off) / (T_high - T_ss_off))   # 1323.8 s
    on, off = periods(r, "on")[1:], periods(r, "off")[1:]                # periodic regime
    assert len(on) >= 2 and len(off) >= 2
    for t in on:
        assert t == pytest.approx(t_on, rel=5e-3)
    for t in off:
        assert t == pytest.approx(t_off, rel=5e-3)


def test_controller_reads_the_virtual_sensor_not_the_true_state():
    biased = ControlSensor(SensorType("temperature"), np.array([0.5]))   # reads 0.5 K high
    r = run(params(db=2.0), sensor=biased, hours=3.0)
    d = r.data[r.data["t_s"] > 3600]
    assert d["T_C"].max() == pytest.approx(SP + 1.0 - 0.5, abs=0.01)
    assert d["T_C"].min() == pytest.approx(SP - 1.0 - 0.5, abs=0.01)


# --------------------------------------------------------------------------
# Minimum on / off times, sensor lag, staged, inverter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("type_,kw", [
    ("on_off", {}),
    ("staged", {"stage_fractions": (0.9, 1.0)}),      # stage 1 alone can cool: cycles
    ("inverter", {"kp": 0.5, "ki": 3e-4, "u_min": 0.95, "u_max": 1.0}),
])
def test_minimum_on_and_off_times_are_never_violated(type_, kw):
    # Dead band 0.4 K: natural on ~400 s / off ~270 s, so 600 / 900 s bind.
    p = params(type_, db=0.4, min_on=600.0, min_off=900.0, sample=5.0, **kw)
    r = run(p, T0=25.0, hours=6.0)
    on, off = periods(r, "on"), periods(r, "off")
    assert len(on) >= 3, "the test needs cycling"
    assert min(on) >= 600.0 - 1e-6
    assert min(off) >= 900.0 - 1e-6


def test_sensor_lag_increases_the_overshoot():
    overshoot = []
    for tau_s in (0.0, 60.0, 240.0):
        r = run(params(db=2.0), sensor=ControlSensor.ideal(tau_s), hours=4.0)
        d = r.data[r.data["t_s"] > 3000]
        overshoot.append(max(d["T_C"].max() - (SP + 1.0), (SP - 1.0) - d["T_C"].min()))
    assert overshoot[0] < 0.01                        # sampling only
    assert overshoot[0] < overshoot[1] < overshoot[2]
    assert overshoot[1] > 0.02                        # lag pushes beyond the dead band


def test_staged_uses_only_the_allowed_fractions():
    r = run(params("staged", db=1.0, sample=5.0, stage_fractions=(0.5, 1.0)), T0=30.0,
            hours=6.0)
    used = set(r.samples["fraction"].round(12))
    assert used <= {0.0, 0.5, 1.0}
    assert {0.5, 1.0} <= used


def test_inverter_holds_setpoint_without_switching():
    r = run(params("inverter", db=1.0, sample=10.0, kp=0.5, ki=3e-4, u_min=0.3, u_max=1.0),
            T0=30.0, hours=8.0)
    assert r.starts == 1
    assert not [e for e in r.events if e["to_mode"] == "off"]
    last = r.data[r.data["t_s"] >= 7 * 3600]
    assert np.max(np.abs(last["T_C"] - SP)) < 0.2
    assert 0.3 < r.samples["fraction"].iloc[-1] < 1.0     # modulating, not saturated


# --------------------------------------------------------------------------
# Heating / cooling interlock
# --------------------------------------------------------------------------

def test_heating_and_cooling_never_on_together():
    def T_out(t):                                     # 4 .. 44 degC, 12 h period
        return 24.0 + 20.0 * math.sin(2 * math.pi * t / 43_200.0)

    p = ControlParams("on_off", "return_air", 1.0, 1.0, 3.0, 120.0, 120.0, 10.0)
    r = run(p, T0=24.0, hours=24.0, T_out=T_out, plant=ConstantCapacityPlant(Q_COOL, Q_COOL))
    d = r.data
    assert not ((d["Q_cooling_W"] > 0) & (d["Q_heating_W"] > 0)).any()
    modes = set(r.samples["mode"])
    assert {"cooling", "heating"} <= modes
    for e in r.events:                                # always through "off"
        assert {e["from_mode"], e["to_mode"]} != {"cooling", "heating"}


def test_neutral_zone_too_small_is_refused():
    with pytest.raises(ValueError, match="fight"):
        ControlParams("on_off", "return_air", 2.0, 2.0, 1.5, 0, 0, 10.0)


# --------------------------------------------------------------------------
# Setpoint curve
# --------------------------------------------------------------------------

CURVE = {"E-110": 19.0, "E-111": 22.0, "E-112": 35.0, "E-113": 26.0,
         "E-114": 45.0, "E-115": 28.5}


def test_setpoint_curve_interpolates_its_points_exactly():
    sc = SetpointCurve.from_evidence(
        Registry().with_overrides(CURVE, label="verification-only"))
    for x, y in [(19.0, 22.0), (35.0, 26.0), (45.0, 28.5)]:
        assert sc(x) == y
    assert sc(27.0) == pytest.approx(22.0 + 0.25 * (27.0 - 19.0))
    assert sc(40.0) == pytest.approx(27.25)
    assert sc(5.0) == 22.0 and sc(50.0) == 28.5          # flat outside
    assert sc.evidence_ids == tuple(CURVE)


def test_setpoint_curve_refuses_empty_values():
    with pytest.raises(MissingEvidenceError) as exc:
        SetpointCurve.from_evidence(Registry())
    assert exc.value.ids == list(CURVE)


def test_setpoint_curve_drives_the_controller():
    sc = SetpointCurve.from_evidence(
        Registry().with_overrides(CURVE, label="verification-only"))
    r = run(params(db=1.0), T0=27.0, hours=4.0, setpoint=sc)       # T_out 45 -> sp 28.5
    assert set(r.samples["setpoint"]) == {28.5}


# --------------------------------------------------------------------------
# Controller from the unit card
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ctype", ["on_off", "staged", "inverter"])
def test_control_params_from_card(ctype):
    card = load_unit_card(DATA / "verification_unit.yaml")
    data = card.model_dump(exclude_none=True)
    data["control_type"]["value"] = ctype
    p = ControlParams.from_card(UnitCard.model_validate(data))
    assert p.type == ctype and p.sample_s == 10.0 and p.dead_band_K == 1.0
    if ctype == "staged":
        assert p.stage_fractions == (0.5, 1.0)


def test_control_params_refuse_missing_fields():
    with pytest.raises(MissingEvidenceError, match="control_type"):
        ControlParams.from_card(load_unit_card("kaeser_noske_dhf14"))
    card = load_unit_card(DATA / "verification_unit.yaml")
    data = card.model_dump(exclude_none=True)
    data["control_type"]["value"] = "inverter"
    del data["control_ki"]
    with pytest.raises(MissingEvidenceError) as exc:
        ControlParams.from_card(UnitCard.model_validate(data))
    assert exc.value.ids == ["control_ki"]


# --------------------------------------------------------------------------
# Total enthalpy balance with switching (DX unit)
# --------------------------------------------------------------------------

def test_total_enthalpy_balance_closes_with_switching():
    card = load_unit_card(DATA / "verification_unit.yaml")
    unit = DXUnit.from_card(card)
    p = ControlParams.from_card(card)                      # on_off, 1 K, 180 s, 10 s
    RH_OUT = 0.40
    W0 = psychro.humidity_ratio(35.0, RH_OUT)
    pax = LoadTerm("passengers", lambda t, T, W: loads.passengers(80, 70.0, 50.0, T_in=T))
    r = simulate_controlled(COACH, 35.0, 35.0, Controller(p), FixedSetpoint(SP),
                            UnitPlant(unit, RH_OUT), ControlSensor.ideal(), 3 * 3600.0, 30.0,
                            W0=W0, loads=[pax])
    assert r.starts >= 3, "the test needs switching"
    inflow = stored = 0.0
    for seg in r.segments:
        q = (seg["Q_env_W"] + seg["Q_load_passengers_W"] + 80 * 50.0
             + seg["unit_Q_supply_total_W"]).to_numpy()
        inflow += simpson(q, x=seg["t_s"].to_numpy())
        T, W = seg["T_C"].to_numpy(), seg["W_kg_kg"].to_numpy()
        h_g = np.array([loads.vapour_enthalpy(x) for x in T])
        stored += COACH.heat_capacity_J_K * (T[-1] - T[0]) + r.M_da_kg * np.sum(
            0.5 * (h_g[1:] + h_g[:-1]) * np.diff(W))
    assert inflow == pytest.approx(stored, rel=1e-3)


def test_verification_mode_energy_balance_closes_with_switching():
    r = run(params(db=1.0, min_on=120.0, min_off=120.0, sample=5.0), hours=4.0)
    assert len(r.events) > 5
    inflow = sum(simpson((s["Q_env_W"] - s["Q_hvac_W"]).to_numpy(), x=s["t_s"].to_numpy())
                 for s in r.segments)
    stored = COACH.heat_capacity_J_K * (r.segments[-1]["T_C"].iloc[-1] - 24.0)
    assert inflow == pytest.approx(stored, rel=1e-3, abs=1e3)
    # segments are contiguous: every switch restarts the integration at the event time
    for a, b in zip(r.segments, r.segments[1:]):
        assert a["t_s"].iloc[-1] == b["t_s"].iloc[0]
        assert a["T_C"].iloc[-1] == pytest.approx(b["T_C"].iloc[0], abs=1e-12)


# --------------------------------------------------------------------------
# Sensor model
# --------------------------------------------------------------------------

def test_sensor_lag_is_exact_first_order_response():
    st = SensorType("temperature", tau_s=30.0)
    t = np.linspace(0, 300, 61)                       # 5 s steps
    y = st.lag(t, np.concatenate([[0.0], np.ones(60)]))
    # input ramps 0 -> 1 over the first 5 s then holds: compare after the ramp
    exact = 1 - (30 / 5) * (1 - np.exp(-5 / 30)) * np.exp(-(t - 5) / 30)
    assert np.allclose(y[1:], exact[1:], atol=1e-12)


def test_resolution_sampling_and_error_bound():
    reg = verification_registry()
    sensors = SensorSet.from_evidence(reg, seed=4)
    t = np.arange(0.0, 20.0, 0.25)
    true = 20.0 + 0.01 * t
    y = sensors.read("T_x", "temperature", true, n_probes=3, t=t)
    res = sensors.types["temperature"].resolution
    assert np.allclose(np.round(y / res) * res, y)                  # on the resolution grid
    held = y[:, 1:4]                                                 # within one 1 s sample
    assert np.all(held == held[:, :1])
    static = sensors.read("T_y", "temperature", true, n_probes=8)
    assert np.all(np.abs(static - true) <= sensors.types["temperature"].bound + 1e-12)


def test_dew_point_hygrometer_reduces_capacity_uncertainty():
    reg = verification_registry()
    card = load_unit_card(DATA / "verification_unit.yaml")
    unit = DXUnit.from_card(card, evidence=reg)
    room = UnitTestRoom.from_evidence(reg)
    indoor = (26.7, psychro.humidity_ratio_from_wet_bulb(26.7, 19.4))
    outdoor = (35.0, psychro.humidity_ratio(35.0, 0.40))
    U = {}
    for hum in ("rh", "dew_point"):
        sensors = SensorSet.from_evidence(reg, seed=0)
        m = room.measure(unit, indoor, outdoor, "cooling", 1.0, 0.0, sensors, humidity=hum)
        g = m.measured["capacity_gross_W"]
        assert abs(g.value - m.true["capacity_gross_W"]) <= g.uncertainty
        U[hum] = g.uncertainty
    assert U["dew_point"] < 0.5 * U["rh"]
