"""Verification of the virtual facilities (core.chamber) and sensors (core.sensors).

VERIFICATION-ONLY PARAMETERS: facility, sensor and unit values come from
tests/data/verification_overrides.yaml and tests/data/verification_unit.yaml.
They exercise the maths and are NOT data for any real facility or unit.
"""

import numpy as np
import pytest
from tests.conftest import verification_registry

from core import psychro
from core.chamber import UnitTestRoom, VehicleChamber, parse_dimensions
from core.procedures import run_procedure
from core.sensors import SensorSet
from core.unit import DXUnit
from evidence.registry import MissingEvidenceError, Registry


@pytest.fixture(scope="module")
def room(vreg):
    return UnitTestRoom.from_evidence(vreg)


@pytest.fixture(scope="module")
def unit(vreg, vcard):
    return DXUnit.from_card(vcard, evidence=vreg)


RATING_INDOOR = (26.7, psychro.humidity_ratio_from_wet_bulb(26.7, 19.4))
RATING_OUTDOOR = (35.0, psychro.humidity_ratio(35.0, 0.40))


# --------------------------------------------------------------------------
# Unit test room
# --------------------------------------------------------------------------

def test_room_needs_its_evidence():
    with pytest.raises(MissingEvidenceError, match="E-05"):
        UnitTestRoom.from_evidence(Registry())


@pytest.mark.parametrize("indoor,plr", [
    (RATING_INDOOR, 1.0),
    ((28.0, psychro.humidity_ratio(28.0, 0.50)), 1.0),
    ((24.0, psychro.humidity_ratio(24.0, 0.45)), 0.6),
])
def test_ideal_sensors_measure_the_model_capacity(room, unit, indoor, plr):
    m = room.measure(unit, indoor, RATING_OUTDOOR, "cooling", plr, 0.0, SensorSet.ideal())
    coil = unit.coil(*indoor, unit.m_supply_kg_s, RATING_OUTDOOR[0], plr)
    assert m.measured["capacity_gross_W"].value == pytest.approx(coil.Q_total_W, rel=1e-3)
    assert m.measured["sensible_capacity_gross_W"].value == pytest.approx(coil.Q_sensible_W,
                                                                          rel=1e-3)
    assert m.measured["capacity_gross_W"].uncertainty == 0.0
    assert m.validity == []


def test_ideal_measurement_at_rating_equals_datasheet(room, unit):
    m = room.measure(unit, RATING_INDOOR, RATING_OUTDOOR, "cooling", 1.0, 0.0, SensorSet.ideal())
    assert m.measured["capacity_gross_W"].value == pytest.approx(unit.Q_rated_W, rel=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_true_capacity_lies_within_measured_uncertainty(room, unit, vreg, seed):
    sensors = SensorSet.from_evidence(vreg, seed=seed)
    m = room.measure(unit, RATING_INDOOR, RATING_OUTDOOR, "cooling", 1.0, 0.0, sensors)
    for name in ("capacity_gross_W", "capacity_net_W", "sensible_capacity_gross_W"):
        meas, true = m.measured[name], m.true[name]
        assert meas.uncertainty > 0
        assert abs(meas.value - true) <= meas.uncertainty, name
        assert meas.uncertainty_rss <= meas.uncertainty


def test_room_capacity_limit_is_flagged(unit):
    small = UnitTestRoom.from_evidence(verification_registry(E_056=1000.0))
    m = small.measure(unit, RATING_INDOOR, RATING_OUTDOOR, "cooling", 1.0, 0.0, SensorSet.ideal())
    assert any(v.startswith("E-056") for v in m.validity)


def test_room_tolerance_is_checked_on_measured_conditions(unit):
    tight = UnitTestRoom.from_evidence(verification_registry(E_052=1e-4))
    sensors = SensorSet.from_evidence(verification_registry(), seed=3)
    m = tight.measure(unit, RATING_INDOOR, RATING_OUTDOOR, "cooling", 1.0, 0.0, sensors)
    assert any(v.startswith("E-052") for v in m.validity)


# --------------------------------------------------------------------------
# Vehicle chamber
# --------------------------------------------------------------------------

def test_chamber_dimensions_and_probes(vreg):
    ch = VehicleChamber.from_evidence(vreg)
    assert (ch.length_m, ch.width_m, ch.height_m) == (28.0, 5.0, 5.5)   # E-061
    assert ch.n_probes == 8                                              # E-067
    assert ch.volume_m3 == pytest.approx(770.0)
    assert parse_dimensions("28 x 5 x 5.5") == (28.0, 5.0, 5.5)


def test_measured_chamber_temperature_is_the_8_probe_mean(vreg):
    ch = VehicleChamber.from_evidence(vreg)
    sensors = SensorSet.from_evidence(vreg, seed=1)
    T = np.linspace(30.0, 35.0, 11)
    probes = ch.probe_readings(T, sensors)
    assert probes.shape == (8, 11)
    assert np.allclose(ch.measured_temperature(T, sensors), probes.mean(axis=0))
    assert np.all(np.abs(probes - T) <= sensors.accuracy["temperature"])


def test_chamber_ramp_respects_rate_limit(vreg):
    ch = VehicleChamber.from_evidence(vreg)          # ramp limit 10 K/h (verification)
    T = ch.air_temperature(20.0, 45.0, rate_K_h=50.0)
    assert T(3600.0) == pytest.approx(30.0)          # limited to 10 K/h
    assert T(2.5 * 3600.0) == pytest.approx(45.0)
    assert T(10 * 3600.0) == 45.0
    assert ch.ramp_duration_s(20.0, 45.0, 50.0) == pytest.approx(2.5 * 3600.0)


def test_chamber_holds_setpoint_within_tolerance_once_stable(vreg, vcoach, vcard):
    res = run_procedure("comfort_en13129_zone2", coach=vcoach, unit_card=vcard, evidence=vreg,
                        sensors="evidence", seed=5)
    ch = VehicleChamber.from_evidence(vreg)
    setpoint = vreg.number("E-023", "degC")
    measured = res.measurements["chamber_T_mean"]
    assert abs(measured.value - setpoint) <= ch.tol_T_K
    assert not any(v.startswith("E-062") for v in res.validity)
    # the reported value is the time-mean of the 8-probe mean
    w = res.segments[-1].df
    probe_mean = ch.measured_temperature(w["T_chamber_C"].to_numpy(),
                                         SensorSet.from_evidence(vreg, seed=5))
    t = w["t_s"].to_numpy()
    assert measured.value == pytest.approx(np.trapezoid(probe_mean, t) / (t[-1] - t[0]))


def test_chamber_capacity_limit_is_flagged(vcoach, vcard):
    reg = verification_registry(E_065=5000.0)
    res = run_procedure("comfort_en13129_zone2", coach=vcoach, unit_card=vcard, evidence=reg)
    assert any(v.startswith("E-065") for v in res.validity)
    assert not res.passed
