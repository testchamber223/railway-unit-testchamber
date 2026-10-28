"""Verification of core.unit (unit card, DX curve, ADP/BF coil, mixing, fan, heater).

VERIFICATION-ONLY UNIT PARAMETERS: tests/data/verification_unit.yaml (30 kW at
35 degC outside / 26.7 degC db / 19.4 degC wb inside, SHR 0.75, 6000 m3/h,
1200 m3/h fresh, 1.5 kW draw-through fan, 20 kW heater) and the curve limits
below are round numbers chosen to exercise the maths. They are NOT data for
any real unit.
"""

import dataclasses
from pathlib import Path

import pytest

from core import psychro
from core.unit import (AirState, CurveLimits, DXUnit, UnitCard, capacity_modifier,
                       load_unit_card, mixing_box)
from evidence.registry import DEFAULT_CSV, MissingEvidenceError, Registry

VERIFICATION_CARD = Path(__file__).parent / "data" / "verification_unit.yaml"
TEST_LIMITS = CurveLimits(12.8, 23.9, 18.0, 46.1)   # test limits, not E-048


@pytest.fixture(scope="module")
def unit():
    return DXUnit.from_card(load_unit_card(VERIFICATION_CARD), limits=TEST_LIMITS)


# --------------------------------------------------------------------------
# Unit card loader
# --------------------------------------------------------------------------

def test_dhf14_card_reports_missing_fields():
    card = load_unit_card("kaeser_noske_dhf14")
    open_fields = dict(card.open_items())
    for name in ["rating_indoor_dry_bulb", "rating_indoor_wet_bulb", "fresh_airflow",
                 "compressor_power", "fan_power", "fan_position", "refrigerant"]:
        assert open_fields[name] == "assumption (value empty)", name
    assert "E-050" in open_fields["rated_shr / rated_sensible_capacity"]
    # datasheet values are present and converted to SI
    assert card.si("rated_total_cooling_capacity") == pytest.approx(4000.0)
    assert card.si("supply_airflow") == pytest.approx(700.0 / 3600.0)
    assert card.si("heating_capacity") == pytest.approx(4500.0)
    assert card.si("heating_rating_outdoor_dry_bulb") == -20.0
    assert "rated_total_cooling_capacity" not in open_fields
    assert "8 open datasheet item(s)" in card.report()


def test_dhf14_model_refuses_to_run_and_lists_fields():
    with pytest.raises(MissingEvidenceError) as exc:
        DXUnit.from_card(load_unit_card("kaeser_noske_dhf14"))
    for name in ["rating_indoor_dry_bulb", "rating_indoor_wet_bulb / rating_indoor_rh",
                 "fresh_airflow", "fan_power", "fan_position", "rated SHR (E-050 empty)"]:
        assert name in exc.value.ids
    assert "rated_total_cooling_capacity" not in exc.value.ids


def _card_dict(**overrides):
    import yaml
    data = yaml.safe_load(VERIFICATION_CARD.read_text())
    data.update(overrides)
    return data


@pytest.mark.parametrize("override", [
    {"supply_airflow": {"value": 6000, "unit": "kg/h", "source": "x", "status": "assumption"}},
    {"rated_shr": {"value": None, "unit": "-", "source": "x", "status": "datasheet"}},
    {"fan_power": {"value": 1500, "unit": "W", "status": "datasheet"}},          # no source
    {"fan_position": {"value": "sideways", "unit": "-", "source": "x", "status": "assumption"}},
    {"colour": {"value": "red", "unit": "-", "source": "x", "status": "assumption"}},
])
def test_invalid_cards_are_rejected(override):
    with pytest.raises(ValueError):
        UnitCard.model_validate(_card_dict(**override))


def test_shr_falls_back_to_evidence_e050(tmp_path):
    data = _card_dict()
    del data["rated_shr"]
    card = UnitCard.model_validate(data)
    with pytest.raises(MissingEvidenceError, match="E-050"):
        DXUnit.from_card(card)

    csv = DEFAULT_CSV.read_text(encoding="utf-8").splitlines()
    csv = [line.replace("E-050,Rated sensible heat ratio fallback (unit datasheet gives no SHR),"
                        "SHR_rated,,", "E-050,Rated sensible heat ratio fallback (unit datasheet "
                        "gives no SHR),SHR_rated,0.75,") for line in csv]
    path = tmp_path / "evidence.csv"
    path.write_text("\n".join(csv) + "\n", encoding="utf-8")
    assert DXUnit.from_card(card, evidence=Registry(path)).shr_rated == 0.75


def test_sensible_capacity_gives_shr():
    data = _card_dict(rated_sensible_capacity={"value": 21, "unit": "kW", "source": "x",
                                               "status": "assumption"})
    del data["rated_shr"]
    assert DXUnit.from_card(UnitCard.model_validate(data)).shr_rated == pytest.approx(0.7)


# --------------------------------------------------------------------------
# Capacity curve and normalisation
# --------------------------------------------------------------------------

def test_published_curve_is_one_at_ahri_point():
    assert capacity_modifier(19.44, 35.0) == pytest.approx(1.0, abs=0.002)


def test_capacity_equals_datasheet_at_rating_condition(unit):
    Q = unit.total_capacity(unit.Twb_rating_C, unit.T_rating_out_C)
    assert Q == pytest.approx(unit.Q_rated_W, rel=1e-9)


def test_coil_reproduces_rating_point(unit):
    r = unit.coil(unit.T_rating_in_C, unit.W_rating_in, unit.m_supply_kg_s,
                  unit.T_rating_out_C, 1.0)
    assert r.Q_total_W == pytest.approx(unit.Q_rated_W, rel=1e-9)
    assert r.shr == pytest.approx(unit.shr_rated, rel=1e-9)
    assert r.wet


def test_capacity_falls_with_outdoor_temperature(unit):
    Q = [unit.total_capacity(19.4, T) for T in (25, 35, 45, 50)]
    assert Q == sorted(Q, reverse=True)


def test_bypass_factor_scales_with_flow(unit):
    bf_r, _, _ = unit.bypass_rated
    assert unit.bypass_factor(unit.m_supply_kg_s) == pytest.approx(bf_r)
    assert unit.bypass_factor(0.5 * unit.m_supply_kg_s) == pytest.approx(bf_r ** 2)


# --------------------------------------------------------------------------
# Coil balances and physical bounds
# --------------------------------------------------------------------------

INLETS = [  # (T_in, RH_in, Tdb_out, part_load)
    (26.7, 0.50, 35.0, 1.0), (30.0, 0.60, 45.0, 1.0), (24.0, 0.40, 25.0, 1.0),
    (30.0, 0.15, 35.0, 1.0), (28.0, 0.70, 40.0, 0.5), (22.0, 0.30, 30.0, 0.25),
    (35.0, 0.35, 50.0, 1.0), (27.0, 0.55, 35.0, 0.0),
]


@pytest.mark.parametrize("T_in,RH_in,Tdb_out,plr", INLETS)
def test_coil_energy_and_moisture_balance(unit, T_in, RH_in, Tdb_out, plr):
    W_in = psychro.humidity_ratio(T_in, RH_in)
    m = unit.m_supply_kg_s
    r = unit.coil(T_in, W_in, m, Tdb_out, plr)
    h_in, h_out = psychro.enthalpy(T_in, W_in), psychro.enthalpy(r.T_out_C, r.W_out)
    assert r.Q_total_W == pytest.approx(m * (h_in - h_out), rel=1e-3, abs=1e-6)
    assert r.condensate_kg_s == pytest.approx(m * (W_in - r.W_out), rel=1e-3, abs=1e-12)


@pytest.mark.parametrize("T_in,RH_in,Tdb_out,plr", INLETS)
def test_coil_outlet_is_physical(unit, T_in, RH_in, Tdb_out, plr):
    W_in = psychro.humidity_ratio(T_in, RH_in)
    r = unit.coil(T_in, W_in, unit.m_supply_kg_s, Tdb_out, plr)
    assert psychro.rh_from_humidity_ratio(r.T_out_C, r.W_out) <= 1.0 + 1e-9
    assert 0.0 <= r.shr <= 1.0
    assert r.condensate_kg_s >= 0.0
    assert r.T_out_C <= T_in


def test_dry_coil_when_dew_point_below_adp(unit):
    _, T_adp_rated, _ = unit.bypass_rated
    T_in, W_in = 30.0, 0.004                       # dew point ~ 0.7 degC
    assert psychro.dew_point(T_in, psychro.rh_from_humidity_ratio(T_in, W_in)) < T_adp_rated
    r = unit.coil(T_in, W_in, unit.m_supply_kg_s, 35.0, 1.0)
    assert not r.wet
    assert r.condensate_kg_s == 0.0 and r.W_out == W_in and r.shr == 1.0
    assert r.Q_total_W > 0


def test_wet_coil_when_dew_point_above_adp(unit):
    r = unit.coil(28.0, psychro.humidity_ratio(28.0, 0.7), unit.m_supply_kg_s, 35.0, 1.0)
    assert r.wet and r.condensate_kg_s > 0 and r.shr < 1.0


def test_part_load_scales_total_capacity(unit):
    W_in = psychro.humidity_ratio(27.0, 0.5)
    full = unit.coil(27.0, W_in, unit.m_supply_kg_s, 35.0, 1.0)
    half = unit.coil(27.0, W_in, unit.m_supply_kg_s, 35.0, 0.5)
    assert half.Q_total_W == pytest.approx(0.5 * full.Q_total_W, rel=1e-9)
    assert half.condensate_kg_s == pytest.approx(0.5 * full.condensate_kg_s, rel=1e-9)


def test_extrapolation_flag(unit):
    W_in = psychro.humidity_ratio(26.7, 0.5)
    assert unit.coil(26.7, W_in, unit.m_supply_kg_s, 35.0, 1.0).extrapolated is False
    assert unit.coil(26.7, W_in, unit.m_supply_kg_s, 50.0, 1.0).extrapolated is True   # y > 46.1
    hot_humid = psychro.humidity_ratio(35.0, 0.7)                                        # x > 23.9
    assert unit.coil(35.0, hot_humid, unit.m_supply_kg_s, 35.0, 1.0).extrapolated is True
    no_limits = dataclasses.replace(unit, limits=None)
    assert no_limits.coil(26.7, W_in, unit.m_supply_kg_s, 50.0, 1.0).extrapolated is None


def test_curve_limits_text_format():
    assert CurveLimits.from_text("12.8..23.9; 18..46.1") == TEST_LIMITS
    with pytest.raises(ValueError):
        CurveLimits.from_text("12.8 to 23.9")


def test_invalid_coil_inputs(unit):
    with pytest.raises(ValueError):
        unit.coil(27.0, 0.01, unit.m_supply_kg_s, 35.0, 1.5)
    with pytest.raises(ValueError):
        unit.coil(27.0, 0.01, 0.0, 35.0, 1.0)


# --------------------------------------------------------------------------
# Mixing box, fan, heater
# --------------------------------------------------------------------------

def test_mixing_equal_flows_is_average():
    a = AirState(24.0, 0.009, 1.0)
    b = AirState(40.0, 0.015, 1.0)
    mix = mixing_box(a, b)
    assert mix.m_da == 2.0
    assert mix.W == pytest.approx((0.009 + 0.015) / 2, rel=1e-12)
    h_avg = (psychro.enthalpy(24.0, 0.009) + psychro.enthalpy(40.0, 0.015)) / 2
    assert psychro.enthalpy(mix.T_C, mix.W) == pytest.approx(h_avg, rel=1e-12)


def test_fan_heat_draw_through_is_after_coil(unit):
    W_z, W_f = psychro.humidity_ratio(27.0, 0.5), psychro.humidity_ratio(35.0, 0.4)
    res = unit.operate(27.0, W_z, 35.0, W_f, 35.0, "cooling", 1.0)
    m = res.supply.m_da
    h_supply = psychro.enthalpy(res.supply.T_C, res.supply.W)
    h_coil_out = psychro.enthalpy(res.coil.T_out_C, res.coil.W_out)
    assert m * (h_supply - h_coil_out) == pytest.approx(unit.P_fan_W, rel=1e-9)


def test_fan_heat_blow_through_is_before_coil(unit):
    blow = dataclasses.replace(unit, fan_position="blow-through")
    W_z, W_f = psychro.humidity_ratio(27.0, 0.5), psychro.humidity_ratio(35.0, 0.4)
    res = blow.operate(27.0, W_z, 35.0, W_f, 35.0, "cooling", 1.0)
    m = res.supply.m_da
    # supply = coil outlet; coil inlet = mixed air + fan heat
    assert (res.supply.T_C, res.supply.W) == (res.coil.T_out_C, res.coil.W_out)
    h_mix = psychro.enthalpy(res.mixed.T_C, res.mixed.W)
    h_supply = psychro.enthalpy(res.supply.T_C, res.supply.W)
    assert m * (h_mix - h_supply) + unit.P_fan_W == pytest.approx(res.coil.Q_total_W, rel=1e-9)


def test_heating_mode(unit):
    W = psychro.humidity_ratio(15.0, 0.4)
    res = unit.operate(15.0, W, -10.0, psychro.humidity_ratio(-10.0, 0.8), -10.0, "heating", 0.6)
    assert res.Q_heater_W == pytest.approx(0.6 * unit.Q_heat_W)
    assert res.coil is None
    m = res.supply.m_da
    h_gain = m * (psychro.enthalpy(res.supply.T_C, res.supply.W)
                  - psychro.enthalpy(res.mixed.T_C, res.mixed.W))
    assert h_gain == pytest.approx(res.Q_heater_W + unit.P_fan_W, rel=1e-9)
    assert res.supply.W == pytest.approx(res.mixed.W)


def test_ventilation_mode_only_adds_fan_heat(unit):
    W = psychro.humidity_ratio(25.0, 0.5)
    res = unit.operate(25.0, W, 25.0, W, 25.0, "ventilation", 0.0)
    m = res.supply.m_da
    assert m * (psychro.enthalpy(res.supply.T_C, W) - psychro.enthalpy(25.0, W)) == \
        pytest.approx(unit.P_fan_W, rel=1e-9)
