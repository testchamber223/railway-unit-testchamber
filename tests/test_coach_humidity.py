"""Verification of the coach moisture state, combined loads and the YAML loader.

VERIFICATION-ONLY PARAMETERS: the coach (A = 300 m2, k = 1.2 W/m2K,
C = 5e6 J/K, V = 150 m3) and all loads below are round numbers chosen to
exercise the maths. They are NOT physical data for any real coach.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from core import loads, psychro
from core.coach import (CoachDefinition, LoadTerm, SingleZoneCoach, load_coach_definition,
                        simulate)
from evidence.registry import COLUMNS, EvidenceError, MissingEvidenceError, Registry

A, K, C, V = 300.0, 1.2, 5e6, 150.0
T_OUT, RH_OUT = 35.0, 0.40
N_PAX, Q_PAX_SENS, Q_PAX_LAT = 80, 70.0, 50.0
V_FRESH = 1200.0  # m3/h
Q_EQUIP, Q_HVAC = 2000.0, 15000.0


@pytest.fixture
def coach():
    return SingleZoneCoach("verification", A, K, C, air_volume_m3=V)


def integral(df, cols, sign=None):
    sign = sign or {}
    y = sum(sign.get(c, 1.0) * df[c].to_numpy() for c in cols)
    return simpson(y, x=df["t_s"].to_numpy())


# --------------------------------------------------------------------------
# Load terms for the combined scenario
# --------------------------------------------------------------------------

def solar_I(t):
    """Half-sine 'day' of 12 h peaking at 700 W/m2 (maths check only)."""
    return 700.0 * max(0.0, math.sin(math.pi * t / 43_200.0))


def all_load_terms():
    def glazing(t, T, W):
        return loads.solar_glazing(0.866 * solar_I(t), 10.0, 0.5), 0.0

    def opaque(t, T, W):   # solar increment only; plain conduction is in Q_env
        q = (loads.solar_opaque_sol_air(K, 60.0, 0.6, solar_I(t), 17.0, T_OUT, T)
             - loads.envelope(K, 60.0, T_OUT, T))
        return q, 0.0

    def pax(t, T, W):
        return loads.passengers(N_PAX, Q_PAX_SENS, Q_PAX_LAT)

    def fresh(t, T, W):
        q_sens, _, m_w = loads.fresh_air(V_FRESH, T_OUT, RH_OUT, T, W)
        return q_sens, m_w

    return [LoadTerm("solar_glazing", glazing), LoadTerm("solar_opaque", opaque),
            LoadTerm("passengers", pax), LoadTerm("fresh_air", fresh)]


@pytest.fixture
def combined(coach):
    W0 = psychro.humidity_ratio(T_OUT, RH_OUT)
    return simulate(coach, T0=T_OUT, T_out=T_OUT, Q_int=Q_EQUIP, Q_hvac=Q_HVAC,
                    t_end_s=6 * 3600.0, dt_out_s=60.0, W0=W0, loads=all_load_terms())


# --------------------------------------------------------------------------
# Humidity state
# --------------------------------------------------------------------------

def test_without_humidity_columns_are_nan(coach):
    df = simulate(coach, 45.0, 45.0, 5000.0, 20000.0, t_end_s=3600.0, dt_out_s=60.0)
    assert df["W_kg_kg"].isna().all() and df["RH"].isna().all()
    assert list(df.columns[:2]) == ["t_s", "T_C"]


def test_loads_require_w0(coach):
    with pytest.raises(ValueError, match="W0"):
        simulate(coach, 25.0, 25.0, 0.0, 0.0, 3600.0, 60.0, loads=all_load_terms())


def test_humidity_requires_air_volume():
    no_volume = SingleZoneCoach("no volume", A, K, C)
    with pytest.raises(ValueError, match="air_volume_m3"):
        simulate(no_volume, 25.0, 25.0, 0.0, 0.0, 3600.0, 60.0, W0=0.01)


def test_dry_air_mass(coach):
    W0 = 0.008
    df = simulate(coach, 25.0, 25.0, 0.0, 0.0, 600.0, 60.0, W0=W0)
    assert df.attrs["M_da_kg"] == pytest.approx(V / psychro.specific_volume(25.0, W0))


def test_passenger_moisture_analytical(coach):
    # Only passenger water: dW/dt = m_w / M_da (constant) -> W(t) = W0 + m_w t / M_da
    W0 = 0.008
    term = LoadTerm("pax", lambda t, T, W: loads.passengers(N_PAX, 0.0, Q_PAX_LAT))
    df = simulate(coach, 25.0, 25.0, 0.0, 0.0, 600.0, 10.0, W0=W0, loads=[term])
    m_w = N_PAX * Q_PAX_LAT / 2_501_000.0
    W_exact = W0 + m_w * df["t_s"] / df.attrs["M_da_kg"]
    assert np.max(np.abs(df["W_kg_kg"] - W_exact)) < 1e-9


def test_fresh_air_moisture_analytical(coach):
    # Only fresh air: M_da dW/dt = m_da_f (W_out - W)
    #   -> W(t) = W_out + (W0 - W_out) exp(-t / tau_w),  tau_w = M_da / m_da_f
    W0 = 0.005
    W_out = psychro.humidity_ratio(T_OUT, RH_OUT)
    m_da_f = V_FRESH / 3600.0 / psychro.specific_volume(T_OUT, W_out)

    def fresh(t, T, W):
        q_sens, _, m_w = loads.fresh_air(V_FRESH, T_OUT, RH_OUT, T, W)
        return q_sens, m_w

    df = simulate(coach, T_OUT, T_OUT, 0.0, 0.0, 3600.0, 10.0, W0=W0,
                  loads=[LoadTerm("fresh_air", fresh)])
    tau_w = df.attrs["M_da_kg"] / m_da_f
    W_exact = W_out + (W0 - W_out) * np.exp(-df["t_s"] / tau_w)
    assert np.max(np.abs(df["W_kg_kg"] - W_exact)) < 1e-7
    assert df["RH"].iloc[-1] == pytest.approx(
        psychro.rh_from_humidity_ratio(df["T_C"].iloc[-1], df["W_kg_kg"].iloc[-1]))


# --------------------------------------------------------------------------
# Balances with all loads
# --------------------------------------------------------------------------

def test_moisture_balance_closes(combined):
    m_cols = ["m_load_passengers_kg_s", "m_load_fresh_air_kg_s", "m_cond_kg_s"]
    water_in = integral(combined, m_cols, sign={"m_cond_kg_s": -1.0})
    stored = combined.attrs["M_da_kg"] * (combined["W_kg_kg"].iloc[-1] - combined["W_kg_kg"].iloc[0])
    assert abs(stored) > 1e-3 * combined.attrs["M_da_kg"]  # a real change, not ~0
    assert water_in == pytest.approx(stored, rel=1e-3)


def test_combined_energy_balance_closes(combined, coach):
    q_cols = ["Q_env_W", "Q_int_W", "Q_hvac_W", "Q_load_solar_glazing_W",
              "Q_load_solar_opaque_W", "Q_load_passengers_W", "Q_load_fresh_air_W"]
    heat_in = integral(combined, q_cols, sign={"Q_hvac_W": -1.0})
    stored = coach.heat_capacity_J_K * (combined["T_C"].iloc[-1] - combined["T_C"].iloc[0])
    assert abs(stored) > 1e6  # T changed by several K
    assert heat_in == pytest.approx(stored, rel=1e-3)


def test_every_load_is_active(combined):
    for col in ["Q_load_solar_glazing_W", "Q_load_solar_opaque_W", "Q_load_passengers_W",
                "Q_load_fresh_air_W", "m_load_passengers_kg_s", "m_load_fresh_air_kg_s"]:
        assert combined[col].abs().max() > 0, col
    assert (combined["m_cond_kg_s"] == 0).all()


# --------------------------------------------------------------------------
# Reference coach loader
# --------------------------------------------------------------------------

def test_reference_coach_refuses_missing_evidence():
    with pytest.raises(MissingEvidenceError) as exc:
        load_coach_definition()
    expected = [f"E-0{i}" for i in range(34, 42)]
    assert exc.value.ids == expected
    for ref in expected:
        assert ref in str(exc.value)
    assert "E-002" not in exc.value.ids      # k-value placeholder has a value


def _complete_files(tmp_path: Path, yaml_overrides=None, unit_overrides=None):
    rows = {  # id: (value, unit) -- maths-check numbers only
        "E-901": ("1.2", "W/m2K"), "E-902": ("24", "m"), "E-903": ("2.8", "m"),
        "E-904": ("2.2", "m"), "E-905": ("300", "m2"), "E-906": ("12", "m2"),
        "E-907": ("80", "-"), "E-908": ("5e6", "J/K"), "E-909": ("2000", "W"),
    }
    for ref, unit in (unit_overrides or {}).items():
        rows[ref] = (rows[ref][0], unit)
    csv_path = tmp_path / "evidence.csv"
    csv_path.write_text("\n".join(
        [",".join(COLUMNS)]
        + [f"{ref},x,x,{val},,{unit},test,,assumption," for ref, (val, unit) in rows.items()]
    ) + "\n", encoding="utf-8")
    fields = {
        "k_value_W_m2K": "E-901", "length_m": "E-902", "width_m": "E-903",
        "height_m": "E-904", "envelope_area_Ae_m2": "E-905",
        "glazing_area_per_side_m2": "E-906", "seats": "E-907",
        "heat_capacity_J_K": "E-908", "equipment_load_W": "E-909",
    }
    fields.update(yaml_overrides or {})
    yaml_path = tmp_path / "coach.yaml"
    yaml_path.write_text("name: test coach\n" + "".join(f"{k}: {v}\n" for k, v in fields.items()))
    return yaml_path, Registry(csv_path)


def test_loader_resolves_complete_definition(tmp_path):
    yaml_path, reg = _complete_files(tmp_path)
    d = load_coach_definition(yaml_path, reg)
    assert isinstance(d, CoachDefinition)
    assert d.seats == 80 and d.evidence_ids["seats"] == "E-907"
    model = d.thermal_model()
    assert model.UA == pytest.approx(1.2 * 300)
    assert model.air_volume_m3 == pytest.approx(24 * 2.8 * 2.2)


def test_loader_rejects_numbers_instead_of_ids(tmp_path):
    yaml_path, reg = _complete_files(tmp_path, yaml_overrides={"seats": "80"})
    with pytest.raises(EvidenceError, match="seats: .* must be an evidence ID"):
        load_coach_definition(yaml_path, reg)


def test_loader_rejects_unit_mismatch(tmp_path):
    yaml_path, reg = _complete_files(tmp_path, unit_overrides={"E-905": "m"})
    with pytest.raises(EvidenceError, match="E-905 has unit 'm', expected 'm2'"):
        load_coach_definition(yaml_path, reg)
