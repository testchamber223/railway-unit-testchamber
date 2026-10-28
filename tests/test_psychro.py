import csv
from pathlib import Path

import pytest

from core import psychro
from evidence import registry

CASES_CSV = Path(__file__).parents[1] / "validation" / "psychro_cases.csv"

# Factor converting a value in the CSV's unit to the SI unit returned by core.psychro.
TO_SI = {
    "Pa": 1.0, "kPa": 1e3,
    "kg/kg": 1.0, "g/kg": 1e-3,
    "J/kg": 1.0, "kJ/kg": 1e3,
    "degC": 1.0,
    "-": 1.0, "%": 1e-2,
    "m3/kg": 1.0, "kg/m3": 1.0,
}

TEMPS = [0.0, 20.0, 27.0, 35.0, 45.0]
RHS = [0.30, 0.50, 0.80]
GRID = [(T, RH) for T in TEMPS for RH in RHS]


# --------------------------------------------------------------------------
# Validation cases from validation/psychro_cases.csv
# --------------------------------------------------------------------------

def _load_cases():
    with CASES_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [pytest.param(r, id=r["case_id"]) for r in rows]


def _parse_inputs(text: str) -> dict[str, float]:
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in text.split(";"))}


def _evaluate(quantity: str, inp: dict[str, float]) -> float:
    """Compute ``quantity`` [SI] from a state given as T plus one of W / RH / Twb."""
    T, p = inp["T"], inp.get("p")
    if quantity == "sat_vapor_pressure":
        return psychro.sat_vapor_pressure(T)

    if "W" in inp:
        W = inp["W"]
    elif "RH" in inp:
        W = psychro.humidity_ratio(T, inp["RH"], p)
    else:
        W = psychro.humidity_ratio_from_wet_bulb(T, inp["Twb"], p)
    RH = psychro.rh_from_humidity_ratio(T, W, p)

    return {
        "humidity_ratio": lambda: W,
        "relative_humidity": lambda: RH,
        "enthalpy": lambda: psychro.enthalpy(T, W),
        "dew_point": lambda: psychro.dew_point(T, RH, p),
        "wet_bulb": lambda: psychro.wet_bulb(T, RH, p),
        "specific_volume": lambda: psychro.specific_volume(T, W, p),
        "moist_air_density": lambda: psychro.moist_air_density(T, W, p),
    }[quantity]()


@pytest.mark.parametrize("case", _load_cases())
def test_validation_case(case):
    factor = TO_SI[case["unit"]]
    computed = _evaluate(case["quantity"], _parse_inputs(case["inputs"]))

    if not case["expected"].strip():
        pytest.skip(
            f"{case['case_id']} {case['quantity']}: expected value empty -- fill from "
            f"{case['source']} (model gives {computed / factor:.6g} {case['unit']})"
        )

    expected = float(case["expected"]) * factor
    tol = float(case["tolerance_pct"]) / 100.0
    assert computed == pytest.approx(expected, rel=tol), (
        f"{case['case_id']}: computed {computed / factor:.6g}, expected "
        f"{case['expected']} {case['unit']} (+/-{case['tolerance_pct']} %)"
    )


# --------------------------------------------------------------------------
# Enthalpy vs. ASHRAE formula built from the evidence registry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("T,RH", GRID)
def test_enthalpy_matches_ashrae_formula(T, RH):
    cp_da, h_g0, cp_w = (registry.get(i) for i in ("E-005", "E-006", "E-007"))
    assert (cp_da.unit, h_g0.unit, cp_w.unit) == ("kJ/(kg.K)", "kJ/kg", "kJ/(kg.K)")

    W = psychro.humidity_ratio(T, RH)
    h_formula = (cp_da.value * T + W * (h_g0.value + cp_w.value * T)) * 1e3  # kJ -> J
    assert psychro.enthalpy(T, W) == pytest.approx(h_formula, rel=1e-3)


# --------------------------------------------------------------------------
# Round trips and identities
# --------------------------------------------------------------------------

@pytest.mark.parametrize("T,RH", GRID)
def test_roundtrip_rh_w_rh(T, RH):
    W = psychro.humidity_ratio(T, RH)
    assert psychro.rh_from_humidity_ratio(T, W) == pytest.approx(RH, abs=1e-6)


@pytest.mark.parametrize("T,RH", GRID)
def test_roundtrip_dew_point_saturation(T, RH):
    W = psychro.humidity_ratio(T, RH)
    Tdp = psychro.dew_point(T, RH)
    assert psychro.humidity_ratio(Tdp, 1.0) == pytest.approx(W, abs=1e-6)


@pytest.mark.parametrize("T", TEMPS)
def test_dew_point_at_saturation_equals_dry_bulb(T):
    assert psychro.dew_point(T, 1.0) == pytest.approx(T, abs=0.01)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_rh_as_percent_raises():
    with pytest.raises(ValueError, match="fraction"):
        psychro.humidity_ratio(25.0, 55)


@pytest.mark.parametrize("T", [-40.1, 80.1])
def test_temperature_out_of_range_raises(T):
    with pytest.raises(ValueError, match="valid range"):
        psychro.sat_vapor_pressure(T)


def test_default_pressure_from_registry():
    p = registry.get("E-004").value
    assert psychro.humidity_ratio(25.0, 0.5) == psychro.humidity_ratio(25.0, 0.5, p)


# --------------------------------------------------------------------------
# Saturation and inverse relations (used by core.unit)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("T,RH", GRID)
def test_wet_bulb_from_humidity_ratio_inverts_eq33(T, RH):
    W = psychro.humidity_ratio(T, RH)
    Twb = psychro.wet_bulb_from_humidity_ratio(T, W)
    assert psychro.humidity_ratio_from_wet_bulb(T, Twb) == pytest.approx(W, rel=1e-9)
    assert Twb == pytest.approx(psychro.wet_bulb(T, RH), abs=2e-3)  # psychrolib bisection


@pytest.mark.parametrize("T", [-10.0, 0.0, 12.0, 25.0, 45.0])
def test_sat_temperature_from_enthalpy_round_trip(T):
    h = psychro.sat_air_enthalpy(T)
    assert psychro.sat_temperature_from_enthalpy(h) == pytest.approx(T, abs=1e-8)
    assert psychro.enthalpy(T, psychro.sat_humidity_ratio(T)) == pytest.approx(h, rel=1e-12)


@pytest.mark.parametrize("T,RH", GRID)
def test_dry_bulb_and_humidity_from_enthalpy(T, RH):
    W = psychro.humidity_ratio(T, RH)
    h = psychro.enthalpy(T, W)
    assert psychro.dry_bulb_from_enthalpy(h, W) == pytest.approx(T, abs=1e-9)
    assert psychro.humidity_ratio_from_enthalpy(T, h) == pytest.approx(W, rel=1e-9)


def test_wet_bulb_rejects_supersaturated_air():
    with pytest.raises(ValueError, match="saturation"):
        psychro.wet_bulb_from_humidity_ratio(20.0, 1.5 * psychro.sat_humidity_ratio(20.0))
