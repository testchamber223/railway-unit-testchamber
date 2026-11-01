"""Verification of the procedure files and the runner (core.procedures).

VERIFICATION-ONLY PARAMETERS: empty evidence rows are filled from
tests/data/verification_overrides.yaml (label "verification-only"); the coach
and unit are verification-only too. Maths checks, not real data.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from tests.conftest import verification_registry
from scipy.integrate import simpson

from core import loads
from core.coach import SingleZoneCoach, time_constant_s
from core.procedures import (PROCEDURES_DIR, ProcedureError, k_value_test, load_procedure,
                             parse_procedure, run_procedure)

ALL = ["k_value_uic553_1", "unit_capacity_rating", "pulldown_cooling",
       "comfort_steady_zone1", "comfort_steady_zone2", "user_custom_example"]


def _data(name):
    return yaml.safe_load((PROCEDURES_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_all_procedure_files_parse():
    for name in ALL:
        assert load_procedure(name).file.name


# --------------------------------------------------------------------------
# Ported k-value test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("Q,hours", [(9000.0, None), (9000.0, 30.0), (7000.0, None)])
def test_ported_k_test_matches_old_function(vreg, Q, hours):
    coach = SingleZoneCoach("verification", 300.0, 1.2, 5e6)
    tau = time_constant_s(coach)
    t_end = 10 * tau if hours is None else hours * 3600.0
    old = k_value_test(coach, 5.0, Q, t_end)            # chamber at E-010 = 5 degC
    new = run_procedure("k_value_uic553_1", coach=coach, evidence=vreg,
                        parameters={"heater_power": Q, "test_duration": t_end / 3600.0})
    assert new.measurements["k_value"].value == pytest.approx(old.k_measured_W_m2K, rel=1e-9)
    assert new.measurements["dT_cabin_chamber_mean"].value == pytest.approx(old.dT_K, rel=1e-9)
    assert new.measurements["cabin_T_mean"].value == pytest.approx(old.T_in_C, rel=1e-9)
    assert [c.passed for c in new.criteria] == [True, True] or old.condition_violations


def test_ported_k_test_reports_not_steady(vreg):
    coach = SingleZoneCoach("verification", 300.0, 1.2, 5e6)
    res = run_procedure("k_value_uic553_1", coach=coach, evidence=vreg,
                        parameters={"test_duration": 5.0})
    assert any("not steady" in v for v in res.validity)
    assert not res.passed


# --------------------------------------------------------------------------
# Validation: unknown keys, missing / empty / plot-only evidence, bare numbers
# --------------------------------------------------------------------------

def test_unknown_keys_are_rejected():
    data = _data("comfort_steady_zone2")
    data["colour"] = "red"
    data["steps"][0]["set_conditions"]["chamber_temp"] = "E-023"
    with pytest.raises(ProcedureError) as exc:
        parse_procedure(data)
    msg = str(exc.value)
    assert "unknown key 'colour'" in msg
    assert "unknown key 'steps[0].set_conditions.chamber_temp'" in msg


def test_unknown_step_type_is_rejected():
    data = _data("comfort_steady_zone2")
    data["steps"].append({"wait_forever": {}})
    with pytest.raises(ProcedureError, match="unknown key 'steps\\[3\\].wait_forever'"):
        parse_procedure(data)


def test_missing_evidence_id_is_rejected(vreg, vcoach, vcard):
    data = _data("comfort_steady_zone2")
    data["steps"][0]["set_conditions"]["chamber_T"] = "E-999"
    with pytest.raises(ProcedureError, match="steps\\[0\\].set_conditions.chamber_T: "
                                             "unknown evidence ID E-999"):
        run_procedure(parse_procedure(data), coach=vcoach, unit_card=vcard, evidence=vreg)


def test_empty_assumption_in_acceptance_criterion_is_rejected(vcoach, vcard):
    # everything filled except E-070, the pull-down acceptance limit
    values = {k: v for k, v in yaml.safe_load(
        (Path(__file__).parent / "data" / "verification_overrides.yaml").read_text()).items()
        if k != "E-070"}
    from evidence.registry import Registry
    reg = Registry().with_overrides(values, label="verification-only")
    with pytest.raises(ProcedureError) as exc:
        run_procedure("pulldown_cooling", coach=vcoach, unit_card=vcard, evidence=reg)
    assert exc.value.problems == [
        "acceptance criterion 'time to reach the setpoint' (at_most): E-070 (Pull-down test "
        "maximum time to reach setpoint) has an EMPTY value (status assumption); fill it in "
        "evidence.csv"]


def test_unsourced_registry_lists_every_empty_value(vcoach, vcard):
    with pytest.raises(ProcedureError) as exc:
        run_procedure("comfort_steady_zone1", coach=vcoach, unit_card=vcard)
    ids = set(re.findall(r"E-\d{3}", " ".join(exc.value.problems)))
    assert {"E-071", "E-072", "E-073", "E-074", "E-030", "E-031", "E-029"} <= ids


@pytest.mark.parametrize("value,message", [
    (45.0, "bare number 45.0"),
    ("forty-five", "is not a reference"),
    ("E-051", "plot-only"),
])
def test_bad_references_are_rejected(vreg, vcoach, vcard, value, message):
    data = _data("comfort_steady_zone2")
    data["steps"][0]["set_conditions"]["chamber_T"] = value
    with pytest.raises(ProcedureError, match=message):
        run_procedure(parse_procedure(data), coach=vcoach, unit_card=vcard, evidence=vreg)


def test_unit_mismatch_and_undeclared_parameter_are_rejected(vreg, vcoach, vcard):
    data = _data("comfort_steady_zone2")
    data["steps"][0]["set_conditions"]["chamber_T"] = "E-016"        # W/m2, not degC
    data["steps"][1]["hold_until_stable"]["max_duration"] = "param:nope"
    with pytest.raises(ProcedureError) as exc:
        run_procedure(parse_procedure(data), coach=vcoach, unit_card=vcard, evidence=vreg)
    msg = str(exc.value)
    assert "E-016 has unit 'W/m2', expected 'degC'" in msg
    assert "parameter 'nope' is not declared" in msg


# --------------------------------------------------------------------------
# TestResult provenance
# --------------------------------------------------------------------------

def test_result_lists_every_evidence_id_used(vreg, vcoach, vcard):
    res = run_procedure("comfort_steady_zone1", coach=vcoach, unit_card=vcard, evidence=vreg,
                        sensors="evidence")
    in_file = set(re.findall(r"E-\d{3}", (PROCEDURES_DIR / "comfort_steady_zone1.yaml")
                             .read_text(encoding="utf-8").split("name:", 1)[1]))
    facility = {"E-061", "E-062", "E-063", "E-064", "E-065", "E-066", "E-067"}
    sensors = {"E-058", "E-059", "E-060"}
    coach = {"E-002", "E-034", "E-035", "E-036", "E-037", "E-038", "E-040"}
    curve = {"E-042", "E-043", "E-044", "E-045", "E-046", "E-047"}
    assert in_file | facility | sensors | coach | curve <= set(res.evidence_ids)
    assert "E-041" not in res.evidence_ids         # equipment load is not applied
    for use in res.evidence_used:
        assert use.used_in, use.id


def test_result_flags_assumptions_overrides_and_user_values(vreg, vcoach, vcard):
    res = run_procedure("comfort_steady_zone1", coach=vcoach, unit_card=vcard, evidence=vreg)
    flagged = {u.id for u in res.assumptions}
    for use in res.evidence_used:
        if use.status in ("assumption", "plot-only assumption") or use.override:
            assert use.id in flagged
    assert {"E-002", "E-071", "E-072"} <= flagged                    # assumption rows
    assert all(u.override == "verification-only" for u in res.assumptions if u.id != "E-002")
    assert "E-024" not in flagged and "E-022" not in flagged         # standard rows
    assert [u.name for u in res.user_defined] == ["max_settle"]
    text = res.summary()
    assert "verification-only" in text and "max_settle" in text


def test_card_fields_are_listed_with_their_status(vreg, vcard):
    res = run_procedure("unit_capacity_rating", unit_card=vcard, evidence=vreg)
    card_uses = [u for u in res.evidence_used if u.id.startswith("card:")]
    assert {u.id for u in card_uses} >= {"card:verification_unit.rated_total_cooling_capacity",
                                         "card:verification_unit.rating_indoor_wet_bulb"}
    assert all(u.is_assumption for u in card_uses)        # verification card = assumptions


# --------------------------------------------------------------------------
# Procedures run
# --------------------------------------------------------------------------

def test_unit_capacity_rating_with_ideal_sensors_passes(vreg, vcard):
    res = run_procedure("unit_capacity_rating", unit_card=vcard, evidence=vreg)
    gross = res.measurements["capacity_gross_W"]
    assert gross.value == pytest.approx(30000.0, rel=1e-6)
    assert res.passed and res.criteria[0].conclusive


def test_custom_procedure_runs_with_user_values(vreg, vcoach, vcard):
    base = run_procedure("user_custom_example", coach=vcoach, unit_card=vcard, evidence=vreg)
    assert base.measurements["chamber_T_mean"].value == pytest.approx(38.0, abs=1e-6)
    assert len(base.user_defined) == 16
    assert not any(u.id in {"E-022", "E-023", "E-013"} for u in base.evidence_used)
    hotter = run_procedure("user_custom_example", coach=vcoach, unit_card=vcard, evidence=vreg,
                           parameters={"outside_T": 41.0})
    assert hotter.measurements["chamber_T_mean"].value == pytest.approx(41.0, abs=1e-6)
    assert hotter.measurements["cabin_T_mean"].value > base.measurements["cabin_T_mean"].value
    assert next(u for u in hotter.user_defined if u.name == "outside_T").value == 41.0


def test_unknown_parameter_override_is_rejected(vreg, vcoach, vcard):
    with pytest.raises(ProcedureError, match="'outside_temp' given but not declared"):
        run_procedure("user_custom_example", coach=vcoach, unit_card=vcard, evidence=vreg,
                      parameters={"outside_temp": 41.0})


@pytest.fixture(scope="module")
def pulldown(vcoach, vcard):
    # Setpoint 32 degC (verification) so that the verification unit reaches it.
    reg = verification_registry(E_068=32.0)
    return run_procedure("pulldown_cooling", coach=vcoach, unit_card=vcard, evidence=reg,
                         sensors="evidence", seed=2)


def test_pulldown_reports_time_to_setpoint_with_uncertainty(pulldown):
    m = pulldown.measurements["time_to_setpoint_h"]
    assert 0 < m.value < 6
    assert m.uncertainty > 0
    assert abs(m.value - m.true_value) <= m.uncertainty
    assert [s.step for s in pulldown.segments] == ["steps[1].hold_until_stable",
                                                   "steps[3].run_until"]


def test_pulldown_whole_system_energy_balance_closes(pulldown, vcoach):
    """Total enthalpy over soak + pull-down: inflow = C dT + M_da h_g(T) dW."""
    C = vcoach.heat_capacity_J_K
    inflow_total = stored_total = 0.0
    for seg in pulldown.segments:
        df = seg.df
        inflow = df["Q_env_W"].to_numpy().copy()
        for col in df.columns:
            if col.startswith("Q_load_solar"):
                inflow += df[col].to_numpy()
        if "unit_Q_supply_total_W" in df:
            inflow += df["unit_Q_supply_total_W"].to_numpy()
        T, W = df["T_C"].to_numpy(), df["W_kg_kg"].to_numpy()
        h_g = np.array([loads.vapour_enthalpy(x) for x in T])
        inflow_total += simpson(inflow, x=df["t_s"].to_numpy())
        stored_total += C * (T[-1] - T[0]) + df.attrs["M_da_kg"] * np.sum(
            0.5 * (h_g[1:] + h_g[:-1]) * np.diff(W))
    assert abs(stored_total) > 1e7
    assert inflow_total == pytest.approx(stored_total, rel=1e-3)


def test_pulldown_with_unreachable_setpoint_fails_validly(vreg, vcoach, vcard):
    res = run_procedure("pulldown_cooling", coach=vcoach, unit_card=vcard, evidence=vreg,
                        parameters={"max_pulldown": 2.0})
    assert not res.passed
    assert any("did not reach" in v for v in res.validity)
