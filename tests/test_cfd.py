"""CFD case files (core/cfd_import.py), Case 0 inputs and comparison.

No real CFD results exist yet. Where results are needed, the tests write
MANUFACTURED CSV files (verification-only) into a temporary copy of a case.
"""

import shutil

import numpy as np
import pytest
import yaml

from ansys.case0_inputs import case0_inputs, rayleigh_of
from core.cfd_import import CASES_DIR, CaseError, list_cases, load_case
from evidence.registry import Registry
from tests.conftest import verification_registry
from validation.case0_compare import compare_case0

CASES = ["case0_devahldavis", "case1_chamber_section", "case2_coach_bay"]


def test_planned_cases_load_with_their_open_items():
    assert list_cases() == CASES
    for c in CASES:
        case = load_case(c)
        assert case.file.status == "planned"
        assert case.missing_results()             # no CFD results yet
    assert load_case("case0_devahldavis").open_items == []
    open2 = " ".join(load_case("case2_coach_bay").open_items)
    for eid in ("E-026", "E-035", "E-036", "E-077"):
        assert eid in open2
    assert "E-094" in " ".join(load_case("case1_chamber_section").open_items)


def test_case_boundary_conditions_resolve_through_the_registry():
    case = load_case("case2_coach_bay", verification_registry())
    base = case.resolved["base"]
    assert base["bc.outside_temperature"] == 45.0                      # E-022
    assert base["bc.unit_supply_airflow"] == pytest.approx(700 / 3600)  # DHF14 card
    assert base["geometry.width"] == 2.8                               # E-035 stand-in
    assert set(case.evidence_used) >= {"E-002", "E-022", "E-026", "E-035", "E-036", "E-077"}


# --------------------------------------------------------------------------
# Loader refusals
# --------------------------------------------------------------------------

@pytest.fixture
def case0_copy(tmp_path):
    dst = tmp_path / "case0"
    shutil.copytree(CASES_DIR / "case0_devahldavis", dst)
    return dst


def _edit(folder, fn):
    path = folder / "case.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_done_case_without_results_is_refused(case0_copy):
    _edit(case0_copy, lambda d: d.update(status="done"))
    with pytest.raises(CaseError, match="result files are missing: results/Ra1e3_coarse.csv"):
        load_case(case0_copy)


def test_done_case_without_meshes_is_refused(case0_copy):
    _edit(case0_copy, lambda d: d.update(status="done", meshes=[]))
    with pytest.raises(CaseError, match="no meshes"):
        load_case(case0_copy)


@pytest.mark.parametrize("edit,message", [
    (lambda d: d.update(colour="red"), "unknown key 'colour'"),
    (lambda d: d["boundary_conditions"][4].update(value="E-999"), "unknown evidence ID E-999"),
    (lambda d: d["boundary_conditions"][4].update(unit="degC"),
     "E-122 has unit 'm/s2', expected 'degC'"),
    (lambda d: d["boundary_conditions"][0].update(value="param:T_hotter"),
     "parameter 'T_hotter' is not declared"),
    (lambda d: d["boundary_conditions"][0].update(value=25.0), "bare number"),
    (lambda d: d["boundary_conditions"][0].update(value="E-051", unit="degC"), "plot-only"),
])
def test_invalid_case_files_are_refused(case0_copy, edit, message):
    _edit(case0_copy, edit)
    with pytest.raises(CaseError, match=message):
        load_case(case0_copy)


def test_empty_evidence_is_refused_once_running(tmp_path):
    dst = tmp_path / "case2"
    shutil.copytree(CASES_DIR / "case2_coach_bay", dst)
    _edit(dst, lambda d: d.update(status="running"))
    with pytest.raises(CaseError, match="E-077 .* is empty"):
        load_case(dst)


# --------------------------------------------------------------------------
# Case 0 inputs
# --------------------------------------------------------------------------

def test_case0_inputs_match_rayleigh_exactly_and_boussinesq():
    df = case0_inputs()
    for _, row in df.iterrows():
        assert rayleigh_of(row) == pytest.approx(row["Ra"], rel=1e-12)
        assert row["beta_dT"] < 0.1                                     # E-123
    assert abs(df["Pr_deviation"].iloc[0]) < 0.01                       # vs E-125


def test_case0_inputs_refuse_boussinesq_violation():
    with pytest.raises(ValueError, match="Boussinesq"):
        case0_inputs(dT_K=40.0)


def test_case0_yaml_matches_case0_inputs():
    case = load_case("case0_devahldavis")
    df = case0_inputs().set_index("run")
    for run, values in case.resolved.items():
        assert values["geometry.side_length"] == pytest.approx(df.loc[run, "L_m"], rel=1e-12)
        assert values["bc.hot_wall"] == df.loc[run, "T_hot_C"]
        assert values["material.thermal_conductivity"] == pytest.approx(
            df.loc[run, "k_W_mK"], rel=1e-12)


# --------------------------------------------------------------------------
# Case 0 comparison with MANUFACTURED results (verification-only)
# --------------------------------------------------------------------------

NU0 = {"Ra1e3": 1.1, "Ra1e4": 2.2, "Ra1e5": 4.5, "Ra1e6": 8.8}   # manufactured, NOT benchmark


def _manufacture_results(folder, C=40.0):
    case = load_case(folder)
    for run in case.file.runs:
        v = case.resolved[run.name]
        k, dT, L = (v["material.thermal_conductivity"], v["bc.hot_wall"] - v["bc.cold_wall"],
                    v["geometry.side_length"])
        for m in case.file.meshes:
            nu = NU0[run.name] + C * m.h ** 2               # manufactured 2nd-order error
            y = np.linspace(0, L, 41)
            q = np.full_like(y, nu * k * dT / L)            # uniform flux -> exact average
            path = case.result_path(run.name, m.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame({"position_m": y, "wall_heat_flux_W_m2": q}).to_csv(path, index=False)


def test_case0_compare_skips_while_results_are_missing():
    for rc in compare_case0():
        assert rc.status == "skipped"
        assert "results missing" in rc.messages[0]


def test_case0_compare_with_manufactured_results(case0_copy):
    _manufacture_results(case0_copy)
    _edit(case0_copy, lambda d: d.update(status="done"))
    # benchmarks still empty -> skipped with a clear message
    out = compare_case0(load_case(case0_copy))
    for rc in out:
        assert rc.gci.p == pytest.approx(2.0, abs=1e-6)
        assert rc.gci.f_extrapolated == pytest.approx(NU0[rc.run], rel=1e-9)
        assert rc.status == "skipped" and "benchmark E-1" in rc.messages[-1]
    # stand-in benchmark 0.5 % above the manufactured value, tolerance 1 % -> pass
    bench = {"E-117": NU0["Ra1e3"] * 1.005, "E-118": NU0["Ra1e4"] * 1.005,
             "E-119": NU0["Ra1e5"] * 1.005, "E-120": NU0["Ra1e6"] * 1.005}
    reg = Registry().with_overrides({**bench, "E-121": 0.01}, label="verification-only")
    assert {rc.status for rc in compare_case0(load_case(case0_copy, reg), reg)} == {"pass"}
    reg = Registry().with_overrides({**bench, "E-121": 0.001}, label="verification-only")
    assert {rc.status for rc in compare_case0(load_case(case0_copy, reg), reg)} == {"fail"}
