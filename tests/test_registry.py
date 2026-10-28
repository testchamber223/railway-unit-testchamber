from pathlib import Path

import pytest

from evidence import registry
from evidence.registry import (COLUMNS, EvidenceError, MissingEvidenceError, Registry,
                               Status)

HEADER = ",".join(COLUMNS)
DOCS = Path(__file__).parents[1] / "docs" / "README.md"

# Row layout: id,name,symbol,value,tolerance,unit,source,clause_or_page,status,notes


def write_csv(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "evidence.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_default_csv_loads():
    reg = Registry()
    assert {"E-001", "E-002", "E-003"} <= {r.id for r in reg.records()}

    e = registry.get("E-001")
    assert e.value == 120
    assert e.unit == "W/person"
    assert e.record.source == "UIC 553"
    assert e.record.status is Status.STANDARD


def test_duplicate_ids_fail(tmp_path):
    path = write_csv(
        tmp_path,
        "E-001,A,a,1,,W,Src,p1,standard,",
        "E-001,B,b,2,,W,Src,p2,standard,",
    )
    with pytest.raises(EvidenceError, match="duplicate id E-001"):
        Registry(path)


def test_bad_status_fails(tmp_path):
    path = write_csv(tmp_path, "E-001,A,a,1,,W,Src,p1,guess,")
    with pytest.raises(EvidenceError, match="status"):
        Registry(path)


def test_empty_source_fails_unless_assumption(tmp_path):
    with pytest.raises(EvidenceError, match="empty source"):
        Registry(write_csv(tmp_path, "E-001,A,a,1,,W,,,datasheet,"))

    reg = Registry(write_csv(tmp_path, "E-001,A,a,1,,W,,,assumption,"))
    assert reg.get("E-001").record.status is Status.ASSUMPTION


def test_empty_value_fails_unless_assumption(tmp_path):
    with pytest.raises(EvidenceError, match="empty value"):
        Registry(write_csv(tmp_path, "E-001,A,a,,,W,Src,p1,standard,"))

    reg = Registry(write_csv(tmp_path, "E-001,A,a,,,-,Src,p1,assumption,"))
    assert reg.get("E-001").value is None


def test_old_column_layout_is_rejected(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_text("id,name,symbol,value,unit,source,clause_or_page,status,notes\n")
    with pytest.raises(EvidenceError, match="expected columns"):
        Registry(path)


def test_text_value_and_tolerance(tmp_path):
    reg = Registry(write_csv(
        tmp_path,
        "E-001,Def,d,outer surface,,-,Src,p1,standard,",
        "E-002,Temp,T,5,2,degC,Src,p2,standard,",
    ))
    assert reg.get("E-001").value == "outer surface"
    assert reg.get("E-002").value == 5.0
    assert reg.get("E-002").record.tolerance == 2.0


def test_tolerance_needs_numeric_value_and_non_negative(tmp_path):
    with pytest.raises(EvidenceError, match="tolerance"):
        Registry(write_csv(tmp_path, "E-001,Def,d,outer,1,-,Src,p1,standard,"))
    with pytest.raises(EvidenceError, match="tolerance"):
        Registry(write_csv(tmp_path, "E-001,T,T,5,-2,degC,Src,p1,standard,"))


def test_non_finite_value_fails(tmp_path):
    with pytest.raises(EvidenceError):
        Registry(write_csv(tmp_path, "E-001,A,a,nan,,W,Src,p1,standard,"))


def test_number_checks_value_and_unit():
    assert registry.number("E-004", "Pa") == 101325.0
    with pytest.raises(EvidenceError, match="unit"):
        registry.number("E-004", "kPa")
    with pytest.raises(MissingEvidenceError, match="empty") as exc:
        registry.number("E-026", "W/person")
    assert exc.value.ids == ["E-026"]
    with pytest.raises(MissingEvidenceError, match="not a number"):
        registry.number("E-008", "-")


def test_list_assumptions_returns_e002():
    assert "E-002" in [r.id for r in registry.list_assumptions()]


def test_e008_is_the_ae_definition():
    e = registry.get("E-008")
    assert e.value == "exterior developed surface Ae incl. roof insulation"
    assert e.record.status is Status.STANDARD
    assert e.record.clause_or_page == "7.2.2"


def test_k_test_conditions_carry_tolerances():
    assert (registry.get("E-009").value, registry.get("E-009").record.tolerance) == (25, 5)
    assert (registry.get("E-010").value, registry.get("E-010").record.tolerance) == (5, 2)


def test_every_open_item_is_listed_in_docs():
    text = DOCS.read_text(encoding="utf-8")
    missing = [r.id for r in registry.list_open_items() if f"| {r.id} " not in text]
    assert not missing, f"open evidence items missing from docs/README.md table: {missing}"


def test_unknown_id_raises():
    with pytest.raises(KeyError, match="E-999"):
        registry.get("E-999")
