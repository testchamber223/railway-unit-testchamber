from pathlib import Path

import pytest

from evidence import registry
from evidence.registry import COLUMNS, EvidenceError, Registry, Status

HEADER = ",".join(COLUMNS)


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
        "E-001,A,a,1,W,Src,p1,standard,",
        "E-001,B,b,2,W,Src,p2,standard,",
    )
    with pytest.raises(EvidenceError, match="duplicate id E-001"):
        Registry(path)


def test_bad_status_fails(tmp_path):
    path = write_csv(tmp_path, "E-001,A,a,1,W,Src,p1,guess,")
    with pytest.raises(EvidenceError, match="status"):
        Registry(path)


def test_empty_source_fails_unless_assumption(tmp_path):
    with pytest.raises(EvidenceError, match="empty source"):
        Registry(write_csv(tmp_path, "E-001,A,a,1,W,,,datasheet,"))

    reg = Registry(write_csv(tmp_path, "E-001,A,a,1,W,,,assumption,"))
    assert reg.get("E-001").record.status is Status.ASSUMPTION


def test_list_assumptions_returns_e002():
    assert [r.id for r in registry.list_assumptions()] == ["E-002"]


def test_unknown_id_raises():
    with pytest.raises(KeyError, match="E-999"):
        registry.get("E-999")
