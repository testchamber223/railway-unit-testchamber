"""Evidence registry: loads and validates ``evidence.csv``.

Every physical constant used by ``core`` must be fetched here, e.g.::

    from evidence import registry
    q_pax = registry.get("E-001").value

Validation rules (enforced on load):
    * ``status`` must be one of the ``Status`` values.
    * ``source`` may only be empty when ``status`` is ``assumption``.
    * ``value`` may only be empty when ``status`` is ``assumption``
      (an open question that must be resolved before the value is used).
    * ``id`` values must be unique.
"""

from __future__ import annotations

import csv
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import (BaseModel, ConfigDict, ValidationError, field_validator,
                      model_validator)

DEFAULT_CSV = Path(__file__).with_name("evidence.csv")

COLUMNS = ["id", "name", "symbol", "value", "unit", "source",
           "clause_or_page", "status", "notes"]


class EvidenceError(ValueError):
    """Raised when the evidence file is malformed or inconsistent."""


class Status(StrEnum):
    STANDARD = "standard"
    DATASHEET = "datasheet"
    LITERATURE = "literature"
    CALCULATED = "calculated"
    ASSUMPTION = "assumption"


class EvidenceRecord(BaseModel):
    """One row of ``evidence.csv``."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, allow_inf_nan=False)

    id: str
    name: str
    symbol: str
    value: float | None
    unit: str
    source: str = ""
    clause_or_page: str = ""
    status: Status
    notes: str = ""

    @field_validator("value", mode="before")
    @classmethod
    def _empty_value_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _source_and_value_required(self) -> EvidenceRecord:
        if self.status is not Status.ASSUMPTION:
            if not self.source:
                raise ValueError(
                    f"{self.id}: empty source is only allowed for status 'assumption'"
                )
            if self.value is None:
                raise ValueError(
                    f"{self.id}: empty value is only allowed for status 'assumption'"
                )
        return self


class Evidence(NamedTuple):
    """Result of ``get``: the value, its unit and the full traceable record."""

    value: float | None
    unit: str
    record: EvidenceRecord


class Registry:
    """In-memory, validated view of an evidence CSV file."""

    def __init__(self, path: Path | str = DEFAULT_CSV) -> None:
        self.path = Path(path)
        self._records = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> dict[str, EvidenceRecord]:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != COLUMNS:
                raise EvidenceError(
                    f"{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}"
                )
            records: dict[str, EvidenceRecord] = {}
            for line, row in enumerate(reader, start=2):
                try:
                    rec = EvidenceRecord.model_validate(row)
                except ValidationError as exc:
                    raise EvidenceError(f"{path.name} line {line}: {exc}") from exc
                if rec.id in records:
                    raise EvidenceError(f"{path.name} line {line}: duplicate id {rec.id}")
                records[rec.id] = rec
        return records

    def get(self, evidence_id: str) -> Evidence:
        try:
            rec = self._records[evidence_id]
        except KeyError:
            raise KeyError(f"Unknown evidence id {evidence_id!r}") from None
        return Evidence(rec.value, rec.unit, rec)

    def list_assumptions(self) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.status is Status.ASSUMPTION]

    def records(self) -> list[EvidenceRecord]:
        return list(self._records.values())


_default: Registry | None = None


def _registry() -> Registry:
    global _default
    if _default is None:
        _default = Registry()
    return _default


def get(evidence_id: str) -> Evidence:
    """Return value, unit and full record for ``evidence_id`` from the default CSV."""
    return _registry().get(evidence_id)


def list_assumptions() -> list[EvidenceRecord]:
    """Return all records in the default CSV whose status is ``assumption``."""
    return _registry().list_assumptions()
