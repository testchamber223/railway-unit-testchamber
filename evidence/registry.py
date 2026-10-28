"""Evidence registry: loads and validates ``evidence.csv``.

Every physical constant used by ``core`` must be fetched here, e.g.::

    from evidence import registry
    p = registry.number("E-004", "Pa")     # value as float, unit checked
    rec = registry.get("E-004").record      # full traceable record

Validation rules (enforced on load):
    * ``status`` must be one of the ``Status`` values.
    * ``source`` may only be empty when ``status`` is ``assumption``.
    * ``value`` may only be empty when ``status`` is ``assumption``
      (an open question that must be resolved before the value is used).
    * ``value`` is a number, or text for definitions (e.g. E-008).
    * ``tolerance`` (optional, same unit as ``value``) is a symmetric +/- band.
    * ``id`` values must be unique.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import (BaseModel, ConfigDict, ValidationError, field_validator,
                      model_validator)

DEFAULT_CSV = Path(__file__).with_name("evidence.csv")

COLUMNS = ["id", "name", "symbol", "value", "tolerance", "unit", "source",
           "clause_or_page", "status", "notes"]


class EvidenceError(ValueError):
    """Raised when the evidence file is malformed or inconsistent."""


class MissingEvidenceError(EvidenceError):
    """Raised when evidence needed for a calculation has no usable value.

    Attributes:
        ids: the evidence ids that are missing or unusable.
    """

    def __init__(self, ids: Iterable[str], message: str) -> None:
        self.ids = list(ids)
        super().__init__(message)


class Status(StrEnum):
    STANDARD = "standard"
    DATASHEET = "datasheet"
    LITERATURE = "literature"
    CALCULATED = "calculated"
    ASSUMPTION = "assumption"


def _parse_number_or_text(v: object) -> object:
    if not isinstance(v, str):
        return v
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return v


class EvidenceRecord(BaseModel):
    """One row of ``evidence.csv``."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, allow_inf_nan=False)

    id: str
    name: str
    symbol: str
    value: float | str | None
    tolerance: float | None = None
    unit: str
    source: str = ""
    clause_or_page: str = ""
    status: Status
    notes: str = ""

    @field_validator("value", "tolerance", mode="before")
    @classmethod
    def _parse(cls, v: object) -> object:
        return _parse_number_or_text(v)

    @field_validator("value", mode="after")
    @classmethod
    def _finite(cls, v: object) -> object:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("value must be finite")
        return v

    @model_validator(mode="after")
    def _check(self) -> EvidenceRecord:
        if self.status is not Status.ASSUMPTION:
            if not self.source:
                raise ValueError(
                    f"{self.id}: empty source is only allowed for status 'assumption'"
                )
            if self.value is None:
                raise ValueError(
                    f"{self.id}: empty value is only allowed for status 'assumption'"
                )
        if self.tolerance is not None:
            if self.tolerance < 0:
                raise ValueError(f"{self.id}: tolerance must be >= 0")
            if not isinstance(self.value, float):
                raise ValueError(f"{self.id}: a tolerance needs a numeric value")
        return self

    @property
    def is_open(self) -> bool:
        """True if the record is an assumption or still flagged 'to confirm'."""
        text = f"{self.clause_or_page} {self.notes}".lower()
        return self.status is Status.ASSUMPTION or "confirm" in text


class Evidence(NamedTuple):
    """Result of ``get``: the value, its unit and the full traceable record."""

    value: float | str | None
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

    def number(self, evidence_id: str, unit: str) -> float:
        """Numeric value of ``evidence_id``, checking that its unit is ``unit``.

        Raises:
            MissingEvidenceError: value empty or not a number.
            EvidenceError: unit differs from ``unit``.
        """
        rec = self.get(evidence_id).record
        if not isinstance(rec.value, float):
            state = "empty" if rec.value is None else f"not a number ({rec.value!r})"
            raise MissingEvidenceError(
                [evidence_id], f"{evidence_id} ({rec.name}): value is {state}"
            )
        if rec.unit != unit:
            raise EvidenceError(f"{evidence_id}: unit is {rec.unit!r}, expected {unit!r}")
        return rec.value

    def list_assumptions(self) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.status is Status.ASSUMPTION]

    def list_open_items(self) -> list[EvidenceRecord]:
        """Records still marked assumption or 'to confirm' / 'confirm against the PDF'."""
        return [r for r in self._records.values() if r.is_open]

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


def number(evidence_id: str, unit: str) -> float:
    """Numeric value of ``evidence_id`` from the default CSV, unit-checked."""
    return _registry().number(evidence_id, unit)


def list_assumptions() -> list[EvidenceRecord]:
    """Return all records in the default CSV whose status is ``assumption``."""
    return _registry().list_assumptions()


def list_open_items() -> list[EvidenceRecord]:
    """Return all records in the default CSV that are assumptions or 'to confirm'."""
    return _registry().list_open_items()
