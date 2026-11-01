"""CFD case files (ansys/cases/<case_id>/case.yaml): schema, loader, results.

A case file records WHAT the Fluent team runs, so that CFD results can be
traced like any other evidence:

    case_id, description, purpose, fluent_version, status (planned | running | done)
    geometry            name -> {value: ref, unit}
    parameters          case-level user-defined values; runs may override them
    runs                list of {name, parameters} (e.g. one per Rayleigh number)
    unit_card           optional unit id for card:<field> references
    solver              models, schemes, convergence criteria
    boundary_conditions list of {name, type, value: ref, unit}
    materials           list of {name, value: ref, unit}
    meshes              list of {name, cell_count, h, h_unit}
    results             path pattern relative to the case folder, e.g.
                        "results/{run}_{mesh}.csv" -- one CSV per run and mesh
    results_format      description of the CSV columns

References are the same as in procedure files: "E-xxx", "card:<field>",
"param:<name>" or {user_defined: x, unit: u}. Nothing here contains CFD
results: those are the CSV files Fluent exports (only CSVs and case.yaml go
into git; cases, data and meshes are ignored).

Loader rules (``load_case``):
    * unknown keys, unknown evidence IDs, plot-only values, unit mismatches,
      undeclared parameters and unknown card fields are always refused;
    * empty evidence values are OPEN ITEMS for a planned case and refused
      for a running or done case;
    * a done case must list meshes and have a result file for every run and
      mesh, otherwise it is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from core.procedures import Ref, UserValue
from core.unit import CARD_FIELDS, load_unit_card
from evidence import registry as _evidence
from evidence.registry import EvidenceError, Registry

CASES_DIR = Path(__file__).parents[1] / "ansys" / "cases"


class CaseError(ValueError):
    def __init__(self, case_id: str, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"CFD case {case_id!r} is invalid:\n"
                         + "\n".join(f"  - {p}" for p in problems))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantity(_Strict):
    value: Ref
    unit: str
    note: str = ""


class BoundaryCondition(_Strict):
    name: str
    type: str                       # e.g. wall_temperature, adiabatic, velocity_inlet
    value: Ref | None = None        # None for e.g. adiabatic or symmetry
    unit: str = "-"
    note: str = ""


class Material(_Strict):
    name: str
    value: Ref
    unit: str
    note: str = ""


class Mesh(_Strict):
    name: str
    cell_count: int
    h: float                        # representative cell size
    h_unit: Literal["m", "-"] = "m"  # "-" = normalised (e.g. h / L)
    note: str = ""


class SolverSettings(_Strict):
    models: list[str]
    schemes: dict[str, str] = {}
    convergence: dict[str, float] = {}


class Run(_Strict):
    name: str
    parameters: dict[str, UserValue] = {}


class CaseFile(_Strict):
    case_id: str
    description: str
    purpose: str
    fluent_version: str
    status: Literal["planned", "running", "done"]
    geometry: dict[str, Quantity]
    parameters: dict[str, UserValue] = {}
    runs: list[Run] = [Run(name="base")]
    unit_card: str | None = None
    solver: SolverSettings
    boundary_conditions: list[BoundaryCondition]
    materials: list[Material] = []
    meshes: list[Mesh]
    results: str = "results/{run}_{mesh}.csv"
    results_format: str = ""


@dataclass
class CfdCase:
    """A validated case: resolved inputs per run, open items, result paths."""

    file: CaseFile
    folder: Path
    resolved: dict[str, dict[str, float | None]]     # run -> input name -> SI value
    open_items: list[str] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return self.file.case_id

    def result_path(self, run: str, mesh: str) -> Path:
        return self.folder / self.file.results.format(run=run, mesh=mesh)

    def missing_results(self) -> list[str]:
        return [self.result_path(r.name, m.name).relative_to(self.folder).as_posix()
                for r in self.file.runs for m in self.file.meshes
                if not self.result_path(r.name, m.name).exists()]

    def read_results(self, run: str, mesh: str) -> pd.DataFrame:
        path = self.result_path(run, mesh)
        if not path.exists():
            raise FileNotFoundError(f"{self.case_id}: no results for run {run!r}, mesh {mesh!r} "
                                    f"({path.relative_to(self.folder)})")
        return pd.read_csv(path)


def _resolve(ref, unit: str, where: str, params: dict[str, UserValue], card, reg: Registry,
             status: str, problems: list[str], open_items: list[str], used: set[str]):
    if isinstance(ref, UserValue):
        if ref.unit != unit:
            problems.append(f"{where}: user value has unit {ref.unit!r}, expected {unit!r}")
            return None
        return ref.user_defined
    if ref.startswith("param:"):
        name = ref[6:]
        if name not in params:
            problems.append(f"{where}: parameter '{name}' is not declared")
            return None
        if params[name].unit != unit:
            problems.append(f"{where}: parameter '{name}' has unit {params[name].unit!r}, "
                            f"expected {unit!r}")
            return None
        return params[name].user_defined
    if ref.startswith("card:"):
        name = ref[5:]
        if card is None:
            problems.append(f"{where}: '{ref}' used but the case names no unit_card")
            return None
        if name not in CARD_FIELDS:
            problems.append(f"{where}: unknown unit-card field '{name}'")
            return None
        if CARD_FIELDS[name][0] != unit:
            problems.append(f"{where}: card:{name} is in {CARD_FIELDS[name][0]!r}, "
                            f"expected {unit!r}")
            return None
        value = card.si(name)
        if value is None:
            msg = f"{where}: unit-card field {name} of {card.name} is empty"
            (open_items if status == "planned" else problems).append(msg)
        return value
    try:
        rec = reg.get(ref).record
    except KeyError:
        problems.append(f"{where}: unknown evidence ID {ref}")
        return None
    except EvidenceError as exc:
        problems.append(f"{where}: {exc}")
        return None
    if rec.unit != unit:
        problems.append(f"{where}: {ref} has unit {rec.unit!r}, expected {unit!r}")
        return None
    used.add(ref)
    if rec.value is None:
        msg = f"{where}: {ref} ({rec.name}) is empty (status {rec.status})"
        (open_items if status == "planned" else problems).append(msg)
        return None
    if not isinstance(rec.value, float):
        problems.append(f"{where}: {ref} is not a number")
        return None
    return rec.value


def load_case(path_or_id: Path | str, evidence: Registry | None = None) -> CfdCase:
    """Load and check ansys/cases/<case_id>/case.yaml (or a path to a case.yaml)."""
    path = Path(path_or_id)
    if path.suffix != ".yaml":
        path = (path if path.is_dir() else CASES_DIR / str(path_or_id)) / "case.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        cf = CaseFile.model_validate(data)
    except ValidationError as exc:
        problems = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"])
            problems.append(f"unknown key '{loc}'" if e["type"] == "extra_forbidden"
                            else f"{loc}: {e['msg']}")
        raise CaseError(data.get("case_id", path.parent.name), problems) from None

    reg = evidence if evidence is not None else Registry()
    problems: list[str] = []
    open_items: list[str] = []
    used: set[str] = set()
    card = None
    if cf.unit_card:
        try:
            card = load_unit_card(cf.unit_card)
        except FileNotFoundError:
            problems.append(f"unit_card {cf.unit_card!r} not found")

    names = [m.name for m in cf.meshes]
    if len(set(names)) != len(names):
        problems.append("mesh names must be unique")
    for m in cf.meshes:
        if m.cell_count <= 0 or m.h <= 0:
            problems.append(f"mesh {m.name}: cell_count and h must be > 0")

    resolved: dict[str, dict[str, float | None]] = {}
    for run in cf.runs:
        params = {**cf.parameters, **run.parameters}
        values: dict[str, float | None] = {}
        for name, q in cf.geometry.items():
            values[f"geometry.{name}"] = _resolve(q.value, q.unit, f"{run.name}: geometry.{name}",
                                                  params, card, reg, cf.status, problems,
                                                  open_items, used)
        for bc in cf.boundary_conditions:
            if bc.value is not None:
                values[f"bc.{bc.name}"] = _resolve(bc.value, bc.unit, f"{run.name}: bc.{bc.name}",
                                                   params, card, reg, cf.status, problems,
                                                   open_items, used)
        for mat in cf.materials:
            values[f"material.{mat.name}"] = _resolve(mat.value, mat.unit,
                                                      f"{run.name}: material.{mat.name}",
                                                      params, card, reg, cf.status, problems,
                                                      open_items, used)
        resolved[run.name] = values

    case = CfdCase(cf, path.parent, resolved, sorted(set(open_items)), sorted(used))
    if cf.status == "done":
        if not cf.meshes:
            problems.append("status is 'done' but no meshes are listed")
        missing = case.missing_results()
        if missing:
            problems.append("status is 'done' but result files are missing: " + ", ".join(missing))
    if problems:
        raise CaseError(cf.case_id, problems)
    return case


def list_cases() -> list[str]:
    return sorted(p.parent.name for p in CASES_DIR.glob("*/case.yaml"))
