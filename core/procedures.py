"""Test procedures: virtual versions of the type tests run in test facilities.

Two layers:

* ``k_value_test`` -- the original hard-coded k-value test (kept as the
  reference implementation; the procedure file k_value_uic553_1.yaml must
  reproduce it).
* Data-driven procedures: procedures/*.yaml files run by ``run_procedure`` in
  a virtual facility (core.chamber), returning a ``TestResult`` with measured
  values and their sensor uncertainty, pass/fail per criterion, every evidence
  ID used, and every assumption and user-defined value.

Procedure parameters (windows, set conditions, tolerances) come from the
evidence registry or are explicitly marked user-defined.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError, model_validator
from scipy.integrate import trapezoid

from core import loads, psychro
from core.chamber import UnitTestRoom, VehicleChamber
from core.coach import (CoachDefinition, LoadTerm, SingleZoneCoach, UnitOperation,
                        net_sensible_W, simulate)
from core.sensors import Channel, SensorSet, propagate
from core.unit import CARD_FIELDS, CURVE_IDS, DXUnit, UnitCard
from evidence import registry
from evidence.registry import EvidenceError, Registry

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class KValueResult:
    """Outcome of ``k_value_test``.

    Attributes:
        k_measured_W_m2K: k = mean(Q) / (A * mean(T_in - T_out)) over the
            averaging period at the end of the run [W/(m2.K)].
        t_steady_s: first time at which a full steady-state window had been
            completed [s].
        T_in_C: mean interior temperature over the averaging period [degC].
        T_out_C: mean chamber temperature over the averaging period [degC].
        dT_K: mean T_in - T_out over the averaging period [K].
        dTdt_end_K_h: interior temperature drift at the end of the run [K/h].
        condition_violations: test conditions outside their evidence tolerance
            (E-009, E-010); empty if the test conditions were valid.
        data: the full simulated time series (see ``core.coach.simulate``).
    """

    k_measured_W_m2K: float
    t_steady_s: float
    T_in_C: float
    T_out_C: float
    dT_K: float
    dTdt_end_K_h: float
    condition_violations: list[str] = field(default_factory=list)
    data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def conditions_ok(self) -> bool:
        return not self.condition_violations


def _time_mean(t: np.ndarray, y: np.ndarray) -> float:
    return float(trapezoid(y, t) / (t[-1] - t[0]))


def _check_condition(evidence_id: str, unit: str, measured: float, what: str) -> str | None:
    rec = registry.get(evidence_id).record
    target = registry.number(evidence_id, unit)
    tol = rec.tolerance or 0.0
    if abs(measured - target) > tol:
        return (f"{evidence_id}: {what} {measured:.2f} {unit} is outside "
                f"{target:g} +/- {tol:g} {unit}")
    return None


def k_value_test(
    coach: SingleZoneCoach,
    T_out: float,
    Q_heater_W: float,
    t_end_s: float,
    dt_out_s: float = 60.0,
) -> KValueResult:
    """Virtual k-value test, heater method (UIC 553-1, 7.2.3 -- confirm against the PDF).

    The closed coach (HVAC off, Q_hvac = 0) starts at the chamber temperature
    T_out [degC]; heaters inside deliver a constant Q_heater_W [W].

    Steady state (UIC 553-1 rule as recorded in the evidence registry): over a
    window of E-011 hours (3 h) the interior drift |dT/dt| stays below E-032
    (0.1 K/h, assumption), dT/dt taken from the model energy balance. The run
    must end with such a window. k is evaluated from the time-means over the
    last E-033 hours (1 h) of the run:

        k = mean(Q_heater) / (A * mean(T_in - T_out))     [W/(m2.K)]

    with A = coach.envelope_area_m2 (definition: E-008). Remaining drift biases
    k by about C * dT/dt / Q_heater, so run well beyond ``t_steady_s``.

    Test conditions are checked against E-009 (T_in - T_out = 25 +/- 5 K) and
    E-010 (T_out = 5 +/- 2 degC); violations are reported in
    ``condition_violations`` rather than raised.

    Raises:
        ValueError: if Q_heater_W <= 0, or the run does not end with a full
            steady-state window.
    """
    if not Q_heater_W > 0:
        raise ValueError(f"Q_heater_W={Q_heater_W} must be > 0")
    window_s = registry.number("E-011", "h") * SECONDS_PER_HOUR
    avg_s = registry.number("E-033", "h") * SECONDS_PER_HOUR
    dTdt_max_K_h = registry.number("E-032", "K/h")
    if t_end_s < window_s:
        raise ValueError(
            f"{coach.name}: not steady: t_end_s={t_end_s} s is shorter than the "
            f"{window_s / SECONDS_PER_HOUR:g} h steady-state window (E-011)"
        )

    df = simulate(coach, T0=T_out, T_out=T_out, Q_int=Q_heater_W, Q_hvac=0.0,
                  t_end_s=t_end_s, dt_out_s=dt_out_s)
    t = df["t_s"].to_numpy()
    dTdt_K_h = net_sensible_W(df).to_numpy() / coach.heat_capacity_J_K * SECONDS_PER_HOUR

    # Window [t_i - window_s, t_i] is steady if no sample in it exceeds the criterion.
    unsteady_count = np.concatenate([[0], np.cumsum(np.abs(dTdt_K_h) >= dTdt_max_K_h)])
    start = np.searchsorted(t, t - window_s, side="left")
    window_steady = (t >= window_s - 1e-9) & (unsteady_count[np.arange(len(t)) + 1]
                                              - unsteady_count[start] == 0)
    if not window_steady[-1]:
        raise ValueError(
            f"{coach.name}: not steady: the last {window_s / SECONDS_PER_HOUR:g} h window "
            f"(E-011) has |dT/dt| >= {dTdt_max_K_h} K/h (E-032); "
            f"end drift {abs(dTdt_K_h[-1]):.3g} K/h"
        )

    last = t >= t[-1] - avg_s - 1e-9
    t_avg = t[last]
    T_in = _time_mean(t_avg, df["T_C"].to_numpy()[last])
    T_out_mean = _time_mean(t_avg, np.full(t_avg.shape, float(T_out)))
    dT = _time_mean(t_avg, (df["T_C"] - T_out).to_numpy()[last])
    Q_mean = _time_mean(t_avg, df["Q_int_W"].to_numpy()[last])

    violations = [v for v in (
        _check_condition("E-009", "K", dT, "temperature difference T_in - T_out"),
        _check_condition("E-010", "degC", T_out_mean, "chamber temperature"),
    ) if v]

    return KValueResult(
        k_measured_W_m2K=Q_mean / (coach.envelope_area_m2 * dT),
        t_steady_s=float(t[np.argmax(window_steady)]),
        T_in_C=T_in,
        T_out_C=T_out_mean,
        dT_K=dT,
        dTdt_end_K_h=float(dTdt_K_h[-1]),
        condition_violations=violations,
        data=df,
    )


# ==========================================================================
# Data-driven procedures (procedures/*.yaml) and their runner
# ==========================================================================
#
# A procedure file describes a test; every numeric value in it is a reference:
#
#   "E-012"            evidence row (value and unit checked)
#   "card:<field>"     unit-card field (SI value), e.g. card:rating_outdoor_dry_bulb
#   "param:<name>"     user-defined parameter declared under ``parameters:``
#   {user_defined: 26.0, unit: degC, tolerance: 1.5, note: "..."}   inline user value
#
# A bare number is rejected. See docs/README.md, "Virtual facilities and test
# procedures", for the file format.

PROCEDURES_DIR = Path(__file__).parents[1] / "procedures"
_REF_RE = re.compile(r"^(E-\d{3}|card:[a-z_]+|param:[A-Za-z0-9_]+)$")


class ProcedureError(ValueError):
    """A procedure file is invalid; ``problems`` lists every issue found."""

    def __init__(self, name: str, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"procedure {name!r} is invalid:\n"
                         + "\n".join(f"  - {p}" for p in problems))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserValue(_Strict):
    """A value chosen by the user (client), not taken from evidence."""

    user_defined: float
    unit: str
    tolerance: float | None = None
    note: str = ""


def _check_ref(v: object, keywords: tuple[str, ...] = ()) -> object:
    if isinstance(v, bool):
        raise ValueError(f"{v!r} is not a reference")
    if isinstance(v, int | float):
        raise ValueError(f"bare number {v}: write an evidence ID ('E-xxx'), 'card:<field>', "
                         f"'param:<name>' or {{user_defined: {v}, unit: ...}}")
    if isinstance(v, str) and v not in keywords and not _REF_RE.match(v):
        allowed = f" or one of {keywords}" if keywords else ""
        raise ValueError(f"{v!r} is not a reference ('E-xxx', 'card:<field>', "
                         f"'param:<name>'){allowed}")
    return v


Ref = Annotated[str | UserValue, BeforeValidator(_check_ref)]
PartLoadRef = Annotated[str | UserValue, BeforeValidator(lambda v: _check_ref(v, ("full",)))]
FreshRef = Annotated[str | UserValue, BeforeValidator(
    lambda v: _check_ref(v, ("from_rating_reference", "none")))]


class Solar(_Strict):
    intensity: Ref
    vertical_factor: Ref
    absorptance: Ref
    g_value: Ref
    h_o: Ref
    surfaces: list[Literal["roof", "left_side", "right_side"]]


class UnitSetting(_Strict):
    mode: Literal["cooling", "heating", "ventilation", "off"]
    part_load: PartLoadRef | None = None

    @model_validator(mode="after")
    def _check(self) -> UnitSetting:
        if self.mode in ("cooling", "heating") and self.part_load is None:
            raise ValueError(f"unit mode {self.mode!r} needs part_load")
        return self


class Passengers(_Strict):
    count: Ref
    sensible_W: Ref
    latent_W: Ref


class ChamberConditions(_Strict):
    chamber_T: Ref | None = None
    chamber_RH: Ref | None = None
    solar: Solar | Literal["off"] | None = None
    unit: UnitSetting | None = None
    heater_W: Ref | None = None
    passengers: Passengers | Literal["off"] | None = None


class AirSpec(_Strict):
    T: Ref
    RH: Ref | None = None
    wet_bulb: Ref | None = None

    @model_validator(mode="after")
    def _one_humidity(self) -> AirSpec:
        if (self.RH is None) == (self.wet_bulb is None):
            raise ValueError("give exactly one of RH or wet_bulb")
        return self


class RoomConditions(_Strict):
    indoor: AirSpec
    outdoor: AirSpec
    fresh_fraction: FreshRef = "none"
    unit: UnitSetting


class HoldUntilStable(_Strict):
    max_duration: Ref


class Hold(_Strict):
    duration: Ref
    require_stable_end: bool = False


class Ramp(_Strict):
    chamber_T: Ref
    rate: Ref


class RunUntil(_Strict):
    signal: Literal["cabin_T"] = "cabin_T"
    target: Ref
    tolerance: Ref
    max_duration: Ref


class Measure(_Strict):
    duration: Ref
    window: Literal["after", "trailing"] = "after"


class Stability(_Strict):
    signal: Literal["cabin_T"] = "cabin_T"
    window: Ref
    max_drift: Ref


class TargetTolerance(_Strict):
    target: Ref
    tolerance: Ref


class Criterion(_Strict):
    name: str
    quantity: str
    within_band: Ref | None = None
    within: TargetTolerance | None = None
    within_relative: TargetTolerance | None = None
    at_most: Ref | None = None
    at_least: Ref | None = None

    @model_validator(mode="after")
    def _one_rule(self) -> Criterion:
        rules = [r for r in ("within_band", "within", "within_relative", "at_most", "at_least")
                 if getattr(self, r) is not None]
        if len(rules) != 1:
            raise ValueError(f"criterion {self.name!r} needs exactly one rule, got {rules}")
        return self

    @property
    def rule(self) -> str:
        return next(r for r in ("within_band", "within", "within_relative", "at_most",
                                "at_least") if getattr(self, r) is not None)


class ProcedureFile(_Strict):
    name: str
    standard: str
    clause: str
    facility: Literal["unit_room", "vehicle_chamber"]
    description: str = ""
    parameters: dict[str, UserValue] = {}
    initial_state: Literal["soaked"] = "soaked"
    stability: Stability | None = None
    steps: list[dict]
    measured_quantities: list[str]
    acceptance: list[Criterion] = []
    mixed_standards_reason: str = ""      # required to use rows from another standard


# Test standards recognised in evidence sources and in a procedure's 'standard'.
# Handbooks and datasheets (ASHRAE Fundamentals, glazing datasheet, ...) are not
# test standards and are not counted.
STANDARD_FAMILIES: tuple[tuple[str, str], ...] = (
    ("UIC 553-1", r"UIC\s*553-1"),
    ("UIC 553", r"UIC\s*553(?!-\d)"),
    ("EN 13129", r"EN\s*13129"),
    ("EN 14750", r"EN\s*14750"),
    ("EN 14511", r"EN\s*14511"),
    ("ASHRAE 37", r"ASHRAE\s*37"),
)


def standard_families(text: str) -> set[str]:
    """Test standards named in ``text`` (e.g. an evidence source)."""
    return {name for name, pattern in STANDARD_FAMILIES if re.search(pattern, text or "")}


STEP_MODELS: dict[str, type[BaseModel] | None] = {
    "set_conditions": None,     # facility-specific model
    "hold_until_stable": HoldUntilStable,
    "hold": Hold,
    "ramp": Ramp,
    "run_until": RunUntil,
    "measure": Measure,
}
FACILITY_STEPS = {
    "vehicle_chamber": set(STEP_MODELS),
    "unit_room": {"set_conditions", "hold_until_stable", "hold", "measure"},
}
QUANTITIES: dict[str, dict[str, str]] = {   # facility -> quantity -> unit
    "vehicle_chamber": {"cabin_T_mean": "degC", "cabin_RH_mean": "-", "chamber_T_mean": "degC",
                        "dT_cabin_chamber_mean": "K", "k_value": "W/m2K",
                        "time_to_setpoint_h": "h"},
    "unit_room": {"capacity_gross_W": "W", "capacity_net_W": "W",
                  "sensible_capacity_gross_W": "W"},
}


@dataclass(frozen=True)
class Procedure:
    file: ProcedureFile
    steps: list[tuple[str, BaseModel]]
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.file.name


def _format_validation(exc: ValidationError, prefix: str = "") -> list[str]:
    out = []
    for e in exc.errors():
        loc = prefix + ".".join(str(x) for x in e["loc"])
        if e["type"] == "extra_forbidden":
            out.append(f"unknown key '{loc}'")
        else:
            msg = e["msg"].removeprefix("Value error, ")
            out.append(f"{loc}: {msg}")
    return out


def _parse_steps(raw_steps: list, facility: str) -> tuple[list[tuple[str, BaseModel]], list[str]]:
    problems: list[str] = []
    steps: list[tuple[str, BaseModel]] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict) or len(raw) != 1:
            problems.append(f"steps[{i}]: each step needs exactly one key "
                            f"(one of {sorted(STEP_MODELS)})")
            continue
        (kind, body), = raw.items()
        if kind not in STEP_MODELS:
            problems.append(f"unknown key 'steps[{i}].{kind}' (step types: {sorted(STEP_MODELS)})")
            continue
        if kind not in FACILITY_STEPS[facility]:
            problems.append(f"steps[{i}]: step '{kind}' is not available in {facility}")
            continue
        model = STEP_MODELS[kind] or (ChamberConditions if facility == "vehicle_chamber"
                                      else RoomConditions)
        try:
            steps.append((kind, model.model_validate(body or {})))
        except ValidationError as exc:
            problems += _format_validation(exc, f"steps[{i}].{kind}.")
    return steps, problems


def parse_procedure(data: dict, name: str = "<procedure>", path: Path | None = None) -> Procedure:
    """Validate the structure of a procedure (keys, types, references).

    All problems are collected (top level and every step) before raising.
    """
    title = data.get("name", name) if isinstance(data, dict) else name
    try:
        pf = ProcedureFile.model_validate(data)
    except ValidationError as exc:
        problems = _format_validation(exc)
        if isinstance(data, dict) and isinstance(data.get("steps"), list) \
                and data.get("facility") in FACILITY_STEPS:
            problems += _parse_steps(data["steps"], data["facility"])[1]
        raise ProcedureError(title, problems) from None
    steps, problems = _parse_steps(pf.steps, pf.facility)
    known = QUANTITIES[pf.facility]
    for q in pf.measured_quantities:
        if q not in known:
            problems.append(f"measured quantity '{q}' is unknown for {pf.facility} "
                            f"(known: {sorted(known)})")
    for c in pf.acceptance:
        if c.quantity not in pf.measured_quantities:
            problems.append(f"acceptance criterion '{c.name}': quantity '{c.quantity}' is not "
                            f"in measured_quantities")
    if not any(k == "measure" or k == "run_until" for k, _ in steps):
        problems.append("the procedure has no 'measure' or 'run_until' step")
    if problems:
        raise ProcedureError(pf.name, problems)
    return Procedure(pf, steps, path)


def load_procedure(path_or_name: Path | str) -> Procedure:
    """Load procedures/<name>.yaml (or a path) and validate its structure."""
    path = Path(path_or_name)
    if not path.suffix:
        path = PROCEDURES_DIR / f"{path_or_name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_procedure(data, path.stem, path)


# --------------------------------------------------------------------------
# Reference resolution with provenance
# --------------------------------------------------------------------------

@dataclass
class EvidenceUse:
    """One evidence row (or unit-card field) used by a test."""

    id: str
    name: str
    value: float | str | None
    unit: str
    status: str
    override: str = ""
    used_in: list[str] = field(default_factory=list)

    @property
    def is_assumption(self) -> bool:
        return self.status in ("assumption", "plot-only assumption") or bool(self.override)


@dataclass
class UserDefinedUse:
    name: str
    value: float
    unit: str
    note: str
    used_in: list[str] = field(default_factory=list)


class _Resolver:
    """Resolves references, records where each is used, collects problems."""

    def __init__(self, evidence: Registry, card: UnitCard | None,
                 declared: dict[str, UserValue], params: dict[str, float]) -> None:
        self.evidence, self.card = evidence, card
        self.declared, self.params = declared, params
        self.evidence_used: dict[str, EvidenceUse] = {}
        self.user_defined: dict[str, UserDefinedUse] = {}
        self.problems: list[str] = []
        self.file_rows: dict = {}      # evidence rows referenced by the procedure file

    def _note_evidence(self, rec, where: str) -> None:
        use = self.evidence_used.setdefault(rec.id, EvidenceUse(
            rec.id, rec.name, rec.value, rec.unit, str(rec.status), rec.override))
        if where not in use.used_in:
            use.used_in.append(where)

    def use_evidence_id(self, eid: str, where: str) -> None:
        """Record an evidence ID used implicitly (facility, coach, model)."""
        try:
            self._note_evidence(self.evidence.get(eid).record, where)
        except (KeyError, EvidenceError) as exc:
            self.problems.append(f"{where}: {exc}")

    def resolve(self, ref, unit: str, where: str, band: bool = False):
        """Value (or (value, tolerance) if band) of ``ref`` in ``unit``; None on error."""
        if isinstance(ref, UserValue):
            return self._user(f"inline in {where}", ref, ref.user_defined, unit, where, band)
        if ref.startswith("param:"):
            name = ref[6:]
            if name not in self.declared:
                self.problems.append(f"{where}: parameter '{name}' is not declared under "
                                     f"'parameters'")
                return None
            decl = self.declared[name]
            return self._user(name, decl, self.params.get(name, decl.user_defined), unit,
                              where, band)
        if ref.startswith("card:"):
            return self._card(ref[5:], unit, where, band)
        return self._evidence(ref, unit, where, band)

    def _user(self, name, decl: UserValue, value, unit, where, band):
        if decl.unit != unit:
            self.problems.append(f"{where}: user-defined '{name}' has unit {decl.unit!r}, "
                                 f"expected {unit!r}")
            return None
        use = self.user_defined.setdefault(name, UserDefinedUse(name, value, unit, decl.note))
        if where not in use.used_in:
            use.used_in.append(where)
        if band:
            if decl.tolerance is None:
                self.problems.append(f"{where}: user-defined '{name}' needs a tolerance")
                return None
            return value, decl.tolerance
        return value

    def _card(self, name, unit, where, band):
        if band:
            self.problems.append(f"{where}: card fields have no tolerance; use 'within'")
            return None
        if self.card is None:
            self.problems.append(f"{where}: 'card:{name}' used but no unit card was given")
            return None
        if name not in CARD_FIELDS:
            self.problems.append(f"{where}: unknown unit-card field 'card:{name}'")
            return None
        f = getattr(self.card, name)
        if f is None or f.value is None:
            state = "missing" if f is None else f"empty (status {f.status})"
            self.problems.append(f"{where}: unit-card field '{name}' of {self.card.name} is "
                                 f"{state}")
            return None
        si = CARD_FIELDS[name][0]
        if si != unit:
            self.problems.append(f"{where}: card:{name} is in {si!r}, expected {unit!r}")
            return None
        key = f"card:{self.card.unit_id}.{name}"
        use = self.evidence_used.setdefault(key, EvidenceUse(
            key, f"{self.card.name}: {name}", f.value, f.unit, f.status))
        if where not in use.used_in:
            use.used_in.append(where)
        return self.card.si(name)

    def _evidence(self, eid, unit, where, band):
        try:
            rec = self.evidence.get(eid).record
        except KeyError:
            self.problems.append(f"{where}: unknown evidence ID {eid}")
            return None
        except EvidenceError as exc:          # plot-only
            self.problems.append(f"{where}: {exc}")
            return None
        if rec.value is None:
            self.problems.append(f"{where}: {eid} ({rec.name}) has an EMPTY value "
                                 f"(status {rec.status}); fill it in evidence.csv")
            return None
        if not isinstance(rec.value, float):
            self.problems.append(f"{where}: {eid} is not a number ({rec.value!r})")
            return None
        if rec.unit != unit:
            self.problems.append(f"{where}: {eid} has unit {rec.unit!r}, expected {unit!r}")
            return None
        self._note_evidence(rec, where)
        self.file_rows[eid] = rec
        if band:
            if rec.tolerance is None:
                self.problems.append(f"{where}: {eid} has no tolerance for a band")
                return None
            return rec.value, rec.tolerance
        return rec.value


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    uncertainty: float          # worst-case bound from sensor accuracies
    uncertainty_rss: float
    unit: str
    true_value: float           # model value with ideal sensors (verification)


@dataclass(frozen=True)
class CriterionResult:
    name: str
    quantity: str
    rule: str
    measured: float
    uncertainty: float
    passed: bool
    conclusive: bool            # False if the uncertainty interval straddles the limit


@dataclass
class Segment:
    """One simulated step of a vehicle-chamber procedure."""

    step: str
    df: pd.DataFrame            # rows used (possibly truncated); has t_abs_s, T_chamber_C
    full: pd.DataFrame          # everything simulated in the step
    t_offset_s: float


@dataclass
class TestResult:
    """Everything a report needs about one procedure run."""

    procedure: str
    standard: str
    clause: str
    facility: str
    measurements: dict[str, Measurement]
    criteria: list[CriterionResult]
    evidence_used: list[EvidenceUse]
    user_defined: list[UserDefinedUse]
    validity: list[str]
    notes: list[str]
    segments: list[Segment] = field(default_factory=list)
    sensors: str = "ideal"

    @property
    def passed(self) -> bool:
        return not self.validity and all(c.passed for c in self.criteria)

    @property
    def evidence_ids(self) -> list[str]:
        return [u.id for u in self.evidence_used]

    @property
    def assumptions(self) -> list[EvidenceUse]:
        """Evidence rows / card fields used that are assumptions or overrides."""
        return [u for u in self.evidence_used if u.is_assumption]

    @property
    def data(self) -> pd.DataFrame:
        return pd.concat([s.df.assign(step=s.step) for s in self.segments], ignore_index=True)

    def summary(self) -> str:
        lines = [f"{self.procedure}  [{self.standard}, {self.clause}]",
                 f"facility: {self.facility}; sensors: {self.sensors}",
                 f"RESULT: {'PASS' if self.passed else 'FAIL'}", "", "Measured:"]
        for m in self.measurements.values():
            lines.append(f"  {m.name} = {m.value:.4g} +/- {m.uncertainty:.2g} {m.unit}")
        lines.append("Criteria:")
        for c in self.criteria:
            flag = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{flag}{'' if c.conclusive else ', inconclusive'}] {c.name}: "
                         f"{c.quantity} {c.rule}")
        if self.validity:
            lines.append("Validity issues:")
            lines += [f"  - {v}" for v in self.validity]
        lines.append(f"Evidence used ({len(self.evidence_used)}): "
                     + ", ".join(self.evidence_ids))
        lines.append(f"Assumptions / overrides ({len(self.assumptions)}):")
        lines += [f"  - {u.id} {u.name} = {u.value} {u.unit} ({u.status}"
                  f"{', ' + u.override if u.override else ''})" for u in self.assumptions]
        lines.append(f"User-defined values ({len(self.user_defined)}):")
        lines += [f"  - {u.name} = {u.value:g} {u.unit}" for u in self.user_defined]
        if self.notes:
            lines.append("Notes:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def _evaluate(c: Criterion, m: Measurement, r: _Resolver, unit: str) -> CriterionResult:
    where = f"acceptance criterion '{c.name}' ({c.rule})"
    v, U = m.value, m.uncertainty
    unresolved = CriterionResult(c.name, c.quantity, "unresolved reference", v, U, False, True)
    if c.rule == "within_band":
        band = r.resolve(c.within_band, unit, where, band=True)
        if band is None:
            return unresolved
        center, tol = band
        lo, hi, text = center - tol, center + tol, f"in {center:g} +/- {tol:g} {unit}"
    elif c.rule in ("within", "within_relative"):
        tt = c.within or c.within_relative
        center = r.resolve(tt.target, unit, where)
        tol = r.resolve(tt.tolerance, unit if c.rule == "within" else "-", where)
        if center is None or tol is None:
            return unresolved
        if c.rule == "within_relative":
            tol, text = tol * abs(center), f"within {tol * 100:g} % of {center:g} {unit}"
        else:
            text = f"in {center:g} +/- {tol:g} {unit}"
        lo, hi = center - tol, center + tol
    elif c.rule == "at_most":
        lo, hi = -math.inf, r.resolve(c.at_most, unit, where)
        if hi is None:
            return unresolved
        text = f"<= {hi:g} {unit}"
    else:
        lo, hi = r.resolve(c.at_least, unit, where), math.inf
        if lo is None:
            return unresolved
        text = f">= {lo:g} {unit}"
    if math.isnan(v):
        return CriterionResult(c.name, c.quantity, text + " (not measured)", v, U, False, True)
    passed = lo <= v <= hi
    conclusive = (lo <= v - U and v + U <= hi) if passed else (v + U < lo or v - U > hi)
    return CriterionResult(c.name, c.quantity, text, v, U, passed, conclusive)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_procedure(
    procedure: Procedure | Path | str,
    *,
    coach: CoachDefinition | SingleZoneCoach | None = None,
    unit_card: UnitCard | None = None,
    evidence: Registry | None = None,
    sensors: Literal["ideal", "evidence"] = "ideal",
    seed: int | None = 0,
    parameters: dict[str, float] | None = None,
    dt_out_s: float = 60.0,
) -> TestResult:
    """Validate and run a procedure; return a TestResult.

    Args:
        procedure: a Procedure, a path, or a name in procedures/.
        coach: the coach (vehicle_chamber); a CoachDefinition is needed for solar.
        unit_card: the HVAC unit card (unit_room, or vehicle_chamber with unit on).
        evidence: registry to resolve IDs (e.g. with labelled overrides).
        sensors: "ideal" or "evidence" (accuracies E-058..E-060, seeded errors).
        parameters: values for user-defined parameters (override file defaults).

    Raises:
        ProcedureError: every problem found before running (unknown keys,
            unknown or empty evidence, plot-only values, unit mismatches,
            missing inputs).
    """
    proc = procedure if isinstance(procedure, Procedure) else load_procedure(procedure)
    pf = proc.file
    reg = evidence if evidence is not None else Registry()
    params = dict(parameters or {})
    r = _Resolver(reg, unit_card, pf.parameters, params)
    for name in params:
        if name not in pf.parameters:
            r.problems.append(f"parameter '{name}' given but not declared in the procedure")

    # ---- dry resolution of every reference (collects all problems) -------
    _resolve_all(proc, r)
    mixing_note = _check_one_standard(pf, r)
    facility = None
    try:
        facility = (VehicleChamber.from_evidence(reg) if pf.facility == "vehicle_chamber"
                    else UnitTestRoom.from_evidence(reg))
        for eid in facility.evidence_ids:
            r.use_evidence_id(eid, f"facility {pf.facility}")
    except (EvidenceError, KeyError) as exc:
        r.problems.append(f"facility {pf.facility}: {exc}")
    sensor_set = SensorSet.ideal()
    if sensors == "evidence":
        try:
            sensor_set = SensorSet.from_evidence(reg, seed)
            for eid in sensor_set.evidence_ids:
                r.use_evidence_id(eid, "virtual sensors")
        except EvidenceError as exc:
            r.problems.append(f"virtual sensors: {exc}")
    unit = None
    needs_unit = pf.facility == "unit_room" or any(
        k == "set_conditions" and getattr(s, "unit", None) is not None and s.unit.mode != "off"
        for k, s in proc.steps)
    if needs_unit:
        if unit_card is None:
            r.problems.append("the procedure runs the HVAC unit but no unit card was given")
        else:
            try:
                unit = DXUnit.from_card(unit_card, evidence=reg)
                for eid in CURVE_IDS:
                    r.use_evidence_id(eid, "unit capacity curve")
            except (EvidenceError, ValueError) as exc:
                r.problems.append(f"unit model: {exc}")
    if pf.facility == "vehicle_chamber":
        if coach is None:
            r.problems.append("vehicle_chamber procedures need a coach")
        elif isinstance(coach, CoachDefinition):
            used = ["k_value_W_m2K", "envelope_area_Ae_m2", "heat_capacity_J_K",
                    "length_m", "width_m", "height_m",           # model, air volume
                    "exterior_length_m", "exterior_width_m", "exterior_height_m"]  # solar,
                                                                 # chamber displacement
            if any(k == "set_conditions" and getattr(s, "solar", None) not in (None, "off")
                   for k, s in proc.steps):
                used.append("glazing_area_per_side_m2")
            for field_name in used:
                r.use_evidence_id(coach.evidence_ids[field_name],
                                  f"coach {coach.name}: {field_name}")
    if r.problems:
        raise ProcedureError(pf.name, r.problems)

    if pf.facility == "vehicle_chamber":
        result = _run_vehicle(proc, r, facility, coach, unit, sensor_set, dt_out_s)
    else:
        result = _run_room(proc, r, facility, unit, sensor_set)
    result.sensors = "ideal" if sensor_set.is_ideal else f"evidence accuracies, seed {seed}"
    if mixing_note:
        result.notes.insert(0, mixing_note)
    if r.problems:             # problems found while evaluating (should not happen)
        raise ProcedureError(pf.name, r.problems)
    return result


def _check_one_standard(pf: ProcedureFile, r: _Resolver) -> str | None:
    """One standard per procedure: no evidence row from another test standard.

    Returns a note when mixing is explicitly allowed by mixed_standards_reason.
    """
    declared = standard_families(pf.standard)
    reason = pf.mixed_standards_reason.strip()
    if len(declared) > 1 and not reason:
        r.problems.append(f"'standard: {pf.standard}' names {sorted(declared)}: a procedure "
                          f"follows ONE standard (or give 'mixed_standards_reason')")
    foreign = []
    for eid, rec in sorted(r.file_rows.items()):
        fam = standard_families(rec.source)
        if fam and not fam & declared:
            foreign.append(f"{eid} ({', '.join(sorted(fam))})")
    if foreign and not reason:
        r.problems.append(f"mixes standards: the procedure follows "
                          f"{', '.join(sorted(declared)) or 'no test standard'} but uses "
                          f"{'; '.join(foreign)}. Use rows of one standard, or give "
                          f"'mixed_standards_reason'")
    if (foreign or len(declared) > 1) and reason:
        return f"MIXED STANDARDS allowed ({'; '.join(foreign)}): {reason}"
    return None


def _resolve_all(proc: Procedure, r: _Resolver) -> None:
    pf = proc.file
    if pf.stability is not None:
        r.resolve(pf.stability.window, "h", "stability.window")
        r.resolve(pf.stability.max_drift, "K/h", "stability.max_drift")
    for i, (kind, s) in enumerate(proc.steps):
        where = f"steps[{i}].{kind}"
        if kind == "set_conditions" and isinstance(s, ChamberConditions):
            _chamber_conditions(s, r, where)
        elif kind == "set_conditions":
            _room_conditions(s, r, where)
        elif kind == "hold_until_stable":
            r.resolve(s.max_duration, "h", f"{where}.max_duration")
            if pf.stability is None:
                r.problems.append(f"{where}: needs a 'stability' rule")
        elif kind == "hold":
            r.resolve(s.duration, "h", f"{where}.duration")
            if s.require_stable_end and pf.stability is None:
                r.problems.append(f"{where}: require_stable_end needs a 'stability' rule")
        elif kind == "ramp":
            r.resolve(s.chamber_T, "degC", f"{where}.chamber_T")
            r.resolve(s.rate, "K/h", f"{where}.rate")
        elif kind == "run_until":
            r.resolve(s.target, "degC", f"{where}.target")
            r.resolve(s.tolerance, "K", f"{where}.tolerance")
            r.resolve(s.max_duration, "h", f"{where}.max_duration")
        elif kind == "measure":
            r.resolve(s.duration, "h", f"{where}.duration")
    units = QUANTITIES[pf.facility]
    for c in pf.acceptance:
        dummy = Measurement(c.quantity, 0.0, 0.0, 0.0, units[c.quantity], 0.0)
        _evaluate(c, dummy, r, units[c.quantity])


def _chamber_conditions(s: ChamberConditions, r: _Resolver, where: str) -> dict:
    out: dict = {}
    if s.chamber_T is not None:
        out["T_set"] = r.resolve(s.chamber_T, "degC", f"{where}.chamber_T")
    if s.chamber_RH is not None:
        out["RH"] = r.resolve(s.chamber_RH, "-", f"{where}.chamber_RH")
    if s.heater_W is not None:
        out["heater_W"] = r.resolve(s.heater_W, "W", f"{where}.heater_W")
    if s.solar == "off":
        out["solar"] = None
    elif s.solar is not None:
        w = f"{where}.solar"
        out["solar"] = dict(
            I=r.resolve(s.solar.intensity, "W/m2", f"{w}.intensity"),
            f_vert=r.resolve(s.solar.vertical_factor, "-", f"{w}.vertical_factor"),
            alpha=r.resolve(s.solar.absorptance, "-", f"{w}.absorptance"),
            g=r.resolve(s.solar.g_value, "-", f"{w}.g_value"),
            h_o=r.resolve(s.solar.h_o, "W/m2K", f"{w}.h_o"),
            surfaces=list(s.solar.surfaces))
    if s.passengers == "off":
        out["passengers"] = None
    elif s.passengers is not None:
        w = f"{where}.passengers"
        out["passengers"] = dict(
            n=r.resolve(s.passengers.count, "-", f"{w}.count"),
            sens=r.resolve(s.passengers.sensible_W, "W/person", f"{w}.sensible_W"),
            lat=r.resolve(s.passengers.latent_W, "W/person", f"{w}.latent_W"))
    if s.unit is not None:
        out["unit"] = _unit_setting(s.unit, r, f"{where}.unit")
    return out


def _unit_setting(u: UnitSetting, r: _Resolver, where: str) -> tuple[str, float]:
    if u.mode == "off" or u.part_load is None:
        return u.mode, 0.0
    if u.part_load == "full":
        return u.mode, 1.0
    plr = r.resolve(u.part_load, "-", f"{where}.part_load")
    if plr is not None and not 0 <= plr <= 1:
        r.problems.append(f"{where}.part_load: {plr} must be in 0..1")
    return u.mode, plr


def _air(spec: AirSpec, r: _Resolver, where: str) -> tuple[float, float] | None:
    T = r.resolve(spec.T, "degC", f"{where}.T")
    if spec.RH is not None:
        RH = r.resolve(spec.RH, "-", f"{where}.RH")
        return None if None in (T, RH) else (T, psychro.humidity_ratio(T, RH))
    wb = r.resolve(spec.wet_bulb, "degC", f"{where}.wet_bulb")
    return None if None in (T, wb) else (T, psychro.humidity_ratio_from_wet_bulb(T, wb))


def _room_conditions(s: RoomConditions, r: _Resolver, where: str) -> dict:
    out = dict(indoor=_air(s.indoor, r, f"{where}.indoor"),
               outdoor=_air(s.outdoor, r, f"{where}.outdoor"),
               unit=_unit_setting(s.unit, r, f"{where}.unit"))
    if s.fresh_fraction in ("none", "from_rating_reference"):
        out["fresh_fraction"] = s.fresh_fraction
    else:
        out["fresh_fraction"] = r.resolve(s.fresh_fraction, "-", f"{where}.fresh_fraction")
    return out


# ---- vehicle chamber -------------------------------------------------------

def _time_mean_window(t: np.ndarray, y: np.ndarray) -> float:
    return float(trapezoid(y, t) / (t[-1] - t[0]))


def _steady_flags(t: np.ndarray, drift_K_h: np.ndarray, window_s: float,
                  max_drift: float) -> np.ndarray:
    """flags[i]: the window [t_i - window, t_i] lies in the data and is steady."""
    bad = np.concatenate([[0], np.cumsum(np.abs(drift_K_h) >= max_drift)])
    start = np.searchsorted(t, t - window_s, side="left")
    return (t - t[0] >= window_s - 1e-9) & (bad[np.arange(len(t)) + 1] - bad[start] == 0)


def _run_vehicle(proc, r, chamber: VehicleChamber, coach, unit, sensors, dt_out_s):
    pf = proc.file
    model = coach.thermal_model() if isinstance(coach, CoachDefinition) else coach
    stab = None
    if pf.stability is not None:
        stab = (r.resolve(pf.stability.window, "h", "stability.window") * SECONDS_PER_HOUR,
                r.resolve(pf.stability.max_drift, "K/h", "stability.max_drift"))

    conds = [(i, _chamber_conditions(s, r, f"steps[{i}].set_conditions"))
             for i, (k, s) in enumerate(proc.steps) if k == "set_conditions"]
    humid = any(c.get("RH") is not None or c.get("passengers") or c.get("solar")
                or (c.get("unit") and c["unit"][0] != "off") for _, c in conds)
    if humid and not any(c.get("RH") is not None for _, c in conds):
        raise ProcedureError(pf.name, ["chamber_RH is needed (unit, passengers or solar "
                                       "require the humidity state)"])
    if any(c.get("solar") for _, c in conds) and not isinstance(coach, CoachDefinition):
        raise ProcedureError(pf.name, ["solar loads need a CoachDefinition (geometry)"])
    if humid:
        for eid in ("E-004", "E-006", "E-007"):
            r.use_evidence_id(eid, "psychrometrics / vapour enthalpy")

    st = dict(T=None, W=None, T_ch=None, T_set=None, RH=None, heater=0.0, solar=None,
              pax=None, unit=("off", 0.0), t=0.0)
    segments: list[Segment] = []
    validity: list[str] = []
    notes = ["chamber air is one well-mixed volume (spatial non-uniformity: CFD, later)",
             "plant-capacity check is quasi-steady and excludes compressor power and "
             "humidity control"]
    measure_windows: list[tuple[int, np.ndarray, str]] = []
    run_until_events: list[tuple[int, float, float, float]] = []
    lamp_power = 0.0

    def build_loads(T_out_fn):
        nonlocal lamp_power
        terms, lamp_power = [], 0.0
        if st["solar"]:
            s, d = st["solar"], coach
            roof, side = d.roof_area_m2, d.side_area_m2          # exterior dimensions
            glaz = d.glazing_area_per_side_m2
            for surf in s["surfaces"]:
                I_s, A_op, A_g = ((s["I"], roof, 0.0) if surf == "roof"
                                  else (s["I"] * s["f_vert"], side - glaz, glaz))
                lamp_power += I_s * (A_op + A_g)
                k = model.k_value_W_m2K

                def fn(t, T, W, I_s=I_s, A_op=A_op, A_g=A_g):
                    T_o = T_out_fn(t)
                    q = (loads.solar_glazing(I_s, A_g, s["g"])
                         + loads.solar_opaque_sol_air(k, A_op, s["alpha"], I_s, s["h_o"], T_o, T)
                         - loads.envelope(k, A_op, T_o, T))
                    return q, 0.0
                terms.append(LoadTerm(f"solar_{surf}", fn))
        if st["pax"]:
            p = st["pax"]
            terms.append(LoadTerm("passengers", lambda t, T, W: loads.passengers(
                p["n"], p["sens"], p["lat"], T_in=T)))
        return terms

    def segment(step: str, duration_s: float, rate_K_h: float | None = None) -> Segment:
        T_fn = chamber.air_temperature(st["T_ch"], st["T_set"], rate_K_h)
        mode, plr = st["unit"]
        op = None if mode == "off" else UnitOperation(unit, mode, plr, st["RH"])
        df = simulate(model, T0=st["T"], T_out=T_fn, Q_int=st["heater"], Q_hvac=0.0,
                      t_end_s=duration_s, dt_out_s=dt_out_s,
                      W0=st["W"] if humid else None, loads=build_loads(T_fn), unit=op)
        df["T_chamber_C"] = [T_fn(t) for t in df["t_s"]]
        df["t_abs_s"] = df["t_s"] + st["t"]
        df.attrs["T_fn"], df.attrs["lamp_power_W"] = T_fn, lamp_power
        seg = Segment(step, df, df, st["t"])
        segments.append(seg)
        _advance(seg, len(df) - 1)
        _plant_check(seg)
        return seg

    def _advance(seg: Segment, i: int) -> None:
        row = seg.full.iloc[i]
        seg.df = seg.full.iloc[: i + 1]
        st["T"], st["T_ch"] = float(row["T_C"]), float(row["T_chamber_C"])
        if humid:
            st["W"] = float(row["W_kg_kg"])
        st["t"] = seg.t_offset_s + float(row["t_s"])

    def _plant_check(seg: Segment) -> None:
        W_ch = psychro.humidity_ratio(st["T_set"], st["RH"]) if humid else None
        vol = coach.exterior_volume_m3 if isinstance(coach, CoachDefinition) else (
            model.air_volume_m3 or 0.0)
        P = chamber.plant_power(seg.full, seg.full["T_chamber_C"].to_numpy(), W_ch,
                                seg.full.attrs["lamp_power_W"], vol)
        if P.max() > chamber.Q_cool_max_W:
            validity.append(f"E-065: chamber cooling demand {P.max():.0f} W exceeds capacity "
                            f"{chamber.Q_cool_max_W:.0f} W in step '{seg.step}'")
        if -P.min() > chamber.Q_heat_max_W:
            validity.append(f"E-064: chamber heating demand {-P.min():.0f} W exceeds capacity "
                            f"{chamber.Q_heat_max_W:.0f} W in step '{seg.step}'")

    def drift(df: pd.DataFrame) -> np.ndarray:
        return net_sensible_W(df).to_numpy() / model.heat_capacity_J_K * SECONDS_PER_HOUR

    stop = False
    for i, (kind, s) in enumerate(proc.steps):
        if stop:
            break
        where = f"steps[{i}].{kind}"
        if kind == "set_conditions":
            c = _chamber_conditions(s, r, where)
            for key, name in (("T_set", "T_set"), ("RH", "RH"), ("heater_W", "heater"),
                              ("solar", "solar"), ("passengers", "pax"), ("unit", "unit")):
                if key in c:
                    st[name] = c[key]
            if st["T"] is None:                         # initial_state: soaked
                if st["T_set"] is None:
                    raise ProcedureError(pf.name, [f"{where}: the first set_conditions "
                                                   f"needs chamber_T"])
                st["T"] = st["T_ch"] = st["T_set"]
                if humid:
                    st["W"] = psychro.humidity_ratio(st["T_set"], st["RH"])
        elif kind == "hold_until_stable":
            seg = segment(where, r.resolve(s.max_duration, "h", where) * SECONDS_PER_HOUR)
            flags = _steady_flags(seg.full["t_s"].to_numpy(), drift(seg.full), *stab)
            if flags.any():
                _advance(seg, int(np.argmax(flags)))
            else:
                validity.append(f"{where}: not stable within the maximum duration "
                                f"(stability rule {pf.stability.window} / "
                                f"{pf.stability.max_drift})")
                stop = True
        elif kind == "hold":
            seg = segment(where, r.resolve(s.duration, "h", where) * SECONDS_PER_HOUR)
            if s.require_stable_end:
                flags = _steady_flags(seg.full["t_s"].to_numpy(), drift(seg.full), *stab)
                if not flags[-1]:
                    validity.append(f"{where}: not steady at the end (window "
                                    f"{pf.stability.window}, drift {pf.stability.max_drift})")
        elif kind == "ramp":
            target = r.resolve(s.chamber_T, "degC", where)
            rate = r.resolve(s.rate, "K/h", where)
            T0_ch, st["T_set"] = st["T_ch"], target
            segment(where, max(chamber.ramp_duration_s(T0_ch, target, rate), dt_out_s), rate)
        elif kind == "run_until":
            target = r.resolve(s.target, "degC", where)
            tol = r.resolve(s.tolerance, "K", where)
            seg = segment(where, r.resolve(s.max_duration, "h", where) * SECONDS_PER_HOUR)
            T = seg.full["T_C"].to_numpy()
            hit = np.flatnonzero(np.abs(T - target) <= tol)
            if hit.size:
                _advance(seg, int(hit[0]))
            else:
                validity.append(f"{where}: cabin did not reach {target:g} +/- {tol:g} degC "
                                f"within the maximum duration")
            run_until_events.append((len(segments) - 1, target, tol, math.nan))
        elif kind == "measure":
            dur = r.resolve(s.duration, "h", where) * SECONDS_PER_HOUR
            if s.window == "after":
                seg = segment(where, dur)
                mask = np.ones(len(seg.df), bool)
            else:
                if not segments:
                    raise ProcedureError(pf.name, [f"{where}: no previous step to measure"])
                seg = segments[-1]
                t = seg.df["t_s"].to_numpy()
                if t[-1] - t[0] < dur - 1e-9:
                    validity.append(f"{where}: previous step shorter than the measurement")
                mask = t >= t[-1] - dur - 1e-9
            measure_windows.append((segments.index(seg), mask, where))

    # ---- quantities -------------------------------------------------------
    measurements = _vehicle_quantities(pf, r, chamber, model, sensors, segments,
                                       measure_windows, run_until_events, humid)
    for _, (si, mask, where) in enumerate(measure_windows):
        w = segments[si].df[mask]
        T_ch_meas = chamber.measured_temperature(w["T_chamber_C"].to_numpy(), sensors)
        T_set = w["T_chamber_C"].iloc[-1]
        dev = np.max(np.abs(T_ch_meas - T_set))
        if dev > chamber.tol_T_K:
            validity.append(f"E-062: measured chamber temperature deviates {dev:.2f} K from "
                            f"setpoint during '{where}' (tolerance {chamber.tol_T_K:g} K)")
    units = QUANTITIES["vehicle_chamber"]
    criteria = [_evaluate(c, measurements[c.quantity], r, units[c.quantity])
                for c in pf.acceptance]
    return TestResult(pf.name, pf.standard, pf.clause, pf.facility, measurements, criteria,
                      sorted(r.evidence_used.values(), key=lambda u: u.id),
                      list(r.user_defined.values()), validity, notes, segments)


def _vehicle_quantities(pf, r, chamber, model, sensors, segments, windows, events, humid):
    units = QUANTITIES["vehicle_chamber"]
    out: dict[str, Measurement] = {}
    ideal = SensorSet.ideal()

    def window_data():
        if not windows:
            return None
        si, mask, _ = windows[-1]
        return segments[si].df[mask]

    def make(name, fn, channels):
        m = propagate(lambda b: fn(b, sensors), channels, sensors)
        out[name] = Measurement(name, m.value, m.uncertainty, m.uncertainty_rss, units[name],
                                fn({}, ideal))

    w = window_data()
    for q in pf.measured_quantities:
        if q == "time_to_setpoint_h":
            if not events:
                r.problems.append("time_to_setpoint_h needs a run_until step")
                continue
            si, target, tol, _ = events[-1]
            full = segments[si].full

            def ttr(b, sens, full=full, target=target, tol=tol):
                t = full["t_s"].to_numpy()
                T = sens.read("T_cabin", "temperature", full["T_C"].to_numpy(), t=t)[0] \
                    + b.get("T_cabin", 0.0)
                d = np.abs(T - target) - tol
                hit = np.flatnonzero(d <= 0)
                if not hit.size:
                    return math.nan
                k = hit[0]
                if k == 0:
                    return 0.0
                t_hit = t[k - 1] + (t[k] - t[k - 1]) * d[k - 1] / (d[k - 1] - d[k])
                return t_hit / SECONDS_PER_HOUR
            make(q, ttr, [Channel("T_cabin", "temperature")])
            continue
        if w is None:
            r.problems.append(f"quantity '{q}' needs a measure step")
            continue
        t = w["t_s"].to_numpy()

        def cabin(b, sens, w=w, t=t):
            T = sens.read("T_cabin", "temperature", w["T_C"].to_numpy(), t=t)[0] \
                + b.get("T_cabin", 0.0)
            return _time_mean_window(t, T)

        def chamber_mean(b, sens, w=w, t=t):
            T = chamber.measured_temperature(w["T_chamber_C"].to_numpy(), sens,
                                             b.get("T_chamber", 0.0), t=t)
            return _time_mean_window(t, T)

        both = [Channel("T_cabin", "temperature"),
                Channel("T_chamber", "temperature", chamber.n_probes)]
        if q == "cabin_T_mean":
            make(q, cabin, both[:1])
        elif q == "chamber_T_mean":
            make(q, chamber_mean, both[1:])
        elif q == "dT_cabin_chamber_mean":
            make(q, lambda b, s: cabin(b, s) - chamber_mean(b, s), both)
        elif q == "k_value":
            Q = _time_mean_window(t, w["Q_int_W"].to_numpy())
            make(q, lambda b, s: Q / (model.envelope_area_m2
                                      * (cabin(b, s) - chamber_mean(b, s))), both)
        elif q == "cabin_RH_mean":
            if not humid:
                r.problems.append("cabin_RH_mean needs the humidity state (chamber_RH)")
                continue

            def rh(b, sens, w=w, t=t):
                RH = sens.read("RH_cabin", "humidity", w["RH"].to_numpy(), t=t)[0] \
                    + b.get("RH_cabin", 0.0)
                return _time_mean_window(t, RH)
            make(q, rh, [Channel("RH_cabin", "humidity")])
    return out


# ---- unit test room --------------------------------------------------------

def _run_room(proc, r, room: UnitTestRoom, unit: DXUnit, sensors) -> TestResult:
    pf = proc.file
    cond, meas = None, None
    validity: list[str] = []
    notes = ["unit test room is a steady-state model: hold steps are no-ops and the "
             "measurement duration is recorded only"]
    for i, (kind, s) in enumerate(proc.steps):
        where = f"steps[{i}].{kind}"
        if kind == "set_conditions":
            cond = _room_conditions(s, r, where)
        elif kind in ("hold", "hold_until_stable"):
            continue
        elif kind == "measure":
            r.resolve(s.duration, "h", where)
            if cond is None:
                raise ProcedureError(pf.name, [f"{where}: set_conditions first"])
            ff = cond["fresh_fraction"]
            if ff == "none":
                ff = 0.0
            elif ff == "from_rating_reference":
                ff = unit.rating_fresh_fraction
            mode, plr = cond["unit"]
            meas = room.measure(unit, cond["indoor"], cond["outdoor"], mode, plr, ff, sensors)
            validity += meas.validity
            notes += meas.notes
    for eid in ("E-004", "E-006", "E-007"):
        r.use_evidence_id(eid, "psychrometrics")
    units = QUANTITIES["unit_room"]
    measurements = {q: Measurement(q, meas.measured[q].value, meas.measured[q].uncertainty,
                                   meas.measured[q].uncertainty_rss, units[q], meas.true[q])
                    for q in pf.measured_quantities}
    criteria = [_evaluate(c, measurements[c.quantity], r, units[c.quantity])
                for c in pf.acceptance]
    return TestResult(pf.name, pf.standard, pf.clause, pf.facility, measurements, criteria,
                      sorted(r.evidence_used.values(), key=lambda u: u.id),
                      list(r.user_defined.values()), validity, notes)
