"""Single-zone coach thermal and moisture model.

The coach interior is one well-mixed air zone with an effective heat capacity
(air + interior mass) and a fixed mass of dry air:

    C    * dT/dt = UA * (T_out - T) + Q_int - Q_hvac + sum(Q_load_i)
    M_da * dW/dt = sum(m_load_i) - m_cond                 (m_cond = 0 for now)

    T       interior air temperature              [degC]
    W       interior humidity ratio               [kg water / kg dry air]
    T_out   outside (chamber) air temperature     [degC]
    C       effective heat capacity               [J/K]
    UA      k-value * envelope area               [W/K]
    M_da    dry-air mass in the coach             [kg]  (= V / v(T0, W0, p))
    Q_int   internal gains (e.g. equipment)       [W]
    Q_hvac  heat removed by the HVAC unit         [W]  (negative = heating)
    Q_load_i, m_load_i  sensible heat [W] and water [kg/s] of each LoadTerm
                        (passengers, fresh air, solar, ...; see core.loads)
    t       time                                  [s]

"With unit" mode (``simulate(..., unit=UnitOperation(...))``): the HVAC unit
(core.unit) supplies m_s of conditioned air at (T_s, W_s); the same mass leaves
at the zone state. Fresh air enters ONLY through the unit's mixing box, so a
separate fresh-air load is rejected in this mode. The supply stream adds
    Q_supply = m_s (h(T_s, W) - h(T, W)),   m_supply = m_s (W_s - W)
to the energy and moisture balances (core.loads.supply_air).

Assumptions: perfectly mixed air, lumped capacity, steady envelope
conduction (no wall dynamics), constant dry-air mass, no condensation yet.
Coach parameters come from the caller or, for the reference coach, from
data/reference_coach.yaml via the evidence registry; nothing physical is
hard-coded here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

from core import loads as _loads
from core import psychro
from core.unit import DXUnit, Mode, UnitResult
from evidence import registry as _evidence
from evidence.registry import EvidenceError, MissingEvidenceError, Registry

Input = float | Callable[[float], float]
"""A model input: a constant, or a function of time t [s]."""

DEFAULT_COACH_YAML = Path(__file__).parents[1] / "data" / "reference_coach.yaml"


@dataclass(frozen=True)
class SingleZoneCoach:
    """Lumped single-zone coach.

    Attributes:
        name: identifier.
        envelope_area_m2: heat-transfer area Ae the k-value refers to [m2]
            (definition: evidence E-008).
        k_value_W_m2K: overall heat-transfer coefficient k [W/(m2.K)].
        heat_capacity_J_K: effective heat capacity C, air + interior mass [J/K].
        air_volume_m3: interior air volume [m3]; needed only for humidity.
    """

    name: str
    envelope_area_m2: float
    k_value_W_m2K: float
    heat_capacity_J_K: float
    air_volume_m3: float | None = None

    def __post_init__(self) -> None:
        for name in ("envelope_area_m2", "k_value_W_m2K", "heat_capacity_J_K", "air_volume_m3"):
            v = getattr(self, name)
            if v is None and name == "air_volume_m3":
                continue
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{self.name}: {name}={v} must be finite and > 0")

    @property
    def UA(self) -> float:
        """Envelope conductance UA = k * A [W/K]."""
        return self.k_value_W_m2K * self.envelope_area_m2


@dataclass(frozen=True)
class LoadTerm:
    """A load that may depend on time and on the interior state.

    ``fn(t [s], T_in [degC], W_in [kg/kg]) -> (Q_sensible [W], m_water [kg/s])``,
    both positive into the coach. Reported as columns ``Q_<name>_W`` and
    ``m_<name>_kg_s``. Set ``is_fresh_air`` for fresh-air terms so that the
    "with unit" mode can refuse them (fresh air then enters via the unit).
    """

    name: str
    fn: Callable[[float, float, float], tuple[float, float]]
    is_fresh_air: bool = False


@dataclass(frozen=True)
class UnitOperation:
    """How the HVAC unit runs in "with unit" mode.

    Attributes:
        unit: the DX unit model (core.unit.DXUnit).
        mode: "cooling", "heating" or "ventilation".
        part_load: compressor (cooling) or heater (heating) fraction 0..1,
            constant for now (controller: piece 7).
        RH_out: outside relative humidity 0..1 of the fresh air, constant or
            f(t [s]); outside temperature is simulate()'s T_out.
    """

    unit: DXUnit
    mode: Mode
    part_load: float
    RH_out: Input

    def __post_init__(self) -> None:
        if not 0.0 <= self.part_load <= 1.0:
            raise ValueError(f"part_load={self.part_load} must be in 0..1")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _as_function(x: Input) -> Callable[[float], float]:
    if callable(x):
        return x
    value = float(x)
    return lambda t: value


def _require_constant(name: str, x: Input) -> float:
    if callable(x):
        raise TypeError(f"{name} must be a constant for this function, not a function of time")
    return float(x)


def time_constant_s(coach: SingleZoneCoach) -> float:
    """Time constant tau = C / UA [s]."""
    return coach.heat_capacity_J_K / coach.UA


def steady_state_T(coach: SingleZoneCoach, T_out: float, Q_int: float, Q_hvac: float) -> float:
    """Steady-state interior temperature [degC] for constant inputs (no load terms).

    Setting dT/dt = 0:  T_ss = T_out + (Q_int - Q_hvac) / UA,
    with T_out [degC], Q_int and Q_hvac [W], UA [W/K].
    """
    T_out = _require_constant("T_out", T_out)
    Q_int = _require_constant("Q_int", Q_int)
    Q_hvac = _require_constant("Q_hvac", Q_hvac)
    return T_out + (Q_int - Q_hvac) / coach.UA


def dry_air_mass_kg(coach: SingleZoneCoach, T: float, W: float, p: float | None = None) -> float:
    """Dry-air mass in the coach [kg]: M_da = V / v(T, W, p), v from ASHRAE Eq. 26."""
    if coach.air_volume_m3 is None:
        raise ValueError(f"{coach.name}: air_volume_m3 is needed to model humidity")
    return coach.air_volume_m3 / psychro.specific_volume(T, W, p)


def net_sensible_W(df: pd.DataFrame) -> pd.Series:
    """Net sensible heat into the coach [W] from a ``simulate`` result (= C dT/dt)."""
    q = df["Q_env_W"] + df["Q_int_W"] - df["Q_hvac_W"]
    for col in df.columns:
        if col.startswith("Q_load_"):
            q = q + df[col]
    return q


def net_moisture_kg_s(df: pd.DataFrame) -> pd.Series:
    """Net water flow into the coach air [kg/s] from a ``simulate`` result (= M_da dW/dt)."""
    m = -df["m_cond_kg_s"]
    for col in df.columns:
        if col.startswith("m_load_"):
            m = m + df[col]
    return m


def _output_times(t_end_s: float, dt_out_s: float) -> np.ndarray:
    if not t_end_s > 0:
        raise ValueError(f"t_end_s={t_end_s} must be > 0")
    if not dt_out_s > 0:
        raise ValueError(f"dt_out_s={dt_out_s} must be > 0")
    n = int(math.floor(t_end_s / dt_out_s + 1e-9))
    t = dt_out_s * np.arange(n + 1)
    if t_end_s - t[-1] > 1e-9 * t_end_s:
        t = np.append(t, t_end_s)
    t[-1] = min(t[-1], t_end_s)
    return t


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(
    coach: SingleZoneCoach,
    T0: float,
    T_out: Input,
    Q_int: Input,
    Q_hvac: Input,
    t_end_s: float,
    dt_out_s: float,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_step: float = np.inf,
    *,
    W0: float | None = None,
    loads: Sequence[LoadTerm] = (),
    p: float | None = None,
    unit: UnitOperation | None = None,
) -> pd.DataFrame:
    """Integrate the coach energy (and, if W0 is given, moisture) balance.

        C    dT/dt = UA (T_out - T) + Q_int - Q_hvac + sum(Q_load_i)
        M_da dW/dt = sum(m_load_i) - m_cond,   m_cond = 0 (not modelled yet)

    Args:
        coach: the coach model.
        T0: initial interior temperature [degC].
        T_out: outside temperature [degC], constant or f(t [s]).
        Q_int: internal gains [W], constant or f(t [s]).
        Q_hvac: heat removed by the HVAC unit [W], constant or f(t [s]).
        t_end_s: end time [s].
        dt_out_s: output interval [s] (the solver chooses its own internal steps).
        rtol, atol: solver tolerances (atol applies to T [K] and W [kg/kg]).
        max_step: largest internal step [s]; set it below the shortest feature
            of a time-varying input (e.g. a step change) so it is not skipped.
        W0: initial humidity ratio [kg/kg dry air]. If None, humidity is not
            modelled and W_kg_kg / RH are NaN. Requires coach.air_volume_m3.
        loads: extra state-dependent loads (requires W0).
        p: pressure [Pa] for dry-air mass and RH; defaults to E-004.
        unit: run in "with unit" mode (requires W0; no fresh-air load terms).

    Returns:
        DataFrame with columns t_s [s], T_C [degC], W_kg_kg [kg/kg], RH [0-1],
        Q_env_W [W] (= UA (T_out - T), positive into the coach), Q_int_W [W],
        Q_hvac_W [W], m_cond_kg_s [kg/s], and Q_load_<name>_W [W] /
        m_load_<name>_kg_s [kg/s] per load term. ``df.attrs["M_da_kg"]`` holds
        the dry-air mass (NaN if humidity is not modelled). In "with unit" mode
        the supply stream appears as Q_load_unit_supply_W / m_load_unit_supply_kg_s
        and unit details as unit_* columns (supply/mixed state, coil total and
        sensible capacity, SHR, condensate, wet flag, extrapolation flag, fan and
        heater heat, fresh-air enthalpy flow m_f (h_out - h_zone)).

    Solver: scipy.integrate.solve_ivp, method LSODA.
    """
    f_out, f_int, f_hvac = _as_function(T_out), _as_function(Q_int), _as_function(Q_hvac)
    UA, C = coach.UA, coach.heat_capacity_J_K
    t_eval = _output_times(t_end_s, dt_out_s)
    loads = tuple(loads)
    humid = W0 is not None
    if loads and not humid:
        raise ValueError("load terms need the humidity state: pass W0")
    if len({term.name for term in loads}) != len(loads):
        raise ValueError("load term names must be unique")
    if unit is not None:
        if not humid:
            raise ValueError("'with unit' mode needs the humidity state: pass W0")
        fresh = [t.name for t in loads if t.is_fresh_air or t.name == "fresh_air"]
        if fresh:
            raise ValueError(f"'with unit' mode: fresh air enters via the unit's mixing box; "
                             f"remove load term(s) {fresh}")
        f_rh = _as_function(unit.RH_out)
    M_da = dry_air_mass_kg(coach, T0, W0, p) if humid else math.nan

    def unit_eval(t: float, T: float, W: float) -> tuple[UnitResult, float, float, float]:
        T_o = f_out(t)
        W_f = psychro.humidity_ratio(T_o, f_rh(t), p)
        res = unit.unit.operate(T, W, T_o, W_f, T_o, unit.mode, unit.part_load)
        s = res.supply
        q_s, _, m_w = _loads.supply_air(s.m_da, s.T_C, s.W, T, W)
        return res, q_s, m_w, W_f

    def m_cond(t: float, T: float, W: float) -> float:
        return 0.0  # condensation on the evaporator: not modelled yet

    def rhs(t: float, y: np.ndarray) -> list[float]:
        T = y[0]
        W = y[1] if humid else math.nan
        Q = UA * (f_out(t) - T) + f_int(t) - f_hvac(t)
        m = -m_cond(t, T, W)
        for term in loads:
            q_i, m_i = term.fn(t, T, W)
            Q += q_i
            m += m_i
        if unit is not None:
            _, q_s, m_w, _ = unit_eval(t, T, W)
            Q += q_s
            m += m_w
        return [Q / C, m / M_da] if humid else [Q / C]

    y0 = [float(T0), float(W0)] if humid else [float(T0)]
    sol = solve_ivp(
        rhs, (0.0, t_eval[-1]), y0, method="LSODA",
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"{coach.name}: solver failed: {sol.message}")

    t, T = sol.t, sol.y[0]
    W = sol.y[1] if humid else np.full_like(t, math.nan)
    RH = (np.array([psychro.rh_from_humidity_ratio(Ti, Wi, p) for Ti, Wi in zip(T, W)])
          if humid else np.full_like(t, math.nan))
    T_out_t = np.array([f_out(ti) for ti in t])
    df = pd.DataFrame({
        "t_s": t,
        "T_C": T,
        "W_kg_kg": W,
        "RH": RH,
        "Q_env_W": UA * (T_out_t - T),
        "Q_int_W": [f_int(ti) for ti in t],
        "Q_hvac_W": [f_hvac(ti) for ti in t],
        "m_cond_kg_s": [m_cond(ti, Ti, Wi) for ti, Ti, Wi in zip(t, T, W)],
    })
    for term in loads:
        q_m = np.array([term.fn(ti, Ti, Wi) for ti, Ti, Wi in zip(t, T, W)])
        df[f"Q_load_{term.name}_W"] = q_m[:, 0]
        df[f"m_load_{term.name}_kg_s"] = q_m[:, 1]
    if unit is not None:
        rows = []
        for ti, Ti, Wi in zip(t, T, W):
            res, q_s, m_w, W_f = unit_eval(ti, Ti, Wi)
            s, c = res.supply, res.coil
            h_z = psychro.enthalpy(Ti, Wi)
            rows.append({
                "Q_load_unit_supply_W": q_s,
                "m_load_unit_supply_kg_s": m_w,
                "unit_T_supply_C": s.T_C,
                "unit_W_supply_kg_kg": s.W,
                "unit_m_supply_kg_s": s.m_da,
                "unit_m_fresh_kg_s": res.m_fresh_kg_s,
                "unit_T_mixed_C": res.mixed.T_C,
                "unit_W_mixed_kg_kg": res.mixed.W,
                "unit_Q_coil_total_W": c.Q_total_W if c else 0.0,
                "unit_Q_coil_sensible_W": c.Q_sensible_W if c else 0.0,
                "unit_shr": c.shr if c else math.nan,
                "unit_m_condensate_kg_s": c.condensate_kg_s if c else 0.0,
                "unit_coil_wet": c.wet if c else False,
                "unit_extrapolated": c.extrapolated if c else None,
                "unit_Q_fan_W": res.Q_fan_W,
                "unit_Q_heater_W": res.Q_heater_W,
                "unit_Q_fresh_enthalpy_W":
                    res.m_fresh_kg_s * (psychro.enthalpy(f_out(ti), W_f) - h_z),
                "unit_Q_supply_total_W": s.m_da * (psychro.enthalpy(s.T_C, s.W) - h_z),
            })
        df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    df.attrs["M_da_kg"] = M_da
    return df


# --------------------------------------------------------------------------
# Coach definitions from YAML (every number resolved through the registry)
# --------------------------------------------------------------------------

COACH_FIELDS: dict[str, str] = {
    # YAML key -> unit the evidence row must carry
    "k_value_W_m2K": "W/m2K",
    "length_m": "m",
    "width_m": "m",
    "height_m": "m",
    "envelope_area_Ae_m2": "m2",
    "glazing_area_per_side_m2": "m2",
    "seats": "-",
    "heat_capacity_J_K": "J/K",
    "equipment_load_W": "W",
}

_EVIDENCE_ID = re.compile(r"^E-\d{3}$")


@dataclass(frozen=True)
class CoachDefinition:
    """A coach whose every number was resolved from the evidence registry.

    Field units follow the names (see ``COACH_FIELDS``); ``evidence_ids`` maps
    each field to the evidence ID it came from.
    """

    name: str
    k_value_W_m2K: float
    length_m: float
    width_m: float
    height_m: float
    envelope_area_Ae_m2: float
    glazing_area_per_side_m2: float
    seats: float
    heat_capacity_J_K: float
    equipment_load_W: float
    evidence_ids: dict[str, str] = field(default_factory=dict)

    @property
    def air_volume_m3(self) -> float:
        """Interior air volume V = length * width * height [m3]."""
        return self.length_m * self.width_m * self.height_m

    def thermal_model(self) -> SingleZoneCoach:
        """The single-zone model of this coach."""
        return SingleZoneCoach(
            name=self.name,
            envelope_area_m2=self.envelope_area_Ae_m2,
            k_value_W_m2K=self.k_value_W_m2K,
            heat_capacity_J_K=self.heat_capacity_J_K,
            air_volume_m3=self.air_volume_m3,
        )


def load_coach_definition(
    path: Path | str = DEFAULT_COACH_YAML, evidence: Registry | None = None,
) -> CoachDefinition:
    """Load a coach YAML and resolve every field through the evidence registry.

    Every field in ``COACH_FIELDS`` must be an evidence ID (``E-xxx``), not a
    number. All problems are collected before raising, so one error lists them.

    Raises:
        MissingEvidenceError: one or more IDs have an empty or non-numeric value
            (``.ids`` lists them); the message names each field and ID.
        EvidenceError: unknown IDs, numbers instead of IDs, missing fields or
            unit mismatches.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    get = evidence.get if evidence is not None else _evidence.get

    values: dict[str, float] = {}
    ids: dict[str, str] = {}
    missing: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for key, unit in COACH_FIELDS.items():
        ref = data.get(key)
        if ref is None:
            errors.append(f"{key}: field missing")
            continue
        if not (isinstance(ref, str) and _EVIDENCE_ID.match(ref)):
            errors.append(f"{key}: {ref!r} must be an evidence ID (E-xxx), not a value")
            continue
        ids[key] = ref
        try:
            rec = get(ref).record
        except KeyError:
            errors.append(f"{key}: unknown evidence id {ref}")
            continue
        if rec.unit != unit:
            errors.append(f"{key}: {ref} has unit {rec.unit!r}, expected {unit!r}")
        if rec.value is None:
            missing.append((ref, key, "empty"))
        elif not isinstance(rec.value, float):
            missing.append((ref, key, f"not a number: {rec.value!r}"))
        else:
            values[key] = rec.value

    if errors:
        raise EvidenceError(f"{path.name}: " + "; ".join(errors))
    if missing:
        lines = "\n".join(f"  {ref}  {key}  ({why})" for ref, key, why in missing)
        raise MissingEvidenceError(
            [ref for ref, _, _ in missing],
            f"{path.name}: cannot build coach {data.get('name')!r}; "
            f"{len(missing)} evidence value(s) missing:\n{lines}",
        )
    return CoachDefinition(name=str(data.get("name", path.stem)), evidence_ids=ids, **values)
