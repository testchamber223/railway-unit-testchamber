"""HVAC unit model, level L1: datasheet values + generic DX performance curve.

Method (EnergyPlus Engineering Reference, "Single-Speed Electric DX Air
Cooling Coil"):

* Total capacity Q = Q_rated * f(x, y) / f(x_rating, y_rating), with the
  biquadratic f = a + b x + c x^2 + d y + e y^2 + f x y (x = coil inlet
  wet-bulb, y = outdoor dry-bulb, degC; coefficients E-042..E-047). The
  normalisation returns exactly the datasheet capacity at the datasheet's own
  rating condition.
* Sensible/latent split by the apparatus-dew-point / bypass-factor method:
  the rated bypass factor BF follows from the rated point and rated SHR; it is
  carried to other flows by A0 = -ln(BF_rated) * m_rated, BF = exp(-A0 / m).
* Dry coil: if the ADP humidity ratio is above the inlet humidity ratio, the
  curve wet-bulb is iterated until the coil is just dry (SHR = 1).
* Cycling compressor with continuous fan: at part-load fraction PLR the
  outlet enthalpy and humidity ratio are the PLR-weighted mix of the
  full-load outlet and the inlet. No part-load degradation curve, no
  airflow-fraction capacity curve (L1 assumptions).

Units: degC, kg/kg dry air, J/kg dry air, kg dry air/s, W.
Unit data come from a unit card (data/units/*.yaml); generic curve data and
fallbacks from the evidence registry. Nothing physical is hard-coded here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from scipy.optimize import brentq

from core import psychro
from evidence import registry as _evidence
from evidence.registry import MissingEvidenceError, Registry

UNITS_DIR = Path(__file__).parents[1] / "data" / "units"
CURVE_IDS = ("E-042", "E-043", "E-044", "E-045", "E-046", "E-047")
LIMITS_ID = "E-048"
SHR_FALLBACK_ID = "E-050"

FanPosition = Literal["draw-through", "blow-through"]
Mode = Literal["cooling", "heating", "ventilation"]

# Numerics (not physics)
_MAX_DRY_ITER = 50
_DRY_REL_TOL = 1e-8


# ==========================================================================
# Unit card: data/units/<unit_id>.yaml
# ==========================================================================

class CardField(BaseModel):
    """One datasheet entry: value with unit and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    value: float | str | None = None
    unit: str = "-"
    source: str = ""
    page: str = ""
    status: Literal["datasheet", "standard", "literature", "assumption"]

    @field_validator("value", mode="before")
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("page", mode="before")
    @classmethod
    def _page_as_text(cls, v: object) -> object:
        return "" if v is None else str(v)

    @model_validator(mode="after")
    def _check(self) -> CardField:
        if self.status != "assumption":
            if self.value is None:
                raise ValueError("empty value is only allowed for status 'assumption'")
            if not self.source:
                raise ValueError("empty source is only allowed for status 'assumption'")
        return self

    @property
    def is_open(self) -> bool:
        return self.status == "assumption" or self.value is None


# field name -> (SI unit it converts to, required for the cooling/heating model)
CARD_FIELDS: dict[str, tuple[str, bool]] = {
    "rated_total_cooling_capacity": ("W", True),
    "rating_outdoor_dry_bulb": ("degC", True),
    "rating_indoor_dry_bulb": ("degC", True),
    "rating_indoor_wet_bulb": ("degC", False),   # wet-bulb OR RH required
    "rating_indoor_rh": ("-", False),
    "rated_sensible_capacity": ("W", False),     # or SHR; else evidence E-050
    "rated_shr": ("-", False),
    "supply_airflow": ("m3/s", True),
    "fresh_airflow": ("m3/s", True),
    "compressor_power": ("W", False),
    "fan_power": ("W", True),
    "fan_position": ("text", True),
    "heating_capacity": ("W", True),
    "heating_rating_outdoor_dry_bulb": ("degC", False),
    "refrigerant": ("text", True),
}

# Unit conversions accepted on cards: unit -> (SI unit, factor)
_TO_SI: dict[str, tuple[str, float]] = {
    "W": ("W", 1.0), "kW": ("W", 1e3),
    "degC": ("degC", 1.0),
    "m3/s": ("m3/s", 1.0), "m3/h": ("m3/s", 1.0 / 3600.0),
    "-": ("-", 1.0), "%": ("-", 1e-2),
}


class UnitCard(BaseModel):
    """Datasheet card of an HVAC unit (schema of data/units/<unit_id>.yaml)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit_id: str
    name: str
    notes: str = ""
    rated_total_cooling_capacity: CardField | None = None
    rating_outdoor_dry_bulb: CardField | None = None
    rating_indoor_dry_bulb: CardField | None = None
    rating_indoor_wet_bulb: CardField | None = None
    rating_indoor_rh: CardField | None = None
    rated_sensible_capacity: CardField | None = None
    rated_shr: CardField | None = None
    supply_airflow: CardField | None = None
    fresh_airflow: CardField | None = None
    compressor_power: CardField | None = None
    fan_power: CardField | None = None
    fan_position: CardField | None = None
    heating_capacity: CardField | None = None
    heating_rating_outdoor_dry_bulb: CardField | None = None
    refrigerant: CardField | None = None

    @model_validator(mode="after")
    def _check_units(self) -> UnitCard:
        for name, (si, _) in CARD_FIELDS.items():
            f = getattr(self, name)
            if f is None:
                continue
            if si == "text":
                if f.value is not None and not isinstance(f.value, str):
                    raise ValueError(f"{name}: expected text, got {f.value!r}")
            elif f.unit not in _TO_SI or _TO_SI[f.unit][0] != si:
                raise ValueError(f"{name}: unit {f.unit!r} cannot be converted to {si}")
            elif f.value is not None and not isinstance(f.value, float | int):
                raise ValueError(f"{name}: expected a number, got {f.value!r}")
        if self.fan_position is not None and self.fan_position.value is not None \
                and self.fan_position.value not in ("draw-through", "blow-through"):
            raise ValueError("fan_position must be 'draw-through' or 'blow-through'")
        return self

    def si(self, name: str) -> float | str | None:
        """Value of field ``name`` in SI units (text as is; None if absent or empty)."""
        f = getattr(self, name)
        if f is None or f.value is None:
            return None
        if CARD_FIELDS[name][0] == "text":
            return f.value
        return float(f.value) * _TO_SI[f.unit][1]

    def open_items(self) -> list[tuple[str, str]]:
        """Every field that is missing or marked assumption: [(field, reason)]."""
        items: list[tuple[str, str]] = []
        for name, (_, required) in CARD_FIELDS.items():
            f = getattr(self, name)
            if f is None:
                if required:
                    items.append((name, "missing"))
                continue
            if f.status == "assumption":
                state = "value empty" if f.value is None else f"value {f.value} {f.unit}"
                items.append((name, f"assumption ({state})"))
        wb, rh = self.rating_indoor_wet_bulb, self.rating_indoor_rh
        if wb is None and rh is None:
            items.append(("rating_indoor_wet_bulb / rating_indoor_rh", "missing (one is required)"))
        if self.rated_shr is None and self.rated_sensible_capacity is None:
            items.append(("rated_shr / rated_sensible_capacity",
                          f"not on datasheet; falls back to evidence {SHR_FALLBACK_ID}"))
        return items

    def report(self) -> str:
        """Human-readable list of open items for reports."""
        items = self.open_items()
        if not items:
            return f"{self.name}: all datasheet fields present."
        lines = "\n".join(f"  - {name}: {why}" for name, why in items)
        return f"{self.name}: {len(items)} open datasheet item(s):\n{lines}"


def load_unit_card(path_or_id: Path | str) -> UnitCard:
    """Load and validate a unit card from a path or a unit id in data/units/."""
    path = Path(path_or_id)
    if not path.suffix:
        path = UNITS_DIR / f"{path_or_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return UnitCard.model_validate(data)


# ==========================================================================
# Generic performance curve
# ==========================================================================

def curve_coefficients(evidence: Registry | None = None) -> tuple[float, ...]:
    """Capacity-curve coefficients a..f from evidence E-042..E-047."""
    number = evidence.number if evidence is not None else _evidence.number
    return tuple(number(i, "-") for i in CURVE_IDS)


def capacity_modifier(Twb_in: float, Tdb_out: float,
                      coefficients: tuple[float, ...] | None = None) -> float:
    """Total-capacity modifier f(x, y) [-] (biquadratic, CAPFT).

        f = a + b x + c x^2 + d y + e y^2 + f x y,
        x = coil inlet wet-bulb Twb_in [degC], y = outdoor dry-bulb Tdb_out [degC]

    EnergyPlus Engineering Reference, Single-Speed Electric DX Air Cooling
    Coil, total cooling capacity modifier as a function of temperature.
    Coefficients default to evidence E-042..E-047.
    """
    a, b, c, d, e, f = coefficients or curve_coefficients()
    x, y = Twb_in, Tdb_out
    return a + b * x + c * x * x + d * y + e * y * y + f * x * y


class CurveLimits(NamedTuple):
    """Validity range of the capacity curve [degC]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def from_text(cls, text: str) -> CurveLimits:
        """Parse 'x_min..x_max; y_min..y_max' (format of evidence E-048)."""
        m = re.fullmatch(r"\s*([-\d.]+)\s*\.\.\s*([-\d.]+)\s*;\s*([-\d.]+)\s*\.\.\s*([-\d.]+)\s*",
                         text)
        if not m:
            raise ValueError(f"cannot parse curve limits {text!r}; expected 'a..b; c..d'")
        return cls(*(float(g) for g in m.groups()))

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def curve_limits(evidence: Registry | None = None) -> CurveLimits | None:
    """Curve limits from evidence E-048, or None while it is empty."""
    get = evidence.get if evidence is not None else _evidence.get
    value = get(LIMITS_ID).value
    return None if value is None else CurveLimits.from_text(str(value))


# ==========================================================================
# Air-side components
# ==========================================================================

class AirState(NamedTuple):
    T_C: float
    W: float            # kg/kg dry air
    m_da: float         # kg dry air / s


def mixing_box(return_air: AirState, fresh_air: AirState) -> AirState:
    """Adiabatic mixing of two dry-air mass flows (ASHRAE Ch. 1, adiabatic mixing).

        m = m_r + m_f;  W = (m_r W_r + m_f W_f) / m;  h = (m_r h_r + m_f h_f) / m;
        T = T(h, W)
    """
    m = return_air.m_da + fresh_air.m_da
    if not m > 0:
        raise ValueError("mixing_box needs a positive total mass flow")
    W = (return_air.m_da * return_air.W + fresh_air.m_da * fresh_air.W) / m
    h = (return_air.m_da * psychro.enthalpy(return_air.T_C, return_air.W)
         + fresh_air.m_da * psychro.enthalpy(fresh_air.T_C, fresh_air.W)) / m
    return AirState(psychro.dry_bulb_from_enthalpy(h, W), W, m)


def add_sensible_heat(air: AirState, Q_W: float) -> AirState:
    """Add sensible heat Q [W] to a stream at constant W: h_out = h_in + Q / m."""
    h = psychro.enthalpy(air.T_C, air.W) + Q_W / air.m_da
    return AirState(psychro.dry_bulb_from_enthalpy(h, air.W), air.W, air.m_da)


def bypass_factor_from_rating(
    T_in: float, W_in: float, Q_total_W: float, m_da: float, shr: float, p: float | None = None,
) -> tuple[float, float, float]:
    """Rated bypass factor by the apparatus-dew-point method: (BF, T_adp, W_adp).

    From the rated inlet (T_in, W_in), total capacity, mass flow and SHR:
        h_out = h_in - Q / m
        W_out = W(T_in, h_in - (1 - SHR) Q / m)     T_out = T(h_out, W_out)
    The straight line through inlet and outlet in (h, W) is extended to the
    saturation curve: that point is the ADP. Then
        BF = (h_out - h_adp) / (h_in - h_adp).
    EnergyPlus Engineering Reference, Single-Speed Electric DX Air Cooling Coil
    (bypass factor from rated conditions).
    """
    if not 0 < shr <= 1:
        raise ValueError(f"rated SHR={shr} must be in (0, 1]")
    h_in = psychro.enthalpy(T_in, W_in)
    dh = Q_total_W / m_da
    h_out = h_in - dh
    W_out = psychro.humidity_ratio_from_enthalpy(T_in, h_in - (1 - shr) * dh)
    T_out = psychro.dry_bulb_from_enthalpy(h_out, W_out)
    if W_out >= psychro.sat_humidity_ratio(T_out, p):
        raise ValueError("rated outlet state is saturated: rated capacity/SHR/airflow "
                         "are inconsistent")
    slope = (W_in - W_out) / (h_in - h_out)        # dW/dh along the process line

    def gap(T: float) -> float:
        h_s = psychro.sat_air_enthalpy(T, p)
        return psychro.sat_humidity_ratio(T, p) - (W_out + slope * (h_s - h_out))

    T_adp = brentq(gap, psychro.T_MIN, T_out, xtol=1e-10)
    W_adp = psychro.sat_humidity_ratio(T_adp, p)
    h_adp = psychro.enthalpy(T_adp, W_adp)
    bf = (h_out - h_adp) / (h_in - h_adp)
    if not 0 < bf < 1:
        raise ValueError(f"rated bypass factor {bf:.3f} is outside (0, 1)")
    return bf, T_adp, W_adp


class CoilResult(NamedTuple):
    T_out_C: float
    W_out: float
    Q_total_W: float          # = m (h_in - h_out)
    Q_sensible_W: float       # = m (h(T_in, W_out) - h_out)
    shr: float
    condensate_kg_s: float    # = m (W_in - W_out)
    wet: bool
    extrapolated: bool | None  # None while curve limits (E-048) are unknown
    Twb_curve_C: float        # wet-bulb used in the capacity curve


class UnitResult(NamedTuple):
    supply: AirState
    mixed: AirState
    m_fresh_kg_s: float
    coil: CoilResult | None
    Q_fan_W: float
    Q_heater_W: float


# ==========================================================================
# DX unit
# ==========================================================================

@dataclass(frozen=True)
class DXUnit:
    """Single-speed DX unit with electric heater, fan and mixing box (level L1).

    All attributes in SI units; build from a card with ``DXUnit.from_card``.
    """

    name: str
    Q_rated_W: float
    T_rating_out_C: float
    T_rating_in_C: float
    W_rating_in: float
    shr_rated: float
    V_supply_m3_s: float
    V_fresh_m3_s: float
    P_fan_W: float
    fan_position: FanPosition
    Q_heat_W: float
    coefficients: tuple[float, ...]
    limits: CurveLimits | None = None
    p: float | None = None

    @classmethod
    def from_card(cls, card: UnitCard, evidence: Registry | None = None,
                  limits: CurveLimits | None | str = "evidence") -> DXUnit:
        """Build the model from a unit card; refuse while required data are missing.

        ``limits="evidence"`` reads E-048 (None while empty); pass a
        ``CurveLimits`` or None to override.

        Raises:
            MissingEvidenceError: lists every card field (and evidence ID) that
                is required but missing or empty.
        """
        missing: list[str] = []

        def need(name: str):
            v = card.si(name)
            if v is None:
                missing.append(name)
            return v

        Q = need("rated_total_cooling_capacity")
        T_out = need("rating_outdoor_dry_bulb")
        T_in = need("rating_indoor_dry_bulb")
        wb, rh = card.si("rating_indoor_wet_bulb"), card.si("rating_indoor_rh")
        if wb is None and rh is None:
            missing.append("rating_indoor_wet_bulb / rating_indoor_rh")
        V_sup = need("supply_airflow")
        V_fresh = need("fresh_airflow")
        P_fan = need("fan_power")
        fan_pos = need("fan_position")
        Q_heat = need("heating_capacity")

        shr = card.si("rated_shr")
        if shr is None and card.si("rated_sensible_capacity") is not None and Q:
            shr = card.si("rated_sensible_capacity") / Q
        if shr is None:
            get = evidence.get if evidence is not None else _evidence.get
            fallback = get(SHR_FALLBACK_ID).value
            if fallback is None:
                missing.append(f"rated SHR ({SHR_FALLBACK_ID} empty)")
            else:
                shr = float(fallback)

        if missing:
            raise MissingEvidenceError(
                missing, f"{card.name}: cannot build the unit model; missing: "
                + ", ".join(missing))

        W_in = (psychro.humidity_ratio_from_wet_bulb(T_in, wb) if wb is not None
                else psychro.humidity_ratio(T_in, rh))
        return cls(
            name=card.name, Q_rated_W=Q, T_rating_out_C=T_out, T_rating_in_C=T_in,
            W_rating_in=W_in, shr_rated=shr, V_supply_m3_s=V_sup, V_fresh_m3_s=V_fresh,
            P_fan_W=P_fan, fan_position=fan_pos, Q_heat_W=Q_heat,
            coefficients=curve_coefficients(evidence),
            limits=curve_limits(evidence) if limits == "evidence" else limits,
        )

    # ---- derived rating quantities ---------------------------------------

    @cached_property
    def m_supply_kg_s(self) -> float:
        """Supply (= rated coil) dry-air mass flow: V_supply / v(rating inlet) [kg/s]."""
        return self.V_supply_m3_s / psychro.specific_volume(
            self.T_rating_in_C, self.W_rating_in, self.p)

    @cached_property
    def Twb_rating_C(self) -> float:
        return psychro.wet_bulb_from_humidity_ratio(self.T_rating_in_C, self.W_rating_in, self.p)

    @cached_property
    def f_rating(self) -> float:
        return capacity_modifier(self.Twb_rating_C, self.T_rating_out_C, self.coefficients)

    @cached_property
    def bypass_rated(self) -> tuple[float, float, float]:
        """(BF, T_adp, W_adp) at the rating point."""
        return bypass_factor_from_rating(self.T_rating_in_C, self.W_rating_in, self.Q_rated_W,
                                         self.m_supply_kg_s, self.shr_rated, self.p)

    @cached_property
    def A0(self) -> float:
        """Coil constant A0 = -ln(BF_rated) * m_rated [kg/s]."""
        return -math.log(self.bypass_rated[0]) * self.m_supply_kg_s

    # ---- components -------------------------------------------------------

    def total_capacity(self, Twb_in: float, Tdb_out: float) -> float:
        """Full-load total capacity [W]: Q_rated * f(Twb_in, Tdb_out) / f(rating)."""
        return self.Q_rated_W * capacity_modifier(Twb_in, Tdb_out, self.coefficients) \
            / self.f_rating

    def bypass_factor(self, m_da: float) -> float:
        """BF at mass flow m_da [kg/s]: exp(-A0 / m)."""
        return math.exp(-self.A0 / m_da)

    def coil(self, T_in: float, W_in: float, m_da: float, Tdb_out: float,
             part_load: float) -> CoilResult:
        """DX cooling coil at inlet (T_in [degC], W_in [kg/kg]), m_da [kg/s].

        Tdb_out [degC] is the outdoor (condenser inlet) dry-bulb; part_load in
        0..1 is the compressor on-fraction. See the module docstring for the
        method (EnergyPlus Engineering Reference, Single-Speed Electric DX Air
        Cooling Coil).
        """
        if not 0.0 <= part_load <= 1.0:
            raise ValueError(f"part_load={part_load} must be in 0..1")
        if not m_da > 0:
            raise ValueError(f"m_da={m_da} must be > 0")
        p = self.p
        h_in = psychro.enthalpy(T_in, W_in)
        bf = self.bypass_factor(m_da)

        W_curve = W_in
        wet = True
        for i in range(_MAX_DRY_ITER):
            Twb = psychro.wet_bulb_from_humidity_ratio(T_in, W_curve, p)
            Q_full = self.total_capacity(Twb, Tdb_out)
            dh = Q_full / m_da
            h_adp = h_in - dh / (1 - bf)
            T_adp = psychro.sat_temperature_from_enthalpy(h_adp, p)
            W_adp = psychro.sat_humidity_ratio(T_adp, p)
            if i == 0 and W_adp < W_in:
                shr = min((psychro.enthalpy(T_in, W_adp) - h_adp) / (h_in - h_adp), 1.0)
                break
            wet = False   # dry coil: iterate the curve wet-bulb to the dry boundary
            if abs(W_curve - W_adp) <= _DRY_REL_TOL * max(W_curve, 1e-5):
                break
            W_curve = W_adp
        else:
            raise RuntimeError(f"{self.name}: dry-coil iteration did not converge")
        if not wet:
            shr = 1.0

        # Full-load outlet
        h_out_f = h_in - dh
        W_out_f = min(W_in, psychro.humidity_ratio_from_enthalpy(T_in, h_in - (1 - shr) * dh))
        T_out_f = psychro.dry_bulb_from_enthalpy(h_out_f, W_out_f)
        if W_out_f > psychro.sat_humidity_ratio(T_out_f, p):   # clamp to saturation
            T_out_f = psychro.sat_temperature_from_enthalpy(h_out_f, p)
            W_out_f = psychro.sat_humidity_ratio(T_out_f, p)

        # Part load: continuous fan, cycling compressor
        if part_load == 0.0:             # compressor off: air passes unchanged
            h_out, W_out, T_out = h_in, W_in, T_in
        else:
            h_out = h_in - part_load * (h_in - h_out_f)
            W_out = W_in - part_load * (W_in - W_out_f)
            T_out = psychro.dry_bulb_from_enthalpy(h_out, W_out)

        Q_total = m_da * (h_in - h_out)
        Q_sens = m_da * (psychro.enthalpy(T_in, W_out) - h_out)
        extrapolated = None if self.limits is None else not self.limits.contains(Twb, Tdb_out)
        return CoilResult(
            T_out_C=T_out, W_out=W_out, Q_total_W=Q_total, Q_sensible_W=Q_sens,
            shr=Q_sens / Q_total if Q_total > 0 else 1.0,
            condensate_kg_s=m_da * (W_in - W_out), wet=wet and part_load > 0,
            extrapolated=extrapolated, Twb_curve_C=Twb,
        )

    def heater(self, air: AirState, part_load: float) -> tuple[AirState, float]:
        """Electric heater: Q = part_load * Q_heat_rated [W], added at constant W."""
        if not 0.0 <= part_load <= 1.0:
            raise ValueError(f"part_load={part_load} must be in 0..1")
        Q = part_load * self.Q_heat_W
        return add_sensible_heat(air, Q), Q

    def operate(self, T_return: float, W_return: float, T_fresh: float, W_fresh: float,
                Tdb_out: float, mode: Mode, part_load: float) -> UnitResult:
        """Whole unit: mixing box -> (fan) -> coil -> (fan) -> heater.

        Supply mass flow is constant (m_supply_kg_s); fresh dry-air mass flow is
        V_fresh / v(fresh state); return = supply - fresh. Fan heat P_fan is
        added to the air before the coil (blow-through) or after it
        (draw-through), assuming the motor is in the air stream.
        """
        if mode not in ("cooling", "heating", "ventilation"):
            raise ValueError(f"unknown mode {mode!r}")
        m_s = self.m_supply_kg_s
        m_f = self.V_fresh_m3_s / psychro.specific_volume(T_fresh, W_fresh, self.p)
        if m_f > m_s:
            raise ValueError("fresh airflow exceeds supply airflow")
        air = mixing_box(AirState(T_return, W_return, m_s - m_f), AirState(T_fresh, W_fresh, m_f))
        mixed = air
        if self.fan_position == "blow-through":
            air = add_sensible_heat(air, self.P_fan_W)
        coil = None
        if mode == "cooling":
            coil = self.coil(air.T_C, air.W, air.m_da, Tdb_out, part_load)
            air = AirState(coil.T_out_C, coil.W_out, air.m_da)
        if self.fan_position == "draw-through":
            air = add_sensible_heat(air, self.P_fan_W)
        Q_heat = 0.0
        if mode == "heating":
            air, Q_heat = self.heater(air, part_load)
        return UnitResult(supply=air, mixed=mixed, m_fresh_kg_s=m_f, coil=coil,
                          Q_fan_W=self.P_fan_W, Q_heater_W=Q_heat)
