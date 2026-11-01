"""HVAC control: setpoint, controller types, virtual control sensor, sampled simulation.

Setpoint
    ``SetpointCurve`` -- piecewise-linear cabin setpoint as a function of the
    outside temperature, points from evidence E-110..E-115 (EN 13129 curve,
    believed 22 + 0.25 (T_out - 19) degC above 19 degC -- MUST be confirmed).
    Flat outside the first / last point. ``FixedSetpoint`` -- a user-defined
    constant.

Controller (``ControlParams`` from unit-card fields control_*)
    on_off    cooling on at  sp + db/2, off at sp - db/2 (capacity 1)
    staged    stage k (k = 1..n) on at sp + db/2 + (k-1) db, off at
              sp - db/2 + (k-1) db; capacity = stage_fractions[k-1]
    inverter  started at sp + db/2; PI on e = T - sp,
              u = clamp(kp e + I, u_min, u_max), I += ki e dt (anti-windup:
              no integration while saturated in the direction of e); stopped
              when T <= sp - db/2 at u = u_min
    heating   all types: heater on at sp_h - db_h/2, off at sp_h + db_h/2,
              sp_h = sp - neutral_zone. neutral_zone > (db + db_h)/2 is
              required so heating and cooling can never fight; mode changes
              between heating and cooling always pass through "off".
    Minimum on and off times apply to every start / stop (compressor or
    heater); stage and capacity changes while running are free.

Sensing
    The controller only sees a VIRTUAL sensor reading (``ControlSensor``):
    return_air (one probe) or cabin_mean (E-109 probes), each with the
    temperature sensor's lag, calibration error, resolution and sampling
    (core.sensors). The single-zone cabin is well mixed, so both locations see
    the same air temperature; they differ in probe count and errors.

Simulation (``simulate_controlled``)
    The controller acts at discrete sample times k * T_c. Between samples the
    coach (and sensor lag) are integrated with solve_ivp (dense output); the
    integration is restarted at every switching event so discontinuities are
    never smoothed. Each interval of constant controller output is a segment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from core.coach import (Input, LoadTerm, SingleZoneCoach, UnitOperation, ZoneSystem,
                        _as_function, dry_air_mass_kg)
from core.sensors import SensorSet, SensorType
from core.unit import DXUnit, UnitCard
from evidence import registry as _evidence
from evidence.registry import EvidenceError, MissingEvidenceError, Registry

SETPOINT_POINT_IDS: tuple[tuple[str, str], ...] = (("E-110", "E-111"), ("E-112", "E-113"),
                                                   ("E-114", "E-115"))
CABIN_PROBES_ID = "E-109"
ControlMode = Literal["off", "cooling", "heating"]


# ==========================================================================
# Setpoint
# ==========================================================================

@dataclass(frozen=True)
class SetpointCurve:
    """Cabin setpoint [degC] vs outside temperature [degC], piecewise linear."""

    points: tuple[tuple[float, float], ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        xs = [x for x, _ in self.points]
        if len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("setpoint curve needs >= 2 points with increasing outside "
                             "temperature")

    @classmethod
    def from_evidence(cls, evidence: Registry | None = None) -> SetpointCurve:
        """Points from E-110..E-115; refuses while any value is empty."""
        get = evidence.get if evidence is not None else _evidence.get
        missing, pts, ids = [], [], []
        for x_id, y_id in SETPOINT_POINT_IDS:
            vals = []
            for eid in (x_id, y_id):
                rec = get(eid).record
                ids.append(eid)
                if not isinstance(rec.value, float):
                    missing.append(eid)
                elif rec.unit != "degC":
                    raise EvidenceError(f"{eid}: unit {rec.unit!r}, expected 'degC'")
                vals.append(rec.value)
            pts.append(tuple(vals))
        if missing:
            raise MissingEvidenceError(
                missing, "setpoint curve: evidence value(s) empty: " + ", ".join(missing)
                + " (EN 13129 curve, MUST be confirmed from the standard)")
        return cls(tuple(pts), tuple(ids))

    def __call__(self, T_out: float) -> float:
        xs, ys = zip(*self.points)
        return float(np.interp(T_out, xs, ys))


@dataclass(frozen=True)
class FixedSetpoint:
    """A constant, user-defined cabin setpoint [degC]."""

    value: float
    note: str = "user-defined"

    def __call__(self, T_out: float) -> float:
        return self.value


# ==========================================================================
# Controller parameters and logic
# ==========================================================================

COMMON_FIELDS = ("control_sensor", "control_dead_band", "control_heating_dead_band",
                 "control_neutral_zone", "control_min_on_time", "control_min_off_time",
                 "control_sample_period")
TYPE_FIELDS = {"on_off": (), "staged": ("control_stage_fractions",),
               "inverter": ("control_kp", "control_ki", "control_min_fraction",
                            "control_max_fraction")}


@dataclass(frozen=True)
class ControlParams:
    """Controller settings (SI): dead bands K, times s, gains 1/K and 1/(K.s)."""

    type: Literal["on_off", "staged", "inverter"]
    sensor: Literal["return_air", "cabin_mean"]
    dead_band_K: float
    heating_dead_band_K: float
    neutral_zone_K: float
    min_on_s: float
    min_off_s: float
    sample_s: float
    stage_fractions: tuple[float, ...] = ()
    kp: float = 0.0
    ki: float = 0.0
    u_min: float = 0.0
    u_max: float = 1.0

    def __post_init__(self) -> None:
        if self.type not in TYPE_FIELDS:
            raise ValueError(f"unknown control type {self.type!r}")
        if not (self.dead_band_K > 0 and self.heating_dead_band_K > 0):
            raise ValueError("dead bands must be > 0")
        if not self.neutral_zone_K > (self.dead_band_K + self.heating_dead_band_K) / 2:
            raise ValueError(
                f"neutral zone {self.neutral_zone_K} K must exceed (dead band + heating dead "
                f"band) / 2 = {(self.dead_band_K + self.heating_dead_band_K) / 2} K, otherwise "
                f"heating and cooling would fight")
        if not self.sample_s > 0 or self.min_on_s < 0 or self.min_off_s < 0:
            raise ValueError("sample period must be > 0 and minimum times >= 0")
        if self.type == "staged":
            f = self.stage_fractions
            if not f or any(not 0 < x <= 1 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
                raise ValueError("stage fractions must increase within (0, 1]")
        if self.type == "inverter" and not 0 < self.u_min < self.u_max <= 1:
            raise ValueError("inverter needs 0 < min fraction < max fraction <= 1")

    @classmethod
    def from_card(cls, card: UnitCard) -> ControlParams:
        """Read control_* fields; refuse (listing them) while required ones are empty."""
        ctype = card.si("control_type")
        if ctype is None:
            raise MissingEvidenceError(["control_type"],
                                       f"{card.name}: control_type is missing or empty")
        need = COMMON_FIELDS + TYPE_FIELDS[ctype]
        missing = [n for n in need if card.si(n) is None]
        if missing:
            raise MissingEvidenceError(missing, f"{card.name}: {ctype} controller needs "
                                                f"{', '.join(missing)}")
        v = {n: card.si(n) for n in need}
        stages = ()
        if ctype == "staged":
            try:
                stages = tuple(float(x) for x in str(v["control_stage_fractions"]).split(","))
            except ValueError:
                raise ValueError("control_stage_fractions must be like '0.5, 1.0'") from None
        return cls(
            type=ctype, sensor=v["control_sensor"], dead_band_K=v["control_dead_band"],
            heating_dead_band_K=v["control_heating_dead_band"],
            neutral_zone_K=v["control_neutral_zone"], min_on_s=v["control_min_on_time"],
            min_off_s=v["control_min_off_time"], sample_s=v["control_sample_period"],
            stage_fractions=stages, kp=v.get("control_kp") or 0.0,
            ki=v.get("control_ki") or 0.0, u_min=v.get("control_min_fraction") or 0.0,
            u_max=v.get("control_max_fraction") or 1.0)


@dataclass
class Controller:
    """Discrete-time controller; call ``step`` at every sample time."""

    params: ControlParams
    mode: ControlMode = "off"
    fraction: float = 0.0
    stage: int = 0
    integral: float = 0.0
    _t_last_change: float = -math.inf     # last start or stop
    _t_last: float | None = None

    def _may_change_onoff(self, t: float, turning_on: bool) -> bool:
        if self._t_last_change == -math.inf:
            return True                       # no history at the start of the run
        held = t - self._t_last_change
        return held >= (self.params.min_off_s if turning_on else self.params.min_on_s) - 1e-9

    def step(self, t: float, T_meas: float, setpoint: float) -> tuple[ControlMode, float]:
        """New (mode, capacity fraction) from the measured temperature and setpoint."""
        P = self.params
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        sp_c, sp_h = setpoint, setpoint - P.neutral_zone_K
        db, dbh = P.dead_band_K, P.heating_dead_band_K
        mode, frac = self.mode, self.fraction

        if mode == "off":
            if T_meas >= sp_c + db / 2 and self._may_change_onoff(t, True):
                mode = "cooling"
                if P.type == "on_off":
                    frac = 1.0
                elif P.type == "staged":
                    self.stage = 1
                    while self.stage < len(P.stage_fractions) and \
                            T_meas >= sp_c + db / 2 + self.stage * db:
                        self.stage += 1
                    frac = P.stage_fractions[self.stage - 1]
                else:
                    self.integral = P.u_max - P.kp * (T_meas - sp_c)
                    frac = P.u_max
            elif T_meas <= sp_h - dbh / 2 and self._may_change_onoff(t, True):
                mode, frac = "heating", 1.0
        elif mode == "heating":
            if T_meas >= sp_h + dbh / 2 and self._may_change_onoff(t, False):
                mode, frac = "off", 0.0
        else:  # cooling
            if P.type == "on_off":
                if T_meas <= sp_c - db / 2 and self._may_change_onoff(t, False):
                    mode, frac = "off", 0.0
            elif P.type == "staged":
                s = self.stage
                while s < len(P.stage_fractions) and T_meas >= sp_c + db / 2 + s * db:
                    s += 1
                while s > 0 and T_meas <= sp_c - db / 2 + (s - 1) * db:
                    s -= 1
                if s == 0 and not self._may_change_onoff(t, False):
                    s = 1                         # minimum on time: stay on lowest stage
                self.stage = s
                if s == 0:
                    mode, frac = "off", 0.0
                else:
                    frac = P.stage_fractions[s - 1]
            else:  # inverter PI
                e = T_meas - sp_c
                u_raw = P.kp * e + self.integral + P.ki * e * dt
                if not ((u_raw > P.u_max and e > 0) or (u_raw < P.u_min and e < 0)):
                    self.integral += P.ki * e * dt
                u = min(max(P.kp * e + self.integral, P.u_min), P.u_max)
                if u <= P.u_min and T_meas <= sp_c - db / 2 and self._may_change_onoff(t, False):
                    mode, frac = "off", 0.0
                else:
                    frac = u

        if (mode == "off") != (self.mode == "off"):
            self._t_last_change = t
        if mode == "off":
            self.stage = 0
        self.mode, self.fraction = mode, frac
        return mode, frac


# ==========================================================================
# Plants and control sensor
# ==========================================================================

@dataclass(frozen=True)
class ConstantCapacityPlant:
    """VERIFICATION mode: sensible capacity, no coil curve, no humidity effect.

    Cooling removes fraction * Q_cool_W, heating adds fraction * Q_heat_W.
    """

    Q_cool_W: float
    Q_heat_W: float = 0.0


@dataclass(frozen=True)
class UnitPlant:
    """The DX unit (core.unit); fan runs continuously ("ventilation" when off)."""

    unit: DXUnit
    RH_out: Input


@dataclass
class ControlSensor:
    """The virtual temperature sensor the controller reads."""

    type: SensorType
    errors: np.ndarray          # one calibration error per probe
    location: str = "return_air"

    @classmethod
    def from_sensors(cls, sensors: SensorSet, location: str = "return_air",
                     evidence: Registry | None = None) -> ControlSensor:
        n = 1
        if location == "cabin_mean":
            number = evidence.number if evidence is not None else _evidence.number
            n = int(round(number(CABIN_PROBES_ID, "-")))
        elif location != "return_air":
            raise ValueError(f"control sensor location {location!r}")
        return cls(sensors.types["temperature"],
                   sensors.errors(f"control_{location}", "temperature", n), location)

    @classmethod
    def ideal(cls, tau_s: float = 0.0, location: str = "return_air") -> ControlSensor:
        return cls(SensorType("temperature", tau_s=tau_s), np.zeros(1), location)

    def reading(self, lagged_T: float) -> float:
        """Mean of the probe readings (calibration error, then resolution)."""
        y = self.type.quantize(lagged_T + self.errors)
        return float(np.mean(y))


# ==========================================================================
# Sampled, restarting simulation
# ==========================================================================

@dataclass
class ControlledRun:
    """Result of ``simulate_controlled``."""

    segments: list[pd.DataFrame]          # one per interval of constant controller output
    samples: pd.DataFrame                 # every controller sample: t, reading, setpoint, mode,...
    events: list[dict]                    # every change of controller output
    params: ControlParams
    M_da_kg: float = math.nan

    @property
    def data(self) -> pd.DataFrame:
        return pd.concat(self.segments, ignore_index=True)

    @property
    def starts(self) -> int:
        """Number of compressor starts (off -> cooling), including a start at t = 0."""
        initial = int(self.samples["mode"].iloc[0] == "cooling")
        return initial + sum(1 for e in self.events
                             if e["from_mode"] == "off" and e["to_mode"] == "cooling")

    def onoff_periods(self) -> list[tuple[str, float, float]]:
        """(on|off, start, end) periods between start/stop events."""
        changes = [e for e in self.events if (e["from_mode"] == "off") != (e["to_mode"] == "off")]
        out, t0 = [], 0.0
        state = "off" if self.samples["mode"].iloc[0] == "off" else "on"
        for e in changes:
            if e["t"] > t0:
                out.append((state, t0, e["t"]))
            state, t0 = ("off" if e["to_mode"] == "off" else "on"), e["t"]
        return out


def simulate_controlled(
    coach: SingleZoneCoach,
    T0: float,
    T_out: Input,
    controller: Controller,
    setpoint: SetpointCurve | FixedSetpoint,
    plant: ConstantCapacityPlant | UnitPlant,
    sensor: ControlSensor,
    t_end_s: float,
    dt_out_s: float = 60.0,
    *,
    W0: float | None = None,
    Q_int: Input = 0.0,
    loads: Sequence[LoadTerm] = (),
    p: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> ControlledRun:
    """Simulate the coach under sampled control (see module docstring).

    The controller is evaluated at t = k * sample_s using the sensor reading
    held at its last sampling instant (sensor sampling period from the sensor
    type; 0 = read at the controller instant). The lagged sensor temperature
    is an extra ODE state when the sensor time constant is > 0.
    """
    P = controller.params
    f_out = _as_function(T_out)
    humid = W0 is not None
    if isinstance(plant, UnitPlant) and not humid:
        raise ValueError("a UnitPlant needs the humidity state: pass W0")
    M_da = dry_air_mass_kg(coach, T0, W0, p) if humid else None
    tau = sensor.type.tau_s
    lag_i = 2 if humid else 1

    def zone_for(mode: str, frac: float) -> ZoneSystem:
        if isinstance(plant, ConstantCapacityPlant):
            q = (frac * plant.Q_cool_W if mode == "cooling"
                 else -frac * plant.Q_heat_W if mode == "heating" else 0.0)
            return ZoneSystem(coach, f_out, Q_int, q, loads=loads, p=p, M_da=M_da)
        op = UnitOperation(plant.unit, mode if mode != "off" else "ventilation",
                           frac if mode != "off" else 0.0, plant.RH_out)
        return ZoneSystem(coach, f_out, Q_int, 0.0, loads=loads, unit=op, p=p, M_da=M_da)

    y = [float(T0)] + ([float(W0)] if humid else []) + ([float(T0)] if tau > 0 else [])
    t = 0.0
    sensor_value = sensor.reading(T0)
    mode, frac = controller.step(0.0, sensor_value, setpoint(f_out(0.0)))
    samples = [dict(t=0.0, reading=sensor_value, setpoint=setpoint(f_out(0.0)), mode=mode,
                    fraction=frac)]
    events: list[dict] = []
    segments: list[pd.DataFrame] = []
    pieces: list[tuple[np.ndarray, np.ndarray]] = []     # (times, states) of the open segment

    def close_segment(zone: ZoneSystem, mode: str, frac: float) -> None:
        """Build the segment table once (pieces share end points: keep them once)."""
        times = np.concatenate([pieces[0][0]] + [tt[1:] for tt, _ in pieces[1:]])
        Y = np.concatenate([pieces[0][1]] + [yy[:, 1:] for _, yy in pieces[1:]], axis=1)
        const = isinstance(plant, ConstantCapacityPlant)
        n = len(times)
        extra = {
            "T_sensor_lag_C": Y[lag_i] if tau > 0 else Y[0],
            "mode": [mode] * n,
            "fraction": np.full(n, frac),
            "Q_cooling_W": np.full(n, frac * plant.Q_cool_W if const and mode == "cooling"
                                   else 0.0),
            "Q_heating_W": np.full(n, frac * plant.Q_heat_W if const and mode == "heating"
                                   else 0.0),
        }
        segments.append(zone.frame(times, Y[0], Y[1] if humid else None, extra))
        pieces.clear()
    k_ctrl = 1
    k_sens = 1
    Ts = sensor.type.sample_s
    chunk = max(50 * P.sample_s, 600.0)

    while t < t_end_s - 1e-9:
        zone = zone_for(mode, frac)

        def rhs(tt, yy, zone=zone):
            dT, dW = zone.derivatives(tt, yy[0], yy[1] if humid else math.nan)
            out = [dT] + ([dW] if humid else [])
            if tau > 0:
                out.append((yy[0] - yy[lag_i]) / tau)
            return out

        t_chunk = min(t_end_s, t + chunk)
        sol = solve_ivp(rhs, (t, t_chunk), y, method="LSODA", dense_output=True,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solver failed at t={t:.1f} s: {sol.message}")

        def lagged(te):
            yy = sol.sol(te)
            return yy[lag_i] if tau > 0 else yy[0]

        t_switch = None
        new = (mode, frac)
        while True:
            t_c = k_ctrl * P.sample_s
            t_s = k_sens * Ts if Ts > 0 else math.inf
            t_ev = min(t_c, t_s)
            if t_ev > t_chunk + 1e-9:
                break
            if t_s <= t_c:                         # sensor sampling instant
                sensor_value = sensor.reading(lagged(t_s))
                k_sens += 1
                if t_s < t_c:
                    continue
            if Ts == 0:
                sensor_value = sensor.reading(lagged(t_c))
            sp = setpoint(f_out(t_c))
            new = controller.step(t_c, sensor_value, sp)
            samples.append(dict(t=t_c, reading=sensor_value, setpoint=sp, mode=new[0],
                                fraction=new[1]))
            k_ctrl += 1
            if new[0] != mode or abs(new[1] - frac) > 1e-9:
                t_switch = t_c
                break

        t_end_piece = t_switch if t_switch is not None else t_chunk
        grid = dt_out_s * np.arange(math.floor(t / dt_out_s) + 1,
                                    math.ceil(t_end_piece / dt_out_s))
        times = np.unique(np.concatenate([[t], grid[(grid > t) & (grid < t_end_piece)],
                                          [t_end_piece]]))
        if len(times) < 3:
            times = np.array([t, 0.5 * (t + t_end_piece), t_end_piece])
        pieces.append((times, sol.sol(times)))
        y = list(sol.sol(t_end_piece))
        t = t_end_piece
        if t_switch is not None:
            close_segment(zone, mode, frac)
            events.append(dict(t=t, from_mode=mode, from_fraction=frac, to_mode=new[0],
                               to_fraction=new[1]))
            mode, frac = new
    if pieces:
        close_segment(zone, mode, frac)
    return ControlledRun(segments, pd.DataFrame(samples), events, P,
                         M_da if M_da is not None else math.nan)
