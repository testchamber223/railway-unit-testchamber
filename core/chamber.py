"""Virtual test facilities: unit test room and full-vehicle climatic chamber.

UnitTestRoom -- psychrometric test room with an indoor and an outdoor room,
each held at a set dry-bulb and humidity. The unit's capacity is measured by
the air-enthalpy method (ASHRAE 37 / EN 14511-3, clause to confirm):

    Q_net   = m_dot * (h_in - h_out)        between unit air inlet and outlet
    Q_gross = Q_net + P_fan                 (fan heat enters between the planes)

with temperatures, humidities and airflow read by virtual sensors
(core.sensors; accuracies E-058..E-060). Control tolerances (E-052..E-055)
and room capacity limits (E-056, E-057) are evidence IDs, assumption until
sourced. Control is ideal (rooms exactly at setpoint while their capacity
suffices); the tolerance check is applied to the MEASURED room conditions.

VehicleChamber -- full-coach climatic chamber. Inside dimensions from E-061
(CETEMET, 28 x 5 x 5.5 m), air temperature and humidity controlled to a
setpoint, measured chamber temperature = mean of E-067 probes (UIC 553-1
App. H), solar lamps on roof and sides (intensities E-013..E-020, vertical
factor E-021). Control tolerances (E-062, E-063), heating / cooling capacity
(E-064, E-065) and ramp-rate limit (E-066) are evidence IDs.

The chamber air is ONE well-mixed volume for now: no spatial non-uniformity
of temperature, humidity or lamp irradiance. That comes later from CFD
(ansys/). The plant-capacity check is quasi-steady (see ``plant_power``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from core import psychro
from core.sensors import Channel, Measured, SensorSet, propagate
from core.unit import DXUnit, Mode, UnitResult
from evidence import registry as _evidence
from evidence.registry import EvidenceError, Registry

SECONDS_PER_HOUR = 3600.0

ROOM_EVIDENCE: dict[str, tuple[str, str]] = {
    "tol_T_indoor_K": ("E-052", "K"),
    "tol_RH_indoor": ("E-053", "-"),
    "tol_T_outdoor_K": ("E-054", "K"),
    "tol_RH_outdoor": ("E-055", "-"),
    "Q_max_indoor_W": ("E-056", "W"),
    "Q_max_outdoor_W": ("E-057", "W"),
}

CHAMBER_EVIDENCE: dict[str, tuple[str, str]] = {
    "tol_T_K": ("E-062", "K"),
    "tol_RH": ("E-063", "-"),
    "Q_heat_max_W": ("E-064", "W"),
    "Q_cool_max_W": ("E-065", "W"),
    "ramp_max_K_h": ("E-066", "K/h"),
    "n_probes": ("E-067", "-"),
}
CHAMBER_DIMS_ID = "E-061"


def _numbers(spec: dict[str, tuple[str, str]], evidence: Registry | None) -> dict[str, float]:
    number = evidence.number if evidence is not None else _evidence.number
    return {name: number(eid, unit) for name, (eid, unit) in spec.items()}


# ==========================================================================
# Unit test room
# ==========================================================================

@dataclass(frozen=True)
class RoomMeasurement:
    """Result of one air-enthalpy measurement in the unit test room."""

    measured: dict[str, Measured]
    true: dict[str, float]
    unit_result: UnitResult
    validity: list[str]
    notes: list[str]


@dataclass(frozen=True)
class UnitTestRoom:
    """Indoor + outdoor psychrometric rooms (ASHRAE 37 / EN 14511-3, clause to confirm)."""

    tol_T_indoor_K: float
    tol_RH_indoor: float
    tol_T_outdoor_K: float
    tol_RH_outdoor: float
    Q_max_indoor_W: float
    Q_max_outdoor_W: float
    evidence_ids: tuple[str, ...] = tuple(eid for eid, _ in ROOM_EVIDENCE.values())

    @classmethod
    def from_evidence(cls, evidence: Registry | None = None) -> UnitTestRoom:
        return cls(**_numbers(ROOM_EVIDENCE, evidence))

    def measure(self, unit: DXUnit, indoor: tuple[float, float], outdoor: tuple[float, float],
                mode: Mode, part_load: float, fresh_fraction: float,
                sensors: SensorSet,
                humidity: Literal["dew_point", "rh"] = "dew_point") -> RoomMeasurement:
        """Run the unit between the rooms and measure it by the air-enthalpy method.

        indoor, outdoor: (T [degC], W [kg/kg]) room setpoints; fresh_fraction:
        dry-air mass fraction of outdoor-room air in the unit inlet.
        Measurement planes: unit air inlet (after the mixing box) and supply
        outlet. Fan power is taken from the unit card (no meter uncertainty).
        humidity: instrument for the humidity channels -- "dew_point"
        (chilled-mirror hygrometer, default) or "rh" (capacitive RH sensor).
        The unit room is steady state, so sensor lag and sampling do not
        matter here; calibration error and resolution do.
        """
        (T_i, W_i), (T_o, W_o) = indoor, outdoor
        res = unit.operate(T_i, W_i, T_o, W_o, T_o, mode, part_load, fresh_fraction=fresh_fraction)
        inlet, outlet, m = res.mixed, res.supply, res.supply.m_da
        P_fan = res.Q_fan_W
        if humidity not in ("dew_point", "rh"):
            raise ValueError(f"humidity={humidity!r} must be 'dew_point' or 'rh'")
        H = "DP" if humidity == "dew_point" else "RH"

        def true_humidity(T, W):
            RH = psychro.rh_from_humidity_ratio(T, W)
            return psychro.dew_point(T, RH) if humidity == "dew_point" else RH

        def humidity_ratio_from_reading(T, reading):
            if humidity == "dew_point":            # W at saturation at the dew point
                return psychro.sat_humidity_ratio(min(reading, T))
            return psychro.humidity_ratio(T, min(max(reading, 0.0), 1.0))

        h_in, h_out = true_humidity(inlet.T_C, inlet.W), true_humidity(outlet.T_C, outlet.W)

        def state(bias, T_ch, H_ch, T_true, H_true):
            T = sensors.read(T_ch, "temperature", T_true)[0, 0] + bias.get(T_ch, 0.0)
            reading = sensors.read(H_ch, humidity if humidity == "dew_point" else "humidity",
                                   H_true)[0, 0] + bias.get(H_ch, 0.0)
            return T, humidity_ratio_from_reading(T, reading)

        def mass_flow(bias):
            return sensors.read("airflow", "airflow", m)[0, 0] * (1 + bias.get("airflow", 0.0))

        def q_net(bias):
            T1, W1 = state(bias, "T_unit_in", f"{H}_unit_in", inlet.T_C, h_in)
            T2, W2 = state(bias, "T_unit_out", f"{H}_unit_out", outlet.T_C, h_out)
            return mass_flow(bias) * (psychro.enthalpy(T1, W1) - psychro.enthalpy(T2, W2))

        def q_sens_gross(bias):
            T1, _ = state(bias, "T_unit_in", f"{H}_unit_in", inlet.T_C, h_in)
            T2, W2 = state(bias, "T_unit_out", f"{H}_unit_out", outlet.T_C, h_out)
            return mass_flow(bias) * (psychro.enthalpy(T1, W2) - psychro.enthalpy(T2, W2)) + P_fan

        h_kind = "dew_point" if humidity == "dew_point" else "humidity"
        channels = [Channel("T_unit_in", "temperature"), Channel(f"{H}_unit_in", h_kind),
                    Channel("T_unit_out", "temperature"), Channel(f"{H}_unit_out", h_kind),
                    Channel("airflow", "airflow")]
        net = propagate(q_net, channels, sensors)
        measured = {
            "capacity_net_W": net,
            "capacity_gross_W": Measured(net.value + P_fan, net.uncertainty, net.uncertainty_rss),
            "sensible_capacity_gross_W": propagate(q_sens_gross, channels, sensors),
        }
        coil = res.coil
        true = {
            "capacity_net_W": m * (psychro.enthalpy(inlet.T_C, inlet.W)
                                   - psychro.enthalpy(outlet.T_C, outlet.W)),
            "capacity_gross_W": coil.Q_total_W if coil else 0.0,
            "sensible_capacity_gross_W": coil.Q_sensible_W if coil else 0.0,
        }

        # Facility validity: measured room conditions vs tolerance, room capacity
        validity: list[str] = []
        for room, (T_set, W_set), tol_T, tol_RH, (eT, eRH) in (
            ("indoor", indoor, self.tol_T_indoor_K, self.tol_RH_indoor, ("E-052", "E-053")),
            ("outdoor", outdoor, self.tol_T_outdoor_K, self.tol_RH_outdoor, ("E-054", "E-055")),
        ):
            RH_set = psychro.rh_from_humidity_ratio(T_set, W_set)
            T_meas = sensors.read(f"T_{room}_room", "temperature", T_set)[0, 0]
            H_meas = sensors.read(f"{H}_{room}_room", h_kind, true_humidity(T_set, W_set))[0, 0]
            RH_meas = psychro.rh_from_humidity_ratio(
                T_meas, humidity_ratio_from_reading(T_meas, H_meas))
            if abs(T_meas - T_set) > tol_T:
                validity.append(f"{eT}: {room} room measured {T_meas:.2f} degC, set "
                                f"{T_set:.2f} +/- {tol_T:g} K")
            if abs(RH_meas - RH_set) > tol_RH:
                validity.append(f"{eRH}: {room} room measured RH {RH_meas:.3f}, set "
                                f"{RH_set:.3f} +/- {tol_RH:g}")
        if abs(true["capacity_net_W"]) > self.Q_max_indoor_W:
            validity.append(f"E-056: indoor room must offset {abs(true['capacity_net_W']):.0f} W "
                            f"> capacity {self.Q_max_indoor_W:.0f} W")
        condenser = true["capacity_gross_W"]
        if condenser > self.Q_max_outdoor_W:
            validity.append(f"E-057: outdoor room must absorb >= {condenser:.0f} W "
                            f"> capacity {self.Q_max_outdoor_W:.0f} W")
        notes = ["outdoor-room load excludes compressor power (EIR curve E-049 not available)",
                 "fan power from the unit card, no power-meter uncertainty",
                 f"humidity measured with {'a dew-point hygrometer' if H == 'DP' else 'RH sensors'}"]
        return RoomMeasurement(measured, true, res, validity, notes)


# ==========================================================================
# Vehicle chamber
# ==========================================================================

def parse_dimensions(text: str) -> tuple[float, float, float]:
    """Parse 'L x W x H' (evidence E-061 format) into three floats [m]."""
    parts = re.split(r"\s*[xX×]\s*", str(text).strip())
    if len(parts) != 3:
        raise EvidenceError(f"cannot parse chamber dimensions {text!r}; expected 'L x W x H'")
    return tuple(float(p) for p in parts)


@dataclass(frozen=True)
class VehicleChamber:
    """Full-vehicle climatic chamber with one well-mixed air volume (CFD later)."""

    length_m: float
    width_m: float
    height_m: float
    tol_T_K: float
    tol_RH: float
    Q_heat_max_W: float
    Q_cool_max_W: float
    ramp_max_K_h: float
    n_probes: int
    evidence_ids: tuple[str, ...] = field(default=(CHAMBER_DIMS_ID,) + tuple(
        eid for eid, _ in CHAMBER_EVIDENCE.values()))

    @classmethod
    def from_evidence(cls, evidence: Registry | None = None) -> VehicleChamber:
        get = evidence.get if evidence is not None else _evidence.get
        L, W, H = parse_dimensions(get(CHAMBER_DIMS_ID).value)
        values = _numbers(CHAMBER_EVIDENCE, evidence)
        values["n_probes"] = int(round(values["n_probes"]))
        return cls(length_m=L, width_m=W, height_m=H, **values)

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m

    def air_temperature(self, T_start: float, T_set: float,
                        rate_K_h: float | None = None) -> Callable[[float], float]:
        """Chamber air temperature T(t) [degC], t [s] from the step start.

        Ramps from T_start to T_set at min(rate_K_h, ramp_max_K_h) K/h, then
        holds T_set (ideal control within the plant capacity).
        """
        rate = self.ramp_max_K_h if rate_K_h is None else min(rate_K_h, self.ramp_max_K_h)
        if not rate > 0:
            raise ValueError("ramp rate must be > 0")
        dT, r = T_set - T_start, rate / SECONDS_PER_HOUR

        def T(t: float) -> float:
            return T_set if abs(dT) <= r * t else T_start + np.sign(dT) * r * t
        return T

    def ramp_duration_s(self, T_start: float, T_set: float, rate_K_h: float | None = None) -> float:
        rate = self.ramp_max_K_h if rate_K_h is None else min(rate_K_h, self.ramp_max_K_h)
        return abs(T_set - T_start) / rate * SECONDS_PER_HOUR

    def probe_readings(self, T_true: np.ndarray, sensors: SensorSet,
                       bias: float = 0.0, t: np.ndarray | None = None) -> np.ndarray:
        """Readings of the n_probes chamber probes, shape (n_probes, n_samples).

        With ``t`` the probe lag and sampling are applied (core.sensors).
        """
        return sensors.read("T_chamber", "temperature", T_true, self.n_probes, t=t) + bias

    def measured_temperature(self, T_true: np.ndarray, sensors: SensorSet,
                             bias: float = 0.0, t: np.ndarray | None = None) -> np.ndarray:
        """Measured chamber temperature = mean of the probes (UIC 553-1 App. H)."""
        return self.probe_readings(T_true, sensors, bias, t).mean(axis=0)

    def plant_power(self, df: pd.DataFrame, T_chamber: np.ndarray, W_chamber: float | None,
                    lamp_power_W: float, coach_volume_m3: float) -> np.ndarray:
        """Quasi-steady heat the chamber plant must REMOVE from the chamber air [W].

        Positive = cooling demand, negative = heating demand. Sum of: heat
        leaving the coach envelope (-Q_env), lamp power not absorbed by the
        coach, unit condenser heat (coil total; compressor power not
        modelled) and fresh/exhaust air exchange, plus the chamber air's own
        heat storage during ramps (air volume = chamber - coach volume).
        """
        P = -df["Q_env_W"].to_numpy() + lamp_power_W
        for col in df.columns:
            if col.startswith("Q_load_solar"):
                P = P - df[col].to_numpy()
        if "unit_Q_coil_total_W" in df:
            P = P + df["unit_Q_coil_total_W"].to_numpy() - df["unit_Q_fresh_enthalpy_W"].to_numpy()
        W = 0.0 if W_chamber is None else W_chamber
        V_air = max(self.volume_m3 - coach_volume_m3, 0.0)
        M_air = np.array([V_air / psychro.specific_volume(T, W) for T in T_chamber])
        cp = np.array([psychro.enthalpy(T + 0.5, W) - psychro.enthalpy(T - 0.5, W)
                       for T in T_chamber])
        dTdt = np.gradient(T_chamber, df["t_s"].to_numpy()) if len(T_chamber) > 1 else 0.0
        return P - M_air * cp * dTdt
