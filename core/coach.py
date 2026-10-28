"""Single-zone coach thermal model.

The coach interior is one well-mixed air zone with an effective heat capacity
(air + interior mass) exchanging heat with the outside through the envelope:

    C * dT/dt = UA * (T_out - T) + Q_int - Q_hvac

    T       interior air temperature              [degC]
    T_out   outside (chamber) air temperature     [degC]
    C       effective heat capacity               [J/K]
    UA      k-value * envelope area               [W/K]
    Q_int   internal gains (passengers, heaters)  [W]
    Q_hvac  heat removed by the HVAC unit         [W]  (negative = heating)
    t       time                                  [s]

Assumptions: perfectly mixed air, lumped capacity, steady envelope
conduction (no wall dynamics), no solar or infiltration terms yet.
Coach parameters are supplied by the caller (from the evidence registry for
real coaches); nothing physical is hard-coded here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

Input = float | Callable[[float], float]
"""A model input: a constant, or a function of time t [s]."""


@dataclass(frozen=True)
class SingleZoneCoach:
    """Lumped single-zone coach.

    Attributes:
        name: identifier.
        envelope_area_m2: heat-transfer area A the k-value refers to [m2]
            (inner/outer/mean definition: see evidence E-008).
        k_value_W_m2K: overall heat-transfer coefficient k [W/(m2.K)].
        heat_capacity_J_K: effective heat capacity C, air + interior mass [J/K].
    """

    name: str
    envelope_area_m2: float
    k_value_W_m2K: float
    heat_capacity_J_K: float

    def __post_init__(self) -> None:
        for field in ("envelope_area_m2", "k_value_W_m2K", "heat_capacity_J_K"):
            v = getattr(self, field)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{self.name}: {field}={v} must be finite and > 0")

    @property
    def UA(self) -> float:
        """Envelope conductance UA = k * A [W/K]."""
        return self.k_value_W_m2K * self.envelope_area_m2


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
    """Steady-state interior temperature [degC] for constant inputs.

    Setting dT/dt = 0:  T_ss = T_out + (Q_int - Q_hvac) / UA,
    with T_out [degC], Q_int and Q_hvac [W], UA [W/K].
    """
    T_out = _require_constant("T_out", T_out)
    Q_int = _require_constant("Q_int", Q_int)
    Q_hvac = _require_constant("Q_hvac", Q_hvac)
    return T_out + (Q_int - Q_hvac) / coach.UA


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
) -> pd.DataFrame:
    """Integrate C dT/dt = UA (T_out - T) + Q_int - Q_hvac from t = 0 to t_end_s.

    Args:
        coach: the coach model.
        T0: initial interior temperature [degC].
        T_out: outside temperature [degC], constant or f(t [s]).
        Q_int: internal gains [W], constant or f(t [s]).
        Q_hvac: heat removed by the HVAC unit [W], constant or f(t [s]).
        t_end_s: end time [s].
        dt_out_s: output interval [s] (the solver chooses its own internal steps).
        rtol, atol: solver tolerances (atol in K).
        max_step: largest internal step [s]; set it below the shortest feature
            of a time-varying input (e.g. a step change) so it is not skipped.

    Returns:
        DataFrame with columns t_s [s], T_C [degC], Q_env_W [W] (= UA (T_out - T),
        positive into the coach), Q_int_W [W], Q_hvac_W [W].

    Solver: scipy.integrate.solve_ivp, method LSODA.
    """
    f_out, f_int, f_hvac = _as_function(T_out), _as_function(Q_int), _as_function(Q_hvac)
    UA, C = coach.UA, coach.heat_capacity_J_K
    t_eval = _output_times(t_end_s, dt_out_s)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        return [(UA * (f_out(t) - y[0]) + f_int(t) - f_hvac(t)) / C]

    sol = solve_ivp(
        rhs, (0.0, t_eval[-1]), [float(T0)], method="LSODA",
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"{coach.name}: solver failed: {sol.message}")

    t, T = sol.t, sol.y[0]
    T_out_t = np.array([f_out(ti) for ti in t])
    return pd.DataFrame({
        "t_s": t,
        "T_C": T,
        "Q_env_W": UA * (T_out_t - T),
        "Q_int_W": [f_int(ti) for ti in t],
        "Q_hvac_W": [f_hvac(ti) for ti in t],
    })
