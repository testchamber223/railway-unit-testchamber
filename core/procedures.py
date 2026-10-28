"""Test procedures: virtual versions of the type tests run in a climatic chamber.

Each procedure drives a model the way the real test drives the vehicle and
evaluates the result the way the test standard does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.coach import SingleZoneCoach, simulate

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class KValueResult:
    """Outcome of ``k_value_test``.

    Attributes:
        k_measured_W_m2K: k = Q_heater / (A (T_in - T_out)) at the end of the run [W/(m2.K)].
        t_steady_s: time from start until |dT/dt| stays below the criterion [s].
        T_in_C: interior temperature at the end of the run [degC].
        T_out_C: chamber temperature [degC].
        dTdt_end_K_h: interior temperature drift at the end of the run [K/h].
        data: the full simulated time series (see ``core.coach.simulate``).
    """

    k_measured_W_m2K: float
    t_steady_s: float
    T_in_C: float
    T_out_C: float
    dTdt_end_K_h: float
    data: pd.DataFrame


def k_value_test(
    coach: SingleZoneCoach,
    T_out: float,
    Q_heater_W: float,
    t_end_s: float,
    dt_out_s: float = 60.0,
    steady_dTdt_K_h: float = 0.1,
) -> KValueResult:
    """Virtual k-value test, heater method (UIC 553-1 k-value test, section to confirm).

    The closed coach (HVAC off, Q_hvac = 0) starts at the chamber temperature
    T_out [degC]; heaters inside deliver a constant Q_heater_W [W]. When the
    interior temperature is steady, the k-value follows from the energy balance

        k = Q_heater / (A * (T_in - T_out))     [W/(m2.K)]

    with A = coach.envelope_area_m2 (reference-area definition: evidence E-008).

    Steady state is |dT/dt| < steady_dTdt_K_h [K/h], with dT/dt taken from the
    model equation at each output time; ``t_steady_s`` is the first time after
    which the criterion holds for the rest of the run. k is evaluated from the
    state at t_end_s; any remaining drift biases it by about
    C * dT/dt / Q_heater, so run well past t_steady_s.

    Raises:
        ValueError: if Q_heater_W <= 0 or steady state is not reached by t_end_s.
    """
    if not Q_heater_W > 0:
        raise ValueError(f"Q_heater_W={Q_heater_W} must be > 0")

    df = simulate(coach, T0=T_out, T_out=T_out, Q_int=Q_heater_W, Q_hvac=0.0,
                  t_end_s=t_end_s, dt_out_s=dt_out_s)

    dTdt_K_h = (df["Q_env_W"] + df["Q_int_W"] - df["Q_hvac_W"]).to_numpy() \
        / coach.heat_capacity_J_K * SECONDS_PER_HOUR
    steady = np.abs(dTdt_K_h) < steady_dTdt_K_h
    if not steady[-1]:
        raise ValueError(
            f"{coach.name}: not steady at t_end_s={t_end_s} s "
            f"(|dT/dt| = {abs(dTdt_K_h[-1]):.3g} K/h >= {steady_dTdt_K_h} K/h)"
        )
    unsteady = np.flatnonzero(~steady)
    i_steady = unsteady[-1] + 1 if unsteady.size else 0

    T_in = float(df["T_C"].iloc[-1])
    return KValueResult(
        k_measured_W_m2K=Q_heater_W / (coach.envelope_area_m2 * (T_in - T_out)),
        t_steady_s=float(df["t_s"].iloc[i_steady]),
        T_in_C=T_in,
        T_out_C=float(T_out),
        dTdt_end_K_h=float(dTdt_K_h[-1]),
        data=df,
    )
