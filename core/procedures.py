"""Test procedures: virtual versions of the type tests run in a climatic chamber.

Each procedure drives a model the way the real test drives the vehicle and
evaluates the result the way the test standard does. Procedure parameters
(windows, set conditions, tolerances) come from the evidence registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from core.coach import SingleZoneCoach, net_sensible_W, simulate
from evidence import registry

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
