"""Reduced-order model: calibrate the multi-zone coach against CFD runs.

Workflow
    1. ``lhs_design`` -- Latin hypercube (scipy.stats.qmc) over named factors
       with ranges; ``write_run_list`` writes the run list CSV for the Fluent
       team (factors + fixed settings, one row per CFD run).
    2. The Fluent team returns steady zone temperatures per run.
    3. ``calibrate`` -- least squares (scipy.optimize.least_squares) of the
       multi-zone parameters on the steady zone temperatures: mixing
       conductance G_mix, supply fractions (stick-breaking, so they stay in
       [0, 1] and sum to 1) and the effective solar aperture A_sol (the
       per-zone coefficient: total gain = I * A_sol, split by solar share).
       RMS and max error are reported per output (each zone and the spread).
    4. ``leave_one_out`` -- refit without each run and predict it.
    5. ``ValidityRange`` -- the bounding box of the DOE inputs; later
       simulations outside it get warnings (run_procedure lists them in
       TestResult.warnings).

Model per run (``ROMSetup.predict``): MultiZoneCoach.steady_state with a
fixed supply (airflow factor at the setting supply temperature / humidity),
outside temperature factor, solar factor * A_sol and occupancy factor * the
setting sensible heat per person, split by the zone shares.

Nothing here contains CFD results: tests and the notebook use MANUFACTURED
data generated from this model with known parameters (verification-only).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from core import psychro
from core.coach import FixedSupply, MultiZoneCoach

FACTORS = ("supply_airflow_m3_h", "outside_T_C", "solar_W_m2", "occupancy")
SECONDS_PER_HOUR = 3600.0


# ==========================================================================
# Design of experiments
# ==========================================================================

def lhs_design(factors: dict[str, tuple[float, float]], n_runs: int, seed: int = 0
               ) -> pd.DataFrame:
    """Latin-hypercube design: one row per run, columns run_id + factors."""
    names = list(factors)
    lo = np.array([factors[k][0] for k in names], float)
    hi = np.array([factors[k][1] for k in names], float)
    if np.any(hi <= lo):
        raise ValueError("each factor range needs low < high")
    sample = qmc.LatinHypercube(d=len(names), seed=seed).random(n_runs)
    df = pd.DataFrame(qmc.scale(sample, lo, hi), columns=names)
    df.insert(0, "run_id", [f"R{k + 1:02d}" for k in range(n_runs)])
    return df


def write_run_list(design: pd.DataFrame, path: Path | str,
                   settings: dict[str, float] | None = None) -> Path:
    """Write the run list CSV for the Fluent team (design + fixed settings)."""
    df = design.copy()
    for k, v in (settings or {}).items():
        df[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    return path


@dataclass(frozen=True)
class ValidityRange:
    """Bounding box of the calibration inputs."""

    bounds: dict[str, tuple[float, float]]

    @classmethod
    def from_design(cls, design: pd.DataFrame, factors: Sequence[str] = FACTORS
                    ) -> ValidityRange:
        return cls({f: (float(design[f].min()), float(design[f].max()))
                    for f in factors if f in design})

    def check(self, values: dict[str, float]) -> list[str]:
        """Warnings for every value outside the calibrated range."""
        out = []
        for name, v in values.items():
            if name in self.bounds and v is not None:
                lo, hi = self.bounds[name]
                if not lo <= v <= hi:
                    out.append(f"ROM validity: {name} = {v:g} is outside the calibrated range "
                               f"{lo:g} .. {hi:g} (extrapolation)")
        return out


# ==========================================================================
# Model and calibration
# ==========================================================================

@dataclass(frozen=True)
class ROMSetup:
    """Fixed parts of the reduced-order model (everything except the fitted θ)."""

    coach: MultiZoneCoach                      # geometry, shares, return weights
    supply_T_C: float                          # DOE setting
    supply_W: float                            # DOE setting
    pax_sensible_W: float                      # per person (setting)

    @property
    def n(self) -> int:
        return self.coach.n

    @property
    def theta_names(self) -> list[str]:
        return (["G_mix_W_K"] + [f"stick_{i + 1}" for i in range(self.n - 1)]
                + ["A_sol_m2"])

    def fractions(self, sticks: Sequence[float]) -> tuple[float, ...]:
        """Stick-breaking: f_1 = s_1, f_2 = (1 - s_1) s_2, ..., f_n = remainder."""
        rest, out = 1.0, []
        for s in sticks:
            out.append(rest * s)
            rest *= 1 - s
        return tuple(out + [rest])

    def sticks(self, fractions: Sequence[float]) -> list[float]:
        rest, out = 1.0, []
        for f in fractions[:-1]:
            out.append(f / rest if rest > 0 else 0.0)
            rest -= f
        return out

    def theta(self, G_mix: float, fractions: Sequence[float], A_sol: float) -> np.ndarray:
        return np.array([G_mix, *self.sticks(fractions), A_sol])

    def unpack(self, theta: Sequence[float]) -> dict:
        return dict(G_mix_W_K=float(theta[0]),
                    supply_fraction=self.fractions(theta[1:self.n]),
                    A_sol_m2=float(theta[-1]))

    def predict(self, theta: Sequence[float], run: dict | pd.Series) -> np.ndarray:
        """Steady zone temperatures [degC] for one run."""
        p = self.unpack(theta)
        mz = dataclasses.replace(self.coach, G_mix_W_K=p["G_mix_W_K"],
                                 supply_fraction=p["supply_fraction"])
        m_s = run["supply_airflow_m3_h"] / SECONDS_PER_HOUR / psychro.specific_volume(
            self.supply_T_C, self.supply_W)
        gains = (np.array(mz.solar_share) * run["solar_W_m2"] * p["A_sol_m2"]
                 + np.array(mz.passenger_share) * run["occupancy"] * self.pax_sensible_W)
        return mz.steady_state(run["outside_T_C"],
                               FixedSupply(m_s, self.supply_T_C, self.supply_W), gains)

    def predict_all(self, theta, runs: pd.DataFrame) -> np.ndarray:
        return np.vstack([self.predict(theta, r) for _, r in runs.iterrows()])

    def default_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([0.0] + [0.0] * (self.n - 1) + [0.0])
        hi = np.array([np.inf] + [1.0] * (self.n - 1) + [np.inf])
        return lo, hi


def _outputs(T: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(T, columns=[f"T_{z}" for z in names])
    df["spread"] = T.max(axis=1) - T.min(axis=1)
    return df


@dataclass
class Calibration:
    theta: np.ndarray
    parameters: dict
    rms: dict[str, float]
    max_abs: dict[str, float]
    predicted: pd.DataFrame
    observed: pd.DataFrame
    validity: ValidityRange
    success: bool
    message: str = ""
    loo: pd.DataFrame | None = field(default=None, repr=False)


def calibrate(setup: ROMSetup, runs: pd.DataFrame, observed_T: np.ndarray,
              theta0: Sequence[float], bounds=None) -> Calibration:
    """Least-squares fit of θ to observed steady zone temperatures (runs x zones)."""
    obs = np.asarray(observed_T, float)
    if obs.shape != (len(runs), setup.n):
        raise ValueError(f"observed_T needs shape ({len(runs)}, {setup.n})")
    lo, hi = bounds if bounds is not None else setup.default_bounds()
    x0 = np.clip(np.asarray(theta0, float), lo, hi)
    x_scale = np.where(np.abs(x0) > 0, np.abs(x0), 1.0)

    def residual(theta):
        return (setup.predict_all(theta, runs) - obs).ravel()

    res = least_squares(residual, x0, bounds=(lo, hi), x_scale=x_scale, xtol=1e-12,
                        ftol=1e-12, gtol=1e-12)
    pred = _outputs(setup.predict_all(res.x, runs), setup.coach.zone_names)
    obs_df = _outputs(obs, setup.coach.zone_names)
    err = pred - obs_df
    return Calibration(
        theta=res.x, parameters=setup.unpack(res.x),
        rms={c: float(np.sqrt(np.mean(err[c] ** 2))) for c in err},
        max_abs={c: float(np.max(np.abs(err[c]))) for c in err},
        predicted=pred, observed=obs_df, validity=ValidityRange.from_design(runs),
        success=bool(res.success), message=str(res.message))


def leave_one_out(setup: ROMSetup, runs: pd.DataFrame, observed_T: np.ndarray,
                  theta0: Sequence[float], bounds=None) -> pd.DataFrame:
    """Refit without each run and predict it: one row of errors per held-out run."""
    obs = np.asarray(observed_T, float)
    rows = []
    for k in range(len(runs)):
        keep = np.arange(len(runs)) != k
        cal = calibrate(setup, runs.iloc[keep].reset_index(drop=True), obs[keep], theta0,
                        bounds)
        pred = _outputs(setup.predict(cal.theta, runs.iloc[k])[None, :],
                        setup.coach.zone_names).iloc[0]
        true = _outputs(obs[k][None, :], setup.coach.zone_names).iloc[0]
        row = {"run_id": runs.iloc[k].get("run_id", k)}
        row.update({f"err_{c}": float(pred[c] - true[c]) for c in pred.index})
        row["max_abs_err"] = float(np.max(np.abs(pred - true)))
        rows.append(row)
    return pd.DataFrame(rows)
