"""Case 0 comparison: Fluent average Nusselt number vs de Vahl Davis (1983).

For every run (Rayleigh number) of ansys/cases/case0_devahldavis:

1. read the Fluent CSV of every mesh and compute the average Nusselt number
   on the hot wall,  Nu = (integral of q dy) / (k * dT)   (Nu = h L / k with
   h = q_mean / dT and q_mean = integral / L), or take a ready Nu_avg column;
2. apply the three-grid GCI (validation/gci.py) to the three finest meshes;
3. compare the extrapolated Nu with the benchmark E-117..E-120: pass if the
   relative difference is within E-121.

Every step that lacks data SKIPS with a message saying what is missing
(results files, benchmark values still to transcribe, tolerance). Run:

    python -m validation.case0_compare
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from core.cfd_import import CfdCase, load_case
from evidence.registry import Registry
from validation.gci import GCIResult, gci

CASE_ID = "case0_devahldavis"
BENCHMARK_IDS = {"Ra1e3": "E-117", "Ra1e4": "E-118", "Ra1e5": "E-119", "Ra1e6": "E-120"}
TOLERANCE_ID = "E-121"


@dataclass
class RunComparison:
    run: str
    status: str                                   # pass | fail | skipped
    messages: list[str] = field(default_factory=list)
    nu_by_mesh: dict[str, float] = field(default_factory=dict)
    gci: GCIResult | None = None
    benchmark: float | None = None
    tolerance: float | None = None

    @property
    def relative_difference(self) -> float | None:
        if self.gci is None or self.benchmark is None:
            return None
        return self.gci.f_extrapolated / self.benchmark - 1


def average_nusselt(df: pd.DataFrame, k: float, dT: float) -> float:
    """Average Nu on the hot wall from a Fluent export (see case.yaml results_format)."""
    if "Nu_avg" in df.columns:
        return float(df["Nu_avg"].iloc[0])
    if not {"position_m", "wall_heat_flux_W_m2"} <= set(df.columns):
        raise ValueError("results need columns position_m and wall_heat_flux_W_m2, or Nu_avg")
    d = df.sort_values("position_m")
    per_length = trapezoid(d["wall_heat_flux_W_m2"].to_numpy(), d["position_m"].to_numpy())
    return float(per_length / (k * dT))


def compare_case0(case: CfdCase | None = None, evidence: Registry | None = None
                  ) -> list[RunComparison]:
    reg = evidence if evidence is not None else Registry()
    case = case or load_case(CASE_ID, reg)
    meshes = sorted(case.file.meshes, key=lambda m: m.h)       # fine -> coarse
    tol = reg.get(TOLERANCE_ID).value
    out = []
    for run in case.file.runs:
        rc = RunComparison(run.name, "skipped")
        out.append(rc)
        values = case.resolved[run.name]
        k = values.get("material.thermal_conductivity")
        dT = values["bc.hot_wall"] - values["bc.cold_wall"]
        missing = [m.name for m in meshes if not case.result_path(run.name, m.name).exists()]
        if missing:
            rc.messages.append(f"results missing for mesh(es) {', '.join(missing)} "
                               f"(Fluent runs not done yet)")
            continue
        for m in meshes:
            rc.nu_by_mesh[m.name] = average_nusselt(case.read_results(run.name, m.name), k, dT)
        if len(meshes) < 3:
            rc.messages.append("GCI needs three meshes")
            continue
        f = [rc.nu_by_mesh[m.name] for m in meshes[:3]]
        rc.gci = gci(*f, *(m.h for m in meshes[:3]), evidence=reg)
        if rc.gci.oscillatory:
            rc.messages.append("oscillatory convergence: treat the GCI with caution")
        bench_id = BENCHMARK_IDS.get(run.name)
        rc.benchmark = reg.get(bench_id).value if bench_id else None
        if rc.benchmark is None:
            rc.messages.append(f"benchmark {bench_id} is empty: transcribe it from "
                               f"de Vahl Davis 1983")
            continue
        if tol is None:
            rc.messages.append(f"acceptance tolerance {TOLERANCE_ID} is empty: decide it")
            continue
        rc.tolerance = tol
        rc.status = "pass" if abs(rc.relative_difference) <= tol else "fail"
    return out


def main() -> None:
    for rc in compare_case0():
        nu = ", ".join(f"{k}={v:.4g}" for k, v in rc.nu_by_mesh.items()) or "-"
        extra = ""
        if rc.gci is not None:
            extra = (f"; p={rc.gci.p:.2f}, Nu_ext={rc.gci.f_extrapolated:.4g} "
                     f"+/- {rc.gci.band:.2g} (GCI)")
        if rc.relative_difference is not None:
            extra += f"; vs benchmark {rc.benchmark:g}: {rc.relative_difference * 100:+.2f} %"
        print(f"{rc.run}: {rc.status.upper()}  Nu: {nu}{extra}")
        for m in rc.messages:
            print(f"    - {m}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
