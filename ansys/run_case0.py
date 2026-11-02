"""Run Case 0 (de Vahl Davis square cavity) in Ansys Fluent from Python.

    python -m ansys.run_case0                         # all runs x all meshes
    python -m ansys.run_case0 --runs Ra1e5 --meshes coarse   # one pilot run
    python -m ansys.run_case0 --meshes-only           # only write .msh files for
                                                      # ansys/fluent_journals (Fluent GUI)

For every run (Ra) and mesh listed in ansys/cases/case0_devahldavis/case.yaml:
1. builds the uniform N x N mesh (ansys/case0_mesh.py, N = 1/h) and writes .msh;
2. starts Fluent 2D double precision (ansys/fluent_session.py), reads the mesh
   and runs Fluent's mesh check;
3. sets up exactly the case file: laminar, energy, Boussinesq air with the
   constant properties from ansys/case0_inputs.py (CoolProp at T_mean),
   gravity E-122, wall temperatures T_hot / T_cold, adiabatic top and bottom,
   PRESTO! pressure, second-order momentum and energy, residual targets;
4. iterates in blocks until the hot-wall Nusselt number stops changing;
5. writes results/<run>_<mesh>.csv (Nu_avg + checks) next to case.yaml.

Nu_avg = q_avg * L / (k * dT), with q_avg the area-weighted average heat flux on
the hot wall (independent of Fluent's 2D reference depth). A second value from
the total heat rate per unit depth and the hot/cold energy balance are stored
as checks. No benchmark value is used here; the comparison is
validation/case0_compare.py.

Everything is logged to ansys/_fluent_runs/case0/run_log.txt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
import time
import traceback
from pathlib import Path

import yaml

from ansys.case0_inputs import KELVIN, case0_inputs
from ansys.case0_mesh import build_cavity_mesh, check_mesh, write_msh
from ansys.fluent_session import launch_solver, stop_started_fluent, transcript_tail
from evidence.registry import Registry

HERE = Path(__file__).parent
CASE_DIR = HERE / "cases" / "case0_devahldavis"
WORK = HERE / "_fluent_runs" / "case0"
LOG = WORK / "run_log.txt"
FLOW_SCHEME = "Coupled"
RESIDUAL_TARGETS = {"continuity": 1e-6, "x-velocity": 1e-6, "y-velocity": 1e-6, "energy": 1e-9}
MAX_ITER = {"coarse": 3000, "medium": 6000, "fine": 12000}
BLOCK = 100
NU_REL_STEADY = 1e-7          # Nu change per block regarded as converged
IMBALANCE_MAX = 1e-3          # |Q_hot + Q_cold| / Q_hot required for a converged result


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def case_meshes() -> dict[str, int]:
    cf = yaml.safe_load((CASE_DIR / "case.yaml").read_text(encoding="utf-8"))
    return {m["name"]: int(round(1.0 / float(m["h"]))) for m in cf["meshes"]}


def first_number(obj, key: str | None = None) -> float:
    """First number in a (nested) Fluent return value, preferring entry ``key``."""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        if key is not None and key in obj:
            return first_number(obj[key])
        for v in obj.values():
            try:
                return first_number(v, key)
            except ValueError:
                pass
    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                return first_number(v, key)
            except ValueError:
                pass
    raise ValueError(f"no number in {obj!r}")


def apply(desc: str, fn, obj=None) -> None:
    """Run one set-up step; on failure log what the setting allows, then re-raise."""
    try:
        fn()
        log(f"  ok   {desc}")
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if obj is not None:
            for attr in ("allowed_values", "child_names", "get_object_names"):
                f = getattr(obj, attr, None)
                try:
                    hint += f" | {attr}: {f() if callable(f) else f}"
                except Exception:  # noqa: BLE001
                    pass
        log(f"  FAIL {desc}: {type(exc).__name__}: {exc}{hint}")
        raise


def setup_case(solver, r, T_hot_K: float, T_cold_K: float, T_mean_K: float) -> None:
    s = solver.settings
    oc = s.setup.general.operating_conditions
    air = s.setup.materials.fluid["air"]
    apply("steady pressure-based", lambda: setattr(s.setup.general.solver, "time", "steady"),
          s.setup.general.solver.time)
    apply("energy on", lambda: setattr(s.setup.models.energy, "enabled", True))
    apply("laminar", lambda: setattr(s.setup.models.viscous, "model", "laminar"),
          s.setup.models.viscous.model)
    apply("gravity on", lambda: setattr(oc.gravity, "enable", True))
    apply(f"gravity (0, -{r.g_m_s2})", lambda: setattr(oc.gravity, "components",
                                                     [0.0, -float(r.g_m_s2)]))
    apply("density boussinesq", lambda: setattr(air.density, "option", "boussinesq"),
          air.density.option)
    apply(f"density {r.rho_kg_m3:.6g}", lambda: setattr(air.density, "value", float(r.rho_kg_m3)))
    for name, val in (("specific_heat", r.cp_J_kgK), ("thermal_conductivity", r.k_W_mK),
                      ("viscosity", r.mu_Pa_s), ("therm_exp_coeff", r.beta_1_K)):
        prop = getattr(air, name)
        apply(f"{name} constant", lambda p=prop: setattr(p, "option", "constant"), prop.option)
        apply(f"{name} {val:.6g}", lambda p=prop, v=val: setattr(p, "value", float(v)))
    apply(f"operating temperature {T_mean_K:.2f} K",
          lambda: setattr(oc, "operating_temperature", T_mean_K))
    walls = s.setup.boundary_conditions.wall
    for zone, T in (("wall_hot", T_hot_K), ("wall_cold", T_cold_K)):
        th = walls[zone].thermal
        apply(f"{zone} temperature condition",
              lambda t=th: setattr(t, "thermal_condition", "Temperature"), th.thermal_condition)
        apply(f"{zone} T = {T:.2f} K", lambda t=th, v=T: setattr(t.temperature, "value", v))
    for zone in ("wall_top", "wall_bottom"):
        th = walls[zone].thermal
        apply(f"{zone} heat flux condition",
              lambda t=th: setattr(t, "thermal_condition", "Heat Flux"), th.thermal_condition)
        apply(f"{zone} q = 0 (adiabatic)", lambda t=th: setattr(t.heat_flux, "value", 0.0))
    m = s.solution.methods
    apply(f"p-v coupling {FLOW_SCHEME}",
          lambda: setattr(m.p_v_coupling, "flow_scheme", FLOW_SCHEME), m.p_v_coupling.flow_scheme)
    ds = m.spatial_discretization.discretization_scheme
    log(f"  discretization before: {ds.get_state()}")
    for eq, scheme in (("pressure", "presto!"), ("mom", "second-order-upwind"),
                       ("temperature", "second-order-upwind")):
        apply(f"{eq}: {scheme}", lambda e=eq, sc=scheme: ds.__setitem__(e, sc), ds)
    eqs = s.solution.monitor.residual.equations
    for name in eqs.get_object_names():
        if name in RESIDUAL_TARGETS:
            apply(f"residual {name} < {RESIDUAL_TARGETS[name]:g}",
                  lambda n=name: setattr(eqs[n], "absolute_criteria", RESIDUAL_TARGETS[n]))
    rd = s.solution.report_definitions
    apply("report q_hot", lambda: rd.flux.__setitem__(
        "q_hot", {"report_type": "flux-heattransfer", "boundaries": ["wall_hot"]}))
    apply("report q_cold", lambda: rd.flux.__setitem__(
        "q_cold", {"report_type": "flux-heattransfer", "boundaries": ["wall_cold"]}))
    apply("report qavg_hot", lambda: rd.surface.__setitem__(
        "qavg_hot", {"report_type": "surface-areaavg", "field": "heat-flux",
                     "surface_names": ["wall_hot"]}))
    init = s.solution.initialization
    apply(f"initial T = {T_mean_K:.2f} K",
          lambda: init.defaults.__setitem__("temperature", T_mean_K))
    apply("standard initialization", lambda: init.standard_initialize())


def reports(solver) -> dict[str, float]:
    out = solver.settings.solution.report_definitions.compute(
        report_defs=["q_hot", "q_cold", "qavg_hot"])
    vals = {}
    for key in ("q_hot", "q_cold", "qavg_hot"):
        found = None
        for item in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(item, dict) and key in item:
                found = first_number(item[key])
        vals[key] = found if found is not None else first_number(out, key)
    return vals


def run_one(solver, r, mesh_name: str, n: int, version: str, pyfluent_version: str) -> dict:
    L, dT, k = float(r.L_m), float(r.dT_K), float(r.k_W_mK)
    T_hot_K, T_cold_K = r.T_hot_C + KELVIN, r.T_cold_C + KELVIN
    T_mean_K = r.T_mean_C + KELVIN
    mesh = build_cavity_mesh(L, n)
    chk = check_mesh(mesh)
    log(f"mesh {mesh_name}: {n} x {n}, L = {L:.6g} m, {chk}")
    if chk["wrong_handedness"] or chk["max_closure_error"] > 1e-12 * L:
        raise RuntimeError(f"mesh self-check failed: {chk}")
    msh = write_msh(mesh, WORK / f"cavity_{r.run}_{mesh_name}.msh")
    s = solver.settings
    apply(f"read mesh {msh.name}", lambda: s.file.read_mesh(file_name=str(msh)))
    apply("Fluent mesh check", lambda: s.mesh.check())
    log("  transcript after mesh check: " + transcript_tail(WORK, n_lines=12))
    setup_case(solver, r, T_hot_K, T_cold_K, T_mean_K)
    t0, it, nu_prev, converged = time.time(), 0, None, False
    history = []
    while it < MAX_ITER[mesh_name]:
        s.solution.run_calculation.iterate(iter_count=BLOCK)
        it += BLOCK
        v = reports(solver)
        nu = v["qavg_hot"] * L / (k * dT)
        history.append(nu)
        change = abs(nu / nu_prev - 1) if nu_prev else math.inf
        imb = (v["q_hot"] + v["q_cold"]) / abs(v["q_hot"]) if v["q_hot"] else math.inf
        log(f"  iter {it:6d}: Nu = {nu:.8f} (change {change:.2e}), Q_hot = {v['q_hot']:.6g}, "
            f"Q_cold = {v['q_cold']:.6g}, imbalance {imb:.2e}")
        if not math.isfinite(nu) or (change == 0.0 and abs(imb) > IMBALANCE_MAX):
            raise RuntimeError("solver stopped or diverged (Nu frozen, hot/cold heat not "
                               "balanced); see the Fluent transcript")
        if change < NU_REL_STEADY and abs(imb) <= IMBALANCE_MAX:
            converged = True
            break
        nu_prev = nu
    if not converged:
        raise RuntimeError(f"not converged after {it} iterations; no result written")
    v = reports(solver)
    nu_area = v["qavg_hot"] * L / (k * dT)
    nu_total = v["q_hot"] / (k * dT)
    imbalance = (v["q_hot"] + v["q_cold"]) / abs(v["q_hot"])
    log(f"RESULT {r.run} {mesh_name}: Nu_avg = {nu_area:.6f} (from total heat per m depth "
        f"{nu_total:.6f}); hot/cold imbalance {imbalance:+.2e}; converged {converged} "
        f"after {it} iterations, {time.time() - t0:.0f} s")
    try:
        s.file.write_case_data(file_name=str(WORK / f"case0_{r.run}_{mesh_name}.cas.h5"))
    except Exception as exc:  # noqa: BLE001 - only a convenience for the GUI
        log(f"  (case/data not saved: {exc})")
    return dict(
        Nu_avg=nu_area, Nu_from_total_heat=nu_total, q_avg_hot_W_m2=v["qavg_hot"],
        Q_hot_W_per_m=v["q_hot"], Q_cold_W_per_m=v["q_cold"],
        energy_imbalance_rel=imbalance, converged=converged, iterations=it,
        Nu_last_block_change=(abs(history[-1] / history[-2] - 1) if len(history) > 1 else ""),
        n=n, n_cells=n * n, h=1.0 / n, Ra=float(r.Ra), L_m=L, T_hot_K=T_hot_K,
        T_cold_K=T_cold_K, k_W_mK=k, Pr=float(r.Pr), flow_scheme=FLOW_SCHEME,
        pressure_scheme="PRESTO!", momentum_energy_scheme="second-order upwind",
        fluent_version=version, pyfluent_version=pyfluent_version,
        run_at=dt.datetime.now().isoformat(timespec="seconds"),
    )


def write_result(run: str, mesh_name: str, row: dict) -> Path:
    path = CASE_DIR / "results" / f"{run}_{mesh_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", help="e.g. Ra1e5 (default: all)")
    ap.add_argument("--meshes", nargs="*", help="coarse / medium / fine (default: all)")
    ap.add_argument("--processors", type=int, default=1)
    ap.add_argument("--meshes-only", action="store_true",
                    help="only write the .msh files (for the Fluent GUI journals) and exit")
    a = ap.parse_args(argv)
    table = case0_inputs(evidence=Registry())
    meshes = case_meshes()
    if a.meshes_only:
        for r in table.itertuples():
            for name, n in meshes.items():
                m = build_cavity_mesh(float(r.L_m), n)
                chk = check_mesh(m)
                if chk["wrong_handedness"] or chk["max_closure_error"] > 1e-12 * float(r.L_m):
                    raise RuntimeError(f"mesh self-check failed: {chk}")
                print(write_msh(m, WORK / f"cavity_{r.run}_{name}.msh"))
        return 0
    runs = [r for r in table.itertuples() if not a.runs or r.run in a.runs]
    mesh_names = [m for m in meshes if not a.meshes or m in a.meshes]
    if not runs or not mesh_names:
        log(f"nothing to do: runs {[r.run for r in table.itertuples()]}, meshes {list(meshes)}")
        return 2
    log(f"=== Case 0: runs {[r.run for r in runs]} x meshes {mesh_names}")
    import ansys.fluent.core as pyfluent  # noqa: PLC0415
    launched = launch_solver(WORK, processor_count=a.processors, dimension=2)
    for note in launched.notes:
        log(f"  launch: {note}")
    solver = launched.session
    version = str(solver.get_fluent_version())
    log(f"Fluent {version}, PyFluent {pyfluent.__version__}, 2D double precision")
    failures = 0
    try:
        for r in runs:
            for mesh_name in mesh_names:
                try:
                    row = run_one(solver, r, mesh_name, meshes[mesh_name], version,
                                  pyfluent.__version__)
                    log(f"  wrote {write_result(r.run, mesh_name, row)}")
                except Exception:  # noqa: BLE001 - keep going, report at the end
                    failures += 1
                    log(f"ERROR {r.run} {mesh_name}:\n{traceback.format_exc()}")
                    log("  transcript: " + transcript_tail(WORK, n_lines=25))
    finally:
        try:
            solver.exit()
        except Exception as exc:  # noqa: BLE001
            log(f"exit: {exc}")
        time.sleep(2)
        log(f"stop: {stop_started_fluent(launched.process, WORK, launched.started_at)}")
    log(f"=== done, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
