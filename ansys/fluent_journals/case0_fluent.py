# Case 0 (de Vahl Davis square cavity) -- runs INSIDE Ansys Fluent 2026 R1.
# How to run (this is how the 2 Nov 2026 results were produced):
#   1. python -m ansys.run_case0 --meshes-only      (writes the 12 .msh files to
#      ansys/_fluent_runs/case0/)
#   2. Fluent Launcher: 2D, Double Precision, working directory
#      ansys/_fluent_runs/case0
#   3. Fluent console: /file/read-journal ../../fluent_journals/case0_all.py
#      (press Enter twice), or case0_pilot.py for one run.
# Uses Fluent's embedded PyFluent session object `solver` and the same
# settings paths as ansys/run_case0.py. Inputs are copied from
# ansys/cases/case0_devahldavis/case.yaml (CoolProp air at 20 degC, dT 10 K,
# g = E-122). Writes results/<run>_<mesh>.csv into that case folder and logs
# every step to journal_log.txt in this run folder.
import csv, datetime as dt, math, os, time, traceback

RUN_DIR = os.getcwd()
CASE_DIR = os.path.abspath(os.path.join(RUN_DIR, "..", "..", "cases", "case0_devahldavis"))
LOG = os.path.join(RUN_DIR, "journal_log.txt")

PROPS = dict(rho=1.2045751824931505, cp=1006.1440320870352, k=0.025873828302933142,
             mu=1.8205675178515367e-05, beta=0.0034209875148764166)
G = 9.80665                                   # E-122
T_MEAN_K, T_HOT_K, T_COLD_K = 293.15, 298.15, 288.15
L_BY_RUN = {"Ra1e3": 0.009870877895075436, "Ra1e4": 0.021266161758219417,
            "Ra1e5": 0.045816556615737344, "Ra1e6": 0.09870877895075433}
N_BY_MESH = {"coarse": 40, "medium": 80, "fine": 160}
MAX_ITER = {"coarse": 3000, "medium": 6000, "fine": 12000}
BLOCK, NU_REL_STEADY = 100, 1e-7
IMBALANCE_MAX = 1e-3       # |Q_hot + Q_cold| / Q_hot required for a converged result
RESIDUAL_TARGETS = {"continuity": 1e-6, "x-velocity": 1e-6, "y-velocity": 1e-6, "energy": 1e-9}
FLOW_SCHEME = "Coupled"


def log(msg):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def first_number(obj, key=None):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        if key is not None and key in obj:
            return first_number(obj[key])
        items = obj.values()
    elif isinstance(obj, (list, tuple)):
        items = obj
    else:
        raise ValueError("no number in %r" % (obj,))
    for v in items:
        try:
            return first_number(v, key)
        except ValueError:
            pass
    raise ValueError("no number in %r" % (obj,))


def apply(desc, fn, obj=None):
    try:
        fn()
        log("  ok   " + desc)
    except Exception as exc:
        hint = ""
        if obj is not None:
            for attr in ("allowed_values", "child_names", "get_object_names"):
                f = getattr(obj, attr, None)
                try:
                    hint += " | %s: %s" % (attr, f() if callable(f) else f)
                except Exception:
                    pass
        log("  FAIL %s: %s: %s%s" % (desc, type(exc).__name__, exc, hint))
        raise


def setup_case(s):
    oc = s.setup.general.operating_conditions
    air = s.setup.materials.fluid["air"]
    apply("steady", lambda: setattr(s.setup.general.solver, "time", "steady"),
          s.setup.general.solver.time)
    apply("energy on", lambda: setattr(s.setup.models.energy, "enabled", True))
    apply("laminar", lambda: setattr(s.setup.models.viscous, "model", "laminar"),
          s.setup.models.viscous.model)
    apply("gravity on", lambda: setattr(oc.gravity, "enable", True))
    apply("gravity (0, -%g)" % G, lambda: setattr(oc.gravity, "components", [0.0, -G]))
    apply("density boussinesq", lambda: setattr(air.density, "option", "boussinesq"),
          air.density.option)
    apply("density %.6g" % PROPS["rho"], lambda: setattr(air.density, "value", PROPS["rho"]))
    for name, key in (("specific_heat", "cp"), ("thermal_conductivity", "k"),
                      ("viscosity", "mu"), ("therm_exp_coeff", "beta")):
        prop = getattr(air, name)
        apply(name + " constant", lambda p=prop: setattr(p, "option", "constant"), prop.option)
        apply("%s %.6g" % (name, PROPS[key]),
              lambda p=prop, v=PROPS[key]: setattr(p, "value", v))
    apply("operating temperature %.2f K" % T_MEAN_K,
          lambda: setattr(oc, "operating_temperature", T_MEAN_K))
    walls = s.setup.boundary_conditions.wall
    log("  wall zones: %s" % walls.get_object_names())
    for zone, T in (("wall_hot", T_HOT_K), ("wall_cold", T_COLD_K)):
        th = walls[zone].thermal
        apply(zone + " temperature condition",
              lambda t=th: setattr(t, "thermal_condition", "Temperature"), th.thermal_condition)
        apply("%s T = %.2f K" % (zone, T), lambda t=th, v=T: setattr(t.temperature, "value", v))
    for zone in ("wall_top", "wall_bottom"):
        th = walls[zone].thermal
        apply(zone + " heat flux condition",
              lambda t=th: setattr(t, "thermal_condition", "Heat Flux"), th.thermal_condition)
        apply(zone + " q = 0", lambda t=th: setattr(t.heat_flux, "value", 0.0))
    m = s.solution.methods
    apply("p-v coupling " + FLOW_SCHEME,
          lambda: setattr(m.p_v_coupling, "flow_scheme", FLOW_SCHEME), m.p_v_coupling.flow_scheme)
    ds = m.spatial_discretization.discretization_scheme
    log("  discretization before: %s" % (ds.get_state(),))
    for eq, scheme in (("pressure", "presto!"), ("mom", "second-order-upwind"),
                       ("temperature", "second-order-upwind")):
        apply("%s: %s" % (eq, scheme), lambda e=eq, sc=scheme: ds.__setitem__(e, sc), ds)
    try:
        s.solution.monitor.residual.options.plot = False     # speed: no live residual plot
        log("  ok   residual plot off")
    except Exception as exc:
        log("  (residual plot left on: %s)" % exc)
    eqs = s.solution.monitor.residual.equations
    for name in eqs.get_object_names():
        if name in RESIDUAL_TARGETS:
            apply("residual %s < %g" % (name, RESIDUAL_TARGETS[name]),
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
    apply("initial T", lambda: init.defaults.__setitem__("temperature", T_MEAN_K))
    apply("standard initialization", lambda: init.standard_initialize())


def reports(s):
    out = s.solution.report_definitions.compute(report_defs=["q_hot", "q_cold", "qavg_hot"])
    vals = {}
    for key in ("q_hot", "q_cold", "qavg_hot"):
        found = None
        for item in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(item, dict) and key in item:
                found = first_number(item[key])
        vals[key] = found if found is not None else first_number(out, key)
    return vals


def run_one(s, run, mesh_name, version):
    L, n = L_BY_RUN[run], N_BY_MESH[mesh_name]
    msh = os.path.join(RUN_DIR, "cavity_%s_%s.msh" % (run, mesh_name))
    log("=== %s %s: L = %.6g m, %d x %d" % (run, mesh_name, L, n, n))
    apply("read mesh " + os.path.basename(msh), lambda: s.file.read_mesh(file_name=msh))
    apply("mesh check", lambda: s.mesh.check())
    setup_case(s)
    k, dT = PROPS["k"], T_HOT_K - T_COLD_K
    t0, it, prev, converged, hist = time.time(), 0, None, False, []
    while it < MAX_ITER[mesh_name]:
        s.solution.run_calculation.iterate(iter_count=BLOCK)
        it += BLOCK
        v = reports(s)
        nu = v["qavg_hot"] * L / (k * dT)
        hist.append(nu)
        change = abs(nu / prev - 1) if prev else float("inf")
        imb = (v["q_hot"] + v["q_cold"]) / abs(v["q_hot"]) if v["q_hot"] else float("inf")
        log("  iter %6d: Nu = %.8f (change %.2e) Q_hot = %.6g Q_cold = %.6g imbalance %.2e"
            % (it, nu, change, v["q_hot"], v["q_cold"], imb))
        if not math.isfinite(nu) or (change == 0.0 and abs(imb) > IMBALANCE_MAX):
            raise RuntimeError("solver stopped or diverged (Nu frozen, hot/cold heat not "
                               "balanced); see the Fluent transcript")
        if change < NU_REL_STEADY and abs(imb) <= IMBALANCE_MAX:
            converged = True
            break
        prev = nu
    if not converged:
        raise RuntimeError("not converged after %d iterations; no result written" % it)
    v = reports(s)
    nu_area = v["qavg_hot"] * L / (k * dT)
    nu_total = v["q_hot"] / (k * dT)
    imb = (v["q_hot"] + v["q_cold"]) / abs(v["q_hot"])
    log("RESULT %s %s: Nu_avg = %.6f (total-heat check %.6f), imbalance %+.2e, "
        "converged %s, %d iterations, %.0f s" % (run, mesh_name, nu_area, nu_total, imb,
                                                 converged, it, time.time() - t0))
    row = dict(Nu_avg=nu_area, Nu_from_total_heat=nu_total, q_avg_hot_W_m2=v["qavg_hot"],
               Q_hot_W_per_m=v["q_hot"], Q_cold_W_per_m=v["q_cold"],
               energy_imbalance_rel=imb, converged=converged, iterations=it,
               Nu_last_block_change=(abs(hist[-1] / hist[-2] - 1) if len(hist) > 1 else ""),
               n=n, n_cells=n * n, h=1.0 / n, L_m=L, T_hot_K=T_HOT_K, T_cold_K=T_COLD_K,
               k_W_mK=k, Pr=PROPS["mu"] * PROPS["cp"] / k, flow_scheme=FLOW_SCHEME,
               pressure_scheme="PRESTO!", momentum_energy_scheme="second-order upwind",
               fluent_version=version, run_by="Fluent GUI journal ansys/_fluent_runs/case0",
               run_at=dt.datetime.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.join(CASE_DIR, "results"), exist_ok=True)
    path = os.path.join(CASE_DIR, "results", "%s_%s.csv" % (run, mesh_name))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    log("  wrote " + path)
    apply("save case/data", lambda: s.file.write_case_data(
        file_name=os.path.join(RUN_DIR, "case0_%s_%s.cas.h5" % (run, mesh_name))))


def main(runs, meshes):
    s = solver.settings
    try:
        version = str(solver.get_fluent_version())
    except Exception:
        version = "Fluent 2026 R1"
    log("##### Case 0 journal: runs %s x meshes %s (%s)" % (runs, meshes, version))
    fails = 0
    for mesh_name in meshes:          # all Ra on the coarse mesh first, then medium, fine
        for run in runs:
            try:
                run_one(s, run, mesh_name, version)
            except Exception:
                fails += 1
                log("ERROR %s %s:\n%s" % (run, mesh_name, traceback.format_exc()))
    log("##### done, %d failure(s)" % fails)
