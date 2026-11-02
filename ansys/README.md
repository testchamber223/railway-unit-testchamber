# ansys

CFD work for the virtual test facility. **Case 0 (benchmark cavity) is done**
with real Fluent results (2 Nov 2026); cases 1 and 2 are `planned`.

| Path | What |
|---|---|
| `cases/<case_id>/case.yaml` | What Fluent runs: geometry, solver settings, boundary conditions (evidence IDs / card fields / params), meshes, results path. Schema and loader: `core/cfd_import.py`. |
| `cases/<case_id>/results/{run}_{mesh}.csv` | Fluent exports (one CSV per run and mesh), added when runs are done. |
| `case0_inputs.py` | Cavity size and wall temperatures for the de Vahl Davis benchmark (`python -m ansys.case0_inputs`). |
| `case0_mesh.py` | Writes the Case 0 square-cavity meshes directly in Fluent's .msh format (no GUI meshing); self-checked for face handedness. |
| `run_case0.py` | Runs Case 0 in Fluent from Python (PyFluent) and writes the result CSVs; `--meshes-only` writes the meshes for the GUI journals. |
| `fluent_journals/` | The Python journals run inside the Fluent GUI that produced the Case 0 results. |
| `fluent_session.py` | Start Fluent from Python, wait until its gRPC port really accepts connections, connect, report the version, shut down (`python -m ansys.fluent_session`; as a test: `RUN_FLUENT=1` then `python -m pytest -m fluent -s`). |

**No `__init__.py` here, on purpose.** PyFluent lives in the `ansys`
namespace package (`ansys.fluent.core`). An `ansys/__init__.py` in this
project would hide it; without one Python merges both, so
`import ansys.fluent.core` and `import ansys.case0_inputs` both work.

PyFluent is optional: `python -m pip install -r requirements-ansys.txt` on a PC
with Ansys Fluent (Student 2026 R1 works). Fluent transcripts, cleanup scripts
and `_fluent_runs/` are ignored by git.

Only `case.yaml`, scripts and result CSVs go into git; Fluent cases, data,
meshes and Workbench projects are ignored (`.gitignore`). See docs/README.md,
"CFD and reduced-order model".

## Case 0 (de Vahl Davis cavity): how the results were produced

Two equivalent routes, same set-up and the same result CSVs
(`cases/case0_devahldavis/results/<run>_<mesh>.csv`):

1. **From VS Code (PyFluent):** `python -m ansys.run_case0` (all runs) or
   `python -m ansys.run_case0 --runs Ra1e5 --meshes coarse` (pilot). Log:
   `_fluent_runs/case0/run_log.txt`.
2. **Inside the Fluent GUI (used on 2 Nov 2026):** `python -m ansys.run_case0
   --meshes-only`, then Fluent Launcher with 2D + Double Precision and working
   directory `ansys/_fluent_runs/case0`, then in the Fluent console
   `/file/read-journal ../../fluent_journals/case0_all.py` and press Enter
   twice. Log: `_fluent_runs/case0/journal_log.txt`.

Compare with the benchmark: `python -m validation.case0_compare` (prints the
mesh study; pass/fail once E-117…E-121 are filled).
