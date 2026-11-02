# ansys

CFD work for the virtual test facility. **No CFD results are included yet**:
every case is `planned`.

| Path | What |
|---|---|
| `cases/<case_id>/case.yaml` | What Fluent runs: geometry, solver settings, boundary conditions (evidence IDs / card fields / params), meshes, results path. Schema and loader: `core/cfd_import.py`. |
| `cases/<case_id>/results/{run}_{mesh}.csv` | Fluent exports (one CSV per run and mesh), added when runs are done. |
| `case0_inputs.py` | Cavity size and wall temperatures for the de Vahl Davis benchmark (`python -m ansys.case0_inputs`). |
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
