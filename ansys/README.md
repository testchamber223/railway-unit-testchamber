# ansys

CFD work for the virtual test facility. **No CFD results are included yet**:
every case is `planned`.

| Path | What |
|---|---|
| `cases/<case_id>/case.yaml` | What Fluent runs: geometry, solver settings, boundary conditions (evidence IDs / card fields / params), meshes, results path. Schema and loader: `core/cfd_import.py`. |
| `cases/<case_id>/results/{run}_{mesh}.csv` | Fluent exports (one CSV per run and mesh), added when runs are done. |
| `case0_inputs.py` | Cavity size and wall temperatures for the de Vahl Davis benchmark (`python -m ansys.case0_inputs`). |

Only `case.yaml`, scripts and result CSVs go into git; Fluent cases, data,
meshes and Workbench projects are ignored (`.gitignore`). See docs/README.md,
"CFD and reduced-order model".
