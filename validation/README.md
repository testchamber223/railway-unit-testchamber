# validation

Verification cases (reference values from standards, handbooks and textbooks)
used to check the `core` models. Each CSV is run automatically by `pytest`.

| File                | Checks          | Run by                  |
|---------------------|-----------------|-------------------------|
| `psychro_cases.csv` | `core.psychro`  | `tests/test_psychro.py` |
| `gci.py` | mesh convergence (Celik et al. 2008) | `tests/test_gci.py` |
| `case0_compare.py` | Fluent Case 0 vs de Vahl Davis 1983 | `python -m validation.case0_compare`, `tests/test_cfd.py` |

Columns: `case_id, quantity, inputs, expected, unit, tolerance_pct, source, page, status`

- `inputs` is `key=value` pairs separated by `;` (e.g. `T=40;Twb=20;p=101325`).
- `unit` is the unit of `expected` as printed in the source; the test converts
  it to SI. Supported: `Pa, kPa, kg/kg, g/kg, J/kg, kJ/kg, degC, %, -, m3/kg, kg/m3`.
- Rows with an empty `expected` are skipped (with the reason printed) until a
  human fills them from the cited source and sets `status`.
