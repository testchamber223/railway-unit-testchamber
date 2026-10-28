# Railway HVAC Virtual Test Facility

An evidence-based simulation of a railway coach HVAC unit inside a climatic test
chamber. Every physical constant is traceable to a source via the evidence
registry — no hard-coded numbers in the physics.

## Layout

| Folder        | Purpose                                                            |
|---------------|--------------------------------------------------------------------|
| `core/`       | Physics library (psychrometrics, coach, unit, chamber, control, sensors, procedures) |
| `evidence/`   | `evidence.csv` + `registry.py` loader/validator                    |
| `data/`       | Coach definitions (YAML) whose every number is an evidence ID      |
| `validation/` | Verification cases                                                 |
| `ansys/`      | Future PyFluent scripts                                            |
| `web/`        | Future website                                                     |
| `docs/`       | Documentation                                                      |
| `tests/`      | pytest suite                                                       |

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv

# Activate
.venv\Scripts\activate          # Windows (PowerShell / cmd)
source .venv/Scripts/activate   # Windows (Git Bash)
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pytest
```

## Evidence rules

`evidence/evidence.csv` columns:
`id, name, symbol, value, tolerance, unit, source, clause_or_page, status, notes`

- `status` ∈ `standard`, `datasheet`, `literature`, `calculated`, `assumption`
- `source` may be empty **only** when `status` is `assumption`
- `value` may be empty only when `status` is `assumption`; it may be text for definitions (e.g. E-008)
- `tolerance` (optional) is a symmetric ± band in the row's unit
- `id` must be unique (format `E-xxx`)

Physics code must read constants through the registry:

```python
from evidence import registry

q_pax = registry.get("E-001")   # -> Evidence(value=120.0, unit='W/person', record=...)
registry.list_assumptions()     # -> every record still flagged as an assumption
```
