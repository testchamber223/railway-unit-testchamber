# docs

Project documentation: model descriptions, assumptions and test-procedure notes.

## Psychrometrics: method, source, how it is verified

**Method.** `core/psychro.py` is a thin wrapper around
[psychrolib](https://github.com/psychrometrics/psychrolib) 2.5 in SI mode. It
adds one unit convention (degC, RH as a fraction 0–1, Pa, kg/kg dry air,
J/kg dry air), input checks (RH outside 0–1 → `ValueError` with a hint if it
looks like a percentage; T outside −40…80 degC → `ValueError`) and a default
pressure taken from the evidence registry (E-004, 101 325 Pa). No property
constant is written in `core/`.

**Source.** psychrolib implements ASHRAE Handbook – Fundamentals (2017),
Ch. 1: Hyland–Wexler saturation pressure (Eq. 5/6), humidity ratio (Eq. 20),
relative humidity (Eq. 12/22), enthalpy (Eq. 30 per psychrolib), wet bulb
(Eq. 33/35), specific volume (Eq. 26) and density (Eq. 11). Each function's
docstring names its equation.

**Verification** (`tests/test_psychro.py`):
1. *Reference cases* — every row of `validation/psychro_cases.csv` with an
   `expected` value is checked within its `tolerance_pct`; rows without one
   are skipped and the model's value is printed so a reviewer can compare.
   Currently: saturation pressure at 25/35/45 degC vs. Çengel & Boles
   Table A-4 (0.2 %); ASHRAE Ch. 1 Example 1 awaits values from the Handbook.
2. *Independent formula* — `enthalpy()` vs. the ASHRAE formula built only
   from registry constants E-005…E-007, within 0.1 %, at 5 temperatures × 3 RH.
3. *Consistency* — round trips RH→W→RH and T,RH→dew point→W_sat agree within
   1e-6; dew point at RH = 1 equals dry bulb within 0.01 K.
4. *Guards* — RH = 55 and out-of-range temperatures raise `ValueError`.

**Open items.** Fill Example 1 expected values; confirm Çengel page numbers;
confirm whether the enthalpy formula is Eq. 30 or Eq. 32 in the 2017 edition
(E-005…E-007 notes).
