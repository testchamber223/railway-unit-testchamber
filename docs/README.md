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

## Single-zone coach model: equation, assumptions, verification

**Equation** (`core/coach.py`):

    C · dT/dt = UA · (T_out − T) + Q_int − Q_hvac,    UA = k · A

T, T_out in degC; C effective heat capacity (air + interior mass) in J/K;
k in W/(m²·K); A in m²; Q_int, Q_hvac in W (Q_hvac = heat removed; negative =
heating); t in s. `T_out`, `Q_int` and `Q_hvac` may each be a constant or a
function of t. Integrated with `scipy.integrate.solve_ivp` (LSODA); output is a
DataFrame `t_s, T_C, Q_env_W, Q_int_W, Q_hvac_W`. Helpers: `time_constant_s`
(τ = C/UA) and `steady_state_T` (T_ss = T_out + (Q_int − Q_hvac)/UA).

**Assumptions.** One perfectly mixed air zone; one lumped heat capacity; steady
envelope conduction (no wall thermal lag); no solar, infiltration or fresh-air
terms yet. The reference area A behind the k-value (inner, outer or mean
surface) is an open question — evidence **E-008** — and changes UA directly.

**k-value test** (`core/procedures.py`, `k_value_test`). The UIC 553-1 heater
method (section to confirm): the closed coach starts at chamber temperature,
heaters deliver a constant Q, HVAC off. Steady state is |dT/dt| < 0.1 K/h
(dT/dt from the model equation); k = Q / (A · (T_in − T_out)) is evaluated
at the end of the run. Any remaining drift biases k by about C·(dT/dt)/Q, so
the run must last well beyond the reported time to steady state.

**Verification** (`tests/test_coach.py`, illustrated in
`docs/piece3_single_zone.ipynb`). All checks use verification-only parameters
(A = 300 m², k = 1.2 W/m²K, C = 5·10⁶ J/K) that are *not* data for any coach.

| Check | Criterion | Achieved |
|---|---|---|
| (a) vs. analytical exponential solution | max error < 0.001 K | 1·10⁻⁶ K |
| (b) ∫(Q_env + Q_int − Q_hvac) dt = C·(T_end − T0) | 0.1 % | 2·10⁻⁶ % |
| (c) rtol 1e-6 vs 1e-9 | < 0.001 K | pass |
| (d) T(10 τ) vs. T_ss (from T0 = 20 °C) | < 0.001 K | pass |
| (e) k-value test recovers input k | 0.5 % | 0.005 % |
| (f) sinusoidal T_out: runs, energy balance closes | 0.1 % | 2·10⁻⁶ % |
| (g) k ≤ 0, C ≤ 0, A ≤ 0 | `ValueError` | pass |

Check (d) starts from 20 °C rather than the 45 °C of (a): from 45 °C the
41.7 K initial offset decays to e⁻¹⁰ · 41.7 K = 1.9 mK after 10 τ, which is
above the 1 mK criterion by the mathematics, not by any model error.
