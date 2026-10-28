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

## Loads and humidity: methods, sources, open evidence items

**Model** (`core/coach.py`). The single zone now carries two states,
temperature T and humidity ratio W:

    C    · dT/dt = UA·(T_out − T) + Q_int − Q_hvac + Σ Q_load,i
    M_da · dW/dt = Σ m_load,i − m_cond            (m_cond = 0 for now)

M_da = V / v(T0, W0, p) is the dry-air mass (V = interior length × width ×
height; v from ASHRAE Eq. 26), held constant. Each load is a `LoadTerm`
`fn(t, T, W) -> (Q_sensible [W], m_water [kg/s])`, so loads may depend on time
and on the interior state; `simulate()` reports `Q_load_<name>_W` and
`m_load_<name>_kg_s` columns plus `W_kg_kg` and `RH`. Without `W0` the model is
exactly the piece-3 model (W and RH are NaN).

**Loads** (`core/loads.py`; positive = into the coach):

| Function | Equation | Evidence |
|---|---|---|
| `envelope` | Q = k·Ae·(T_out − T_in) | k: E-002; Ae definition: E-008 |
| `solar_glazing` | Q = I·A_glass·g | I: E-013…E-020 (× E-021 on vertical glazing); g: E-031 |
| `solar_opaque_sol_air` | T_sa = T_out + α·I/h_o; Q = k·A·(T_sa − T_in) | α: E-030; h_o: E-029 |
| `passengers` | Q_s = n·q_s; m_w = n·q_l / h_g0 | q_s, q_l: E-026/E-027 (split of E-025); h_g0: E-006 |
| `fresh_air` | m_da = V̇/v_out; Q_s = m_da·(h(T_out,W_in) − h(T_in,W_in)); Q_l = m_da·(h(T_out,W_out) − h(T_out,W_in)); m_w = m_da·(W_out − W_in) | V̇: E-028 per seat; psychrometrics: core.psychro |

Modelling choices to review: (1) `solar_opaque_sol_air` includes plain
conduction through A_opaque, so when A_opaque is already in Ae only the solar
increment (sol-air minus `envelope`) is added; (2) the opaque k is taken as the
coach-average k; (3) the fresh-air volume is taken at outside-air conditions;
(4) passenger latent heat is converted to water with h_g0 = 2501 kJ/kg (E-006),
consistent with the ASHRAE enthalpy formula; (5) no condensation yet, so RH may
exceed 1 in scenarios without dehumidification.

**Reference coach** (`data/reference_coach.yaml`). Every numeric field is an
evidence ID. `load_coach_definition()` resolves them through the registry,
checks each unit, and refuses to build the coach while any value is empty — it
currently lists E-034…E-041. A number written in the YAML instead of an ID is
rejected.

**k-value test** (`core/procedures.py`, UIC 553-1 7.2.3 — confirm against the
PDF). Steady when a full E-011 window (3 h) has |dT/dt| < E-032 (0.1 K/h,
assumption); the run must end with such a window. k = mean(Q)/(A·mean(T_in −
T_out)) over the last E-033 hour (1 h). Test conditions are checked against
E-009 (ΔT = 25 ± 5 K) and E-010 (T_out = 5 ± 2 °C) and violations are reported
in `condition_violations`.

**Evidence schema change.** `evidence.csv` gained a `tolerance` column (± in the
row's unit, used by E-009, E-010, E-024), and `value` may be text for
definitions (E-008). `registry.number(id, unit)` returns a float and checks the
unit; it raises `MissingEvidenceError` for empty or text values.

**Verification** (`tests/test_loads.py`, `tests/test_coach_humidity.py`,
`tests/test_coach.py`; verification-only parameters):

| Check | Criterion | Achieved |
|---|---|---|
| Each load function vs. a hand calculation in the test | 1e-9 relative | pass |
| Fresh air at the interior state | zero sensible, latent and water | pass |
| Passenger water only: W(t) linear (analytical) | < 1e-9 kg/kg | pass |
| Fresh air only: W(t) exponential (analytical) | < 1e-7 kg/kg | 1.6e-8 kg/kg |
| Moisture balance with all loads | 0.1 % | 8e-4 % |
| Energy balance with all loads | 0.1 % | 2e-5 % |
| k-test (UIC 553-1 rule) recovers k | 0.5 % | 0.005 % |
| k-test steady time = analytical t(0.1 K/h) + 3 h | ± one output step | pass |
| Reference coach loader | lists missing IDs | E-034…E-041 |

Open evidence items are listed in the last section of this file.

## Unit model L1: method, sources, limits (generic curves until real unit data)

**Unit card** (`data/units/<unit_id>.yaml`, schema `core.unit.UnitCard`). Every
field is `{value, unit, source, page, status}` with status `datasheet`,
`standard`, `literature` or `assumption`; an empty value is allowed only for
`assumption`. Units are converted to SI on read (kW→W, m³/h→m³/s, %→fraction).
`UnitCard.open_items()` / `report()` list every field that is missing or an
assumption, and `DXUnit.from_card()` refuses to build the model while any
required value is empty, listing those fields.

*Kaeser-Noske DHF14-RVKE* (`data/units/kaeser_noske_dhf14.yaml`, datasheet via
railway-news.com, p. 1): cooling 4.0 kW at 35 °C outside, heating 4.5 kW at
−20 °C outside, supply air 700 m³/h. Not on the datasheet, so assumption with
an empty value: indoor rating condition (dry- and wet-bulb), fresh airflow,
compressor power, fan power, fan position, refrigerant; SHR / sensible
capacity is absent and falls back to E-050 (empty). **The DHF14 model
therefore does not run yet**; the loader reports 8 open items.

**Method** (`core/unit.py`; EnergyPlus Engineering Reference, *Single-Speed
Electric DX Air Cooling Coil*):

| Step | Equation / rule | Source |
|---|---|---|
| Capacity curve | f = a + b·x + c·x² + d·y + e·y² + f·x·y; x = coil-inlet wet-bulb, y = outdoor dry-bulb (°C) | E-042…E-047 (EnergyPlus EMS Application Guide, Ex. 11) |
| Normalisation | Q_tot = Q_rated · f(x, y) / f(x_rating, y_rating) — exact at the datasheet point | unit card |
| Rated bypass factor | rated outlet from Q_rated and SHR; line inlet→outlet in (h, W) extended to saturation = ADP; BF = (h_out − h_ADP)/(h_in − h_ADP) | unit card; SHR fallback E-050 |
| BF at other flows | A0 = −ln(BF_rated)·ṁ_rated, BF = exp(−A0/ṁ) | EnergyPlus |
| Operating point | h_ADP = h_in − (Q/ṁ)/(1 − BF); SHR = (h(T_in, W_ADP) − h_ADP)/(h_in − h_ADP) | EnergyPlus |
| Dry coil | if W_ADP ≥ W_in, iterate the curve wet-bulb to the dry boundary; SHR = 1, no condensate | EnergyPlus |
| Part load | continuous fan, cycling compressor: outlet h and W = PLR-weighted mix of full-load outlet and inlet | EnergyPlus (no PLF curve) |
| Mixing box | mass-weighted W and h of return and fresh dry-air flows | ASHRAE Ch. 1 |
| Fan | P_fan added as sensible heat before the coil (blow-through) or after it (draw-through) | card |
| Heater | Q = PLR · Q_heat_rated, sensible | card |

Mass flows: supply ṁ_s = V̇_supply / v(rating inlet) (constant); fresh
ṁ_f = V̇_fresh / v(outside air); return = ṁ_s − ṁ_f.

**Coupling** (`simulate(..., unit=UnitOperation(unit, mode, part_load, RH_out))`).
The zone receives ṁ_s at the supply state and loses the same mass at zone
conditions: Q_supply = ṁ_s (h(T_s, W) − h(T, W)), ṁ_w = ṁ_s (W_s − W)
(`core.loads.supply_air`). Fresh air enters only through the mixing box; a
separate fresh-air load term is refused in this mode. Part load is a constant
input until the controller (piece 7).

**Limits of L1 — read before using results.**
- The capacity curve is a *generic* EnergyPlus example, not the unit's own
  performance map; only the rating point is unit-specific.
- Curve validity limits (E-048) are unknown, so the extrapolation flag is
  `None` in all runs until E-048 is filled.
- No airflow-fraction capacity curve, no part-load degradation (PLF), no
  compressor power (E-049 empty), no condensate re-evaporation.
- The card's "indoor" rating condition is taken as the coil inlet state.
- Fan motor heat is assumed to be entirely in the air stream.
- The split between sensible and latent heat depends on the humidity used as
  reference: the coil's sensible capacity and the zone's sensible loads differ
  by a term ṁ_f·c_p,w·ΔW·ΔT (0.4 % in the no-passenger test case).
- Passenger latent heat is converted with h_g0 (E-006), so the vapour's
  sensible part (≈ ṁ_w·1.86·T) is not in the coil's total-load bookkeeping.

**Verification** (`tests/test_unit.py`, `tests/test_coach_unit.py`;
verification-only unit `tests/data/verification_unit.yaml`; illustrated in
`docs/piece5_unit.ipynb`):

| Check | Criterion | Achieved |
|---|---|---|
| capacity_modifier(19.44, 35) | within 0.002 of 1 | 0.99984 |
| Q_tot at the datasheet rating condition | 1e-9 | exact (2e-16) |
| Coil at rating inlet reproduces rated SHR | 1e-9 | 2e-15 |
| Coil energy / moisture balance (8 inlet states, incl. part load and dry) | 0.1 % | pass |
| Outlet RH ≤ 100 %, 0 ≤ SHR ≤ 1, T_out ≤ T_in | — | pass |
| Dry coil when inlet dew point < ADP | no condensate, SHR = 1 | pass |
| Extrapolation flag with test limits | raised outside, not inside | pass |
| Mixing equal flows | mean W and h | 1e-12 |
| Pull-down: zone energy balance | 0.1 % | 6e-5 % |
| Pull-down: zone moisture balance | 0.1 % | 8e-3 % |
| Unit identity ṁ_s(h_s − h_z) = ṁ_f(h_o − h_z) − Q_coil + P_fan | every step | 1e-9 |
| Steady state: coil sensible = sensible loads + fan | 0.5 % | 0.014 % |
| No double counting (no passengers/sun): coil total = envelope + fresh-air Δh + fan | 0.5 % | 1e-3 % |
| Heating mode: supply gain = fresh-air Δh + heater + fan; energy balance | 1e-9 / 0.1 % | pass |

## Open evidence items

Every evidence row still marked `assumption`, "to confirm" or "confirm against
the PDF", grouped by who could supply it. `tests/test_registry.py` fails if an
open item is missing from this table. The EOG grouping is a first guess at
that party's scope — reassign rows as needed. Unit-card fields (e.g. the
DHF14's missing rating condition) are reported by `UnitCard.report()`.

**Team (coach thermal data doc)**

| ID | Item | Value | Status | Source / clause | What is open |
|---|---|---|---|---|---|
| E-002 | Coach overall heat transfer coefficient (k-value) | 1.2 W/m2K | assumption | - | placeholder value; needs coach-specific k (test or datasheet) |
| E-034 | Reference coach interior length (air volume) | *empty* | assumption | - | value missing; to confirm |
| E-035 | Reference coach interior width (air volume) | *empty* | assumption | - | value missing; to confirm |
| E-036 | Reference coach interior height (air volume) | *empty* | assumption | - | value missing; to confirm |
| E-037 | Reference coach exterior developed surface Ae (definition E-008) | *empty* | assumption | - | value missing; to confirm |
| E-038 | Reference coach glazing area per side | *empty* | assumption | - | value missing; to confirm |
| E-039 | Reference coach number of seats | *empty* | assumption | - | value missing; to confirm |
| E-040 | Reference coach effective interior heat capacity (air + interior mass) | *empty* | assumption | - | value missing; to confirm |
| E-041 | Reference coach equipment heat load | *empty* | assumption | - | value missing; to confirm |

**EOG (supplier / product data)**

| ID | Item | Value | Status | Source / clause | What is open |
|---|---|---|---|---|---|
| E-003 | XPS thermal conductivity | 0.032 W/mK | datasheet | Ravatherm XPS X PLUS datasheet / to confirm | page / product variant to confirm |
| E-030 | Opaque surface solar absorptance | *empty* | assumption | ASHRAE Fundamentals Ch.18 / glazing datasheet, to find | value missing; check against the PDF |
| E-031 | Glazing total solar energy transmittance (g-value) | *empty* | assumption | ASHRAE Fundamentals Ch.18 / glazing datasheet, to find | value missing; check against the PDF |

**Unit manufacturer / EnergyPlus data (unit performance)**

| ID | Item | Value | Status | Source / clause | What is open |
|---|---|---|---|---|---|
| E-048 | DX cooling capacity curve validity limits (x: inlet wet-bulb; y: outdoor dry-bulb) | *empty* | assumption | - | value missing; until filled the extrapolation flag is None |
| E-049 | DX EIR (power) curve coefficients | *empty* | assumption | - | value missing; needed only for compressor power |
| E-050 | Rated sensible heat ratio fallback (unit datasheet gives no SHR) | *empty* | assumption | - | value missing; coil refuses to run for cards without SHR |

**Standard documents (UIC 553, UIC 553-1, EN 13129, ASHRAE)**

| ID | Item | Value | Status | Source / clause | What is open |
|---|---|---|---|---|---|
| E-001 | Passenger sensible heat | 120 W/person | standard | UIC 553 / to confirm | CONFLICT with E-025 (sensible vs. total 120 W/person); clause to confirm |
| E-005 | Specific heat of dry air (enthalpy formula) | 1.006 kJ/(kg.K) | standard | ASHRAE Handbook Fundamentals 2017, Ch.1 / Eq. 32 | equation number: Eq. 32 vs. Eq. 30 (psychrolib) |
| E-006 | Enthalpy of water vapour at 0 C (formula constant) | 2501 kJ/kg | standard | ASHRAE Handbook Fundamentals 2017, Ch.1 / Eq. 32 | equation number: Eq. 32 vs. Eq. 30 (psychrolib) |
| E-007 | Specific heat of water vapour (formula constant) | 1.86 kJ/(kg.K) | standard | ASHRAE Handbook Fundamentals 2017, Ch.1 / Eq. 32 | equation number: Eq. 32 vs. Eq. 30 (psychrolib) |
| E-008 | k-value reference area definition (inner, outer or mean surface) | exterior developed surface Ae incl. roof insulation | standard | UIC 553-1, 2nd ed. 2005 / 7.2.2 | check against the PDF |
| E-009 | k-test temperature difference (inside - outside) | 25 +/- 5 K | standard | UIC 553-1, 2nd ed. 2005 / 7.2.3 | check against the PDF |
| E-010 | k-test outside (chamber) temperature | 5 +/- 2 degC | standard | UIC 553-1, 2nd ed. 2005 / 7.2.3 / Table 117 | DISCREPANCY: +/-5 K (text) vs. +/-2 K (Table 117); +/-2 K used |
| E-011 | k-test steady-state window | 3 h | standard | UIC 553-1, 2nd ed. 2005 / 7.2.3 | check against the PDF |
| E-012 | Interior temperature measuring height | 1.1 m | standard | UIC 553-1, 2nd ed. 2005 / 7.2.2 / App. C | check against the PDF |
| E-013 | Solar intensity, Zone I, 40 degC case | 800 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-014 | Solar intensity, Zone I, 28 degC case | 600 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-015 | Solar intensity, Zone I, 22 degC case | 500 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-016 | Solar intensity, Zone II, 35 degC case | 700 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-017 | Solar intensity, Zone II, 28 degC case | 600 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-018 | Solar intensity, Zone II, 22 degC case | 500 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-019 | Solar intensity, Zone III, 28 degC case | 600 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-020 | Solar intensity, Zone III, 22 degC case | 500 W/m2 | standard | UIC 553-1, 2nd ed. 2005 / 5.4.4 | check against the PDF |
| E-021 | Solar factor for vertical surfaces (cos 30 deg) | 0.866 | standard | UIC 553-1, 2nd ed. 2005 / App. B | check against the PDF |
| E-022 | EN 13129 Zone I extreme outside temperature | 45 degC | standard | EN 13129:2016 / to confirm | clause to confirm; check against the PDF |
| E-023 | EN 13129 Zone II design outside temperature | 35 degC | standard | EN 13129:2016 / to confirm | clause to confirm; check against the PDF |
| E-024 | EN 13129 interior comfort band | 24.5 +/- 2.5 degC | standard | EN 13129:2016 / to confirm | clause to confirm; check against the PDF |
| E-025 | Passenger total heat | 120 W/person | standard | UIC 553 / graph, to confirm | graph reading to confirm; CONFLICT with E-001 |
| E-026 | Passenger sensible heat (split of E-025) | *empty* | assumption | - | value missing; check against the PDF |
| E-027 | Passenger latent heat (split of E-025) | *empty* | assumption | - | value missing; check against the PDF |
| E-028 | Fresh air per seat (outside temperature above 26 degC) | 15 m3/(h.seat) | standard | UIC 553 / to confirm | clause to confirm; check against the PDF |
| E-029 | Outside surface heat transfer coefficient | *empty* | assumption | ASHRAE Fundamentals Ch.18 / glazing datasheet, to find | value missing; check against the PDF |
| E-032 | k-test stability criterion inside the steady-state window (max \|dT_in/dt\|) | 0.1 K/h | assumption | - | criterion inside the 3 h window not in notes; 0.1 K/h assumed |
| E-033 | k-test averaging period (k from the mean of the last hour) | 1 h | standard | UIC 553-1, 2nd ed. 2005 / 7.2.3 | check against the PDF |
