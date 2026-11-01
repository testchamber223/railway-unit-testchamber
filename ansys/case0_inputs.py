"""Case 0 inputs: de Vahl Davis square-cavity benchmark, Ra = 1e3 .. 1e6.

For each Rayleigh number the cavity side L is chosen so that

    Ra = g * beta * dT * L^3 / (nu * alpha)          (exactly)

for a chosen temperature difference dT, with air properties evaluated at
the mean temperature T_mean and pressure p (E-004) from CoolProp's "Air"
pseudo-pure fluid (Lemmon et al. 2000 equation of state; Lemmon & Jacobsen
2004 transport properties -- see the CoolProp documentation; Bell et al.
2014, Ind. Eng. Chem. Res. 53(6), 2498-2508). g = E-122.

Boussinesq validity: beta * dT must stay below E-123 (0.1, assumption).
The benchmark is defined for Pr = 0.71 (E-125); the real-air Pr at T_mean is
reported so the deviation is visible (Fluent can match Pr exactly by setting
k = mu * cp / Pr if the team decides to).

T_mean and dT are user choices (defaults 20 degC, 10 K). Run:

    python -m ansys.case0_inputs [T_mean_C] [dT_K]

to print a table Fluent users can copy (Boussinesq density model, constant
properties at T_mean).
"""

from __future__ import annotations

import sys

import pandas as pd
from CoolProp.CoolProp import PropsSI

from evidence import registry as _evidence
from evidence.registry import Registry

RAYLEIGH = (1e3, 1e4, 1e5, 1e6)
KELVIN = 273.15


def air_properties(T_K: float, p_Pa: float) -> dict[str, float]:
    """Air properties at (T, p) from CoolProp ("Air" pseudo-pure fluid)."""
    rho = PropsSI("D", "T", T_K, "P", p_Pa, "Air")
    cp = PropsSI("C", "T", T_K, "P", p_Pa, "Air")
    k = PropsSI("L", "T", T_K, "P", p_Pa, "Air")
    mu = PropsSI("V", "T", T_K, "P", p_Pa, "Air")
    beta = PropsSI("isobaric_expansion_coefficient", "T", T_K, "P", p_Pa, "Air")
    nu, alpha = mu / rho, k / (rho * cp)
    return dict(rho=rho, cp=cp, k=k, mu=mu, beta=beta, nu=nu, alpha=alpha, Pr=nu / alpha)


def case0_inputs(T_mean_C: float = 20.0, dT_K: float = 10.0, rayleigh=RAYLEIGH,
                 evidence: Registry | None = None) -> pd.DataFrame:
    """Cavity side and wall temperatures per Ra (one row per Ra).

    Raises:
        ValueError: if beta * dT exceeds the Boussinesq limit E-123.
    """
    number = evidence.number if evidence is not None else _evidence.number
    g = number("E-122", "m/s2")
    p = number("E-004", "Pa")
    limit = number("E-123", "-")
    pr_bench = number("E-125", "-")
    props = air_properties(T_mean_C + KELVIN, p)
    if props["beta"] * dT_K >= limit:
        raise ValueError(f"beta * dT = {props['beta'] * dT_K:.3f} >= {limit} (E-123): "
                         f"Boussinesq approximation not valid; reduce dT")
    rows = []
    for Ra in rayleigh:
        L = (Ra * props["nu"] * props["alpha"] / (g * props["beta"] * dT_K)) ** (1 / 3)
        rows.append(dict(
            run=f"Ra{Ra:.0e}".replace("+0", "").replace("+", ""), Ra=Ra, L_m=L,
            T_mean_C=T_mean_C, dT_K=dT_K, T_hot_C=T_mean_C + dT_K / 2,
            T_cold_C=T_mean_C - dT_K / 2, beta_dT=props["beta"] * dT_K,
            rho_kg_m3=props["rho"], cp_J_kgK=props["cp"], k_W_mK=props["k"],
            mu_Pa_s=props["mu"], beta_1_K=props["beta"], Pr=props["Pr"],
            Pr_deviation=props["Pr"] / pr_bench - 1, g_m_s2=g))
    return pd.DataFrame(rows)


def rayleigh_of(row) -> float:
    """Recompute Ra from a table row (consistency check)."""
    nu = row["mu_Pa_s"] / row["rho_kg_m3"]
    alpha = row["k_W_mK"] / (row["rho_kg_m3"] * row["cp_J_kgK"])
    return row["g_m_s2"] * row["beta_1_K"] * row["dT_K"] * row["L_m"] ** 3 / (nu * alpha)


def main(argv: list[str]) -> None:
    T_mean = float(argv[1]) if len(argv) > 1 else 20.0
    dT = float(argv[2]) if len(argv) > 2 else 10.0
    df = case0_inputs(T_mean, dT)
    pd.set_option("display.width", 200)
    print(f"Case 0 (de Vahl Davis cavity) inputs, air at {T_mean} degC, dT = {dT} K "
          f"(CoolProp 'Air'; g = E-122; Boussinesq limit E-123)")
    print(df[["run", "Ra", "L_m", "T_hot_C", "T_cold_C", "beta_dT"]].to_string(
        index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nFluent material (Boussinesq), same for all runs:")
    r = df.iloc[0]
    for col in ("rho_kg_m3", "cp_J_kgK", "k_W_mK", "mu_Pa_s", "beta_1_K", "Pr"):
        print(f"  {col:10s} = {r[col]:.6g}")
    print(f"  operating temperature = {T_mean + KELVIN:.2f} K; Pr deviates "
          f"{r['Pr_deviation'] * 100:+.2f} % from the benchmark Pr = 0.71 (E-125)")


if __name__ == "__main__":
    main(sys.argv)
