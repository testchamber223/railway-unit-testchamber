"""Coach heat and moisture loads.

Sign convention: every load is positive when it adds heat (or water) to the
coach interior. All functions are pure: parameter values are passed in by the
caller, who takes them from the evidence registry (IDs cited per function).

Units: W for heat, kg/s for water, degC, m2, W/m2, kg/kg dry air, RH 0-1.
"""

from __future__ import annotations

from core import psychro
from evidence import registry

SECONDS_PER_HOUR = 3600.0
J_PER_KJ = 1e3


def _non_negative(**kw: float) -> None:
    for name, v in kw.items():
        if v < 0:
            raise ValueError(f"{name}={v} must be >= 0")


def _fraction(**kw: float) -> None:
    for name, v in kw.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}={v} must be in 0..1")


def envelope(k: float, Ae: float, T_out: float, T_in: float) -> float:
    """Conduction through the envelope [W]:  Q = k * Ae * (T_out - T_in).

    k [W/(m2.K)]: coach k-value (E-002 placeholder; reference coach via YAML).
    Ae [m2]: exterior developed surface incl. roof insulation (definition E-008).
    T_out, T_in [degC].
    """
    _non_negative(k=k, Ae=Ae)
    return k * Ae * (T_out - T_in)


def solar_glazing(I: float, A_glass: float, g: float) -> float:
    """Solar gain through glazing [W]:  Q = I * A_glass * g.

    I [W/m2]: solar intensity on the glazing plane -- UIC 553-1 intensities
    E-013..E-020, times the vertical-surface factor E-021 for side windows.
    A_glass [m2]: sunlit glazing area (reference coach: E-038 per side).
    g [-]: glazing total solar energy transmittance (E-031).
    """
    _non_negative(I=I, A_glass=A_glass)
    _fraction(g=g)
    return I * A_glass * g


def solar_opaque_sol_air(
    k: float, A_opaque: float, alpha: float, I: float, h_o: float, T_out: float, T_in: float,
) -> float:
    """Conduction through sunlit opaque surfaces using sol-air temperature [W].

        T_sol_air = T_out + alpha * I / h_o
        Q = k * A_opaque * (T_sol_air - T_in)

    k [W/(m2.K)] (E-002), A_opaque [m2], alpha [-] opaque solar absorptance
    (E-030), I [W/m2] (E-013..E-021), h_o [W/(m2.K)] outside surface
    coefficient (E-029), T_out, T_in [degC]. Steady state (no wall lag).

    This INCLUDES the plain conduction k * A_opaque * (T_out - T_in). If A_opaque
    is also part of the Ae passed to ``envelope``, add only the solar increment
    ``solar_opaque_sol_air(...) - envelope(k, A_opaque, T_out, T_in)``.
    """
    _non_negative(k=k, A_opaque=A_opaque, I=I)
    _fraction(alpha=alpha)
    if not h_o > 0:
        raise ValueError(f"h_o={h_o} must be > 0")
    T_sol_air = T_out + alpha * I / h_o
    return k * A_opaque * (T_sol_air - T_in)


def passengers(n: float, sensible_W: float, latent_W: float) -> tuple[float, float]:
    """Passenger loads: (sensible heat [W], water released [kg/s]).

        Q_sens = n * sensible_W
        m_w    = n * latent_W / h_g0

    n [-] passengers; sensible_W, latent_W [W/person] (E-026 / E-027, split of
    the total E-025). h_g0 = 2501 kJ/kg (E-006) converts latent heat to water,
    consistent with the ASHRAE moist-air enthalpy formula used in core.psychro.
    """
    _non_negative(n=n, sensible_W=sensible_W, latent_W=latent_W)
    h_g0 = registry.number("E-006", "kJ/kg") * J_PER_KJ
    return n * sensible_W, n * latent_W / h_g0


def fresh_air(
    volume_flow_m3_h: float, T_out: float, RH_out: float, T_in: float, W_in: float,
    p: float | None = None,
) -> tuple[float, float, float]:
    """Fresh-air load: (sensible [W], latent [W], water [kg/s]).

    Outside air at (T_out, RH_out) replaces the same mass of dry air leaving at
    the interior state (T_in, W_in). With h(T, W) from core.psychro.enthalpy:

        W_out  = humidity_ratio(T_out, RH_out, p)
        m_da   = volume_flow / 3600 / v(T_out, W_out, p)        [kg dry air/s]
        Q_sens = m_da * (h(T_out, W_in)  - h(T_in, W_in))
        Q_lat  = m_da * (h(T_out, W_out) - h(T_out, W_in))
        m_w    = m_da * (W_out - W_in)

    Q_sens + Q_lat = m_da * (h_out - h_in). volume_flow_m3_h [m3/h] is taken at
    outside-air conditions (UIC 553 per-seat rate: E-028). p [Pa] defaults to
    E-004.
    """
    _non_negative(volume_flow_m3_h=volume_flow_m3_h)
    W_out = psychro.humidity_ratio(T_out, RH_out, p)
    m_da = volume_flow_m3_h / SECONDS_PER_HOUR / psychro.specific_volume(T_out, W_out, p)
    h_out_Win = psychro.enthalpy(T_out, W_in)
    Q_sens = m_da * (h_out_Win - psychro.enthalpy(T_in, W_in))
    Q_lat = m_da * (psychro.enthalpy(T_out, W_out) - h_out_Win)
    return Q_sens, Q_lat, m_da * (W_out - W_in)
