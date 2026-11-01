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


def vapour_enthalpy(T: float) -> float:
    """Specific enthalpy of water vapour h_g(T) = h_g0 + cp_w * T [J/kg] at T [degC].

    The vapour term of the ASHRAE moist-air enthalpy formula (E-006, E-007).
    """
    h_g0 = registry.number("E-006", "kJ/kg") * J_PER_KJ
    cp_w = registry.number("E-007", "kJ/(kg.K)") * J_PER_KJ
    return h_g0 + cp_w * T


def passengers(
    n: float, sensible_W: float, latent_W: float, T_in: float | None = None,
) -> tuple[float, float]:
    """Passenger loads: (sensible heat [W], water released [kg/s]).

        Q_sens = n * sensible_W
        m_w    = n * latent_W / h_g(T_in),   h_g(T) = h_g0 + cp_w T  (E-006, E-007)

    n [-] passengers; sensible_W, latent_W [W/person] (E-026 / E-027, split of
    the total E-025); T_in [degC] cabin temperature. With T_in the vapour enters
    at the cabin vapour enthalpy, so the latent gain n * latent_W is conserved
    exactly in the total-enthalpy balance. Without T_in, h_g0 (0 degC) is used.
    """
    _non_negative(n=n, sensible_W=sensible_W, latent_W=latent_W)
    h_g = vapour_enthalpy(0.0 if T_in is None else T_in)
    return n * sensible_W, n * latent_W / h_g


def fresh_air(
    volume_flow_m3_h: float, T_out: float, RH_out: float, T_in: float, W_in: float,
    p: float | None = None,
) -> tuple[float, float, float]:
    """Fresh-air load: (sensible [W], latent [W], water [kg/s]).

    Outside air at (T_out, RH_out) replaces the same mass of dry air leaving at
    the interior state (T_in, W_in). With h(T, W) from core.psychro.enthalpy:

        W_out  = humidity_ratio(T_out, RH_out, p)
        m_da   = volume_flow / 3600 / v(T_out, W_out, p)        [kg dry air/s]
        Q_sens = m_da * (h(T_out, W_out) - h(T_in, W_out))
        Q_lat  = m_da * (W_out - W_in) * h_g(T_in)
        m_w    = m_da * (W_out - W_in)

    (see ``supply_air`` for the sensible/latent convention).

    Q_sens + Q_lat = m_da * (h_out - h_in). volume_flow_m3_h [m3/h] is taken at
    outside-air conditions (UIC 553 per-seat rate: E-028). p [Pa] defaults to
    E-004.
    """
    _non_negative(volume_flow_m3_h=volume_flow_m3_h)
    W_out = psychro.humidity_ratio(T_out, RH_out, p)
    m_da = volume_flow_m3_h / SECONDS_PER_HOUR / psychro.specific_volume(T_out, W_out, p)
    return supply_air(m_da, T_out, W_out, T_in, W_in)


def supply_air(
    m_da: float, T_supply: float, W_supply: float, T_in: float, W_in: float,
) -> tuple[float, float, float]:
    """Load of an air stream entering the coach: (sensible [W], latent [W], water [kg/s]).

    m_da [kg dry air/s] enters at (T_supply, W_supply) and the same mass leaves
    at the interior state (T_in, W_in). With h(T, W) from core.psychro.enthalpy:

        Q_sens = m_da * (h(T_supply, W_supply) - h(T_in, W_supply))
        m_w    = m_da * (W_supply - W_in)
        Q_lat  = m_w * h_g(T_in)  =  m_da * (h_supply - h_in) - Q_sens

    The sensible part is evaluated at the stream's own humidity W_supply (the
    vapour that is heated or cooled is the stream's), and the latent part is the
    vapour added at the cabin vapour enthalpy h_g(T_in). Hence
    Q_sens + Q_lat = m_da (h_supply - h_in) exactly, which closes the cabin's
    total-enthalpy balance. Negative values mean cooling / dehumidifying. Used
    for fresh air (``fresh_air``) and the HVAC supply air (core.coach unit mode).
    """
    _non_negative(m_da=m_da)
    h_sup = psychro.enthalpy(T_supply, W_supply)
    Q_sens = m_da * (h_sup - psychro.enthalpy(T_in, W_supply))
    Q_total = m_da * (h_sup - psychro.enthalpy(T_in, W_in))
    return Q_sens, Q_total - Q_sens, m_da * (W_supply - W_in)
