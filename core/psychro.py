"""Psychrometrics: moist-air properties, wrapping psychrolib (SI units).

Units used throughout this module:
    T, Twb, dew point   degC
    RH                  fraction 0-1 (not percent)
    p, pressures        Pa
    W                   kg water / kg dry air
    enthalpy            J / kg dry air
    specific volume     m3 / kg dry air
    density             kg moist air / m3

All equation numbers refer to ASHRAE Handbook - Fundamentals (2017), Ch. 1,
as cited by psychrolib. ``p`` defaults to standard atmospheric pressure from
the evidence registry (E-004). Inputs are scalars.
"""

from __future__ import annotations

import psychrolib
from scipy.optimize import brentq

from evidence import registry

psychrolib.SetUnitSystem(psychrolib.SI)

# Validity range accepted by this module (input checking, not physics).
T_MIN = -40.0  # degC
T_MAX = 80.0   # degC


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _check_T(T: float, name: str = "T") -> float:
    if not T_MIN <= T <= T_MAX:
        raise ValueError(f"{name}={T} degC is outside the valid range {T_MIN}..{T_MAX} degC")
    return T


def _check_RH(RH: float) -> float:
    if not 0.0 <= RH <= 1.0:
        hint = " -- RH must be a fraction, e.g. 0.55 not 55" if 1.0 < RH <= 100.0 else ""
        raise ValueError(f"RH={RH} is outside 0..1{hint}")
    return RH


def _check_W(W: float) -> float:
    if W < 0.0:
        raise ValueError(f"W={W} kg/kg is negative")
    return W


def _pressure(p: float | None) -> float:
    if p is None:
        return registry.get("E-004").value
    if p <= 0.0:
        raise ValueError(f"p={p} Pa must be positive")
    return p


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

def sat_vapor_pressure(T: float) -> float:
    """Saturation vapour pressure of water [Pa] at temperature T [degC].

    Hyland-Wexler correlation, ASHRAE Eq. 5 (over ice, T <= 0 degC) and
    Eq. 6 (over liquid water, T > 0 degC).
    """
    return psychrolib.GetSatVapPres(_check_T(T))


def humidity_ratio(T: float, RH: float, p: float | None = None) -> float:
    """Humidity ratio W [kg/kg dry air] from T [degC], RH [0-1], p [Pa].

    pw = RH * pws(T) (ASHRAE Eq. 12 / 22), then W = 0.621945 pw / (p - pw)
    (ASHRAE Eq. 20).
    """
    return psychrolib.GetHumRatioFromRelHum(_check_T(T), _check_RH(RH), _pressure(p))


def rh_from_humidity_ratio(T: float, W: float, p: float | None = None) -> float:
    """Relative humidity [0-1] from T [degC], W [kg/kg dry air], p [Pa].

    ASHRAE Eq. 20 solved for pw, then RH = pw / pws(T) (Eq. 12 / 22).
    """
    return psychrolib.GetRelHumFromHumRatio(_check_T(T), _check_W(W), _pressure(p))


def enthalpy(T: float, W: float) -> float:
    """Moist-air specific enthalpy [J/kg dry air] from T [degC], W [kg/kg dry air].

    h = 1.006 T + W (2501 + 1.86 T)  [kJ/kg], ASHRAE Eq. 30 (2017 numbering per
    psychrolib; evidence E-005..E-007 record it as Eq. 32 -- to confirm).
    Computed by psychrolib; tests check it against E-005..E-007.
    """
    return psychrolib.GetMoistAirEnthalpy(_check_T(T), _check_W(W))


def dew_point(T: float, RH: float, p: float | None = None) -> float:
    """Dew-point temperature [degC] from T [degC], RH [0-1].

    Solves pws(Tdp) = RH * pws(T) by Newton iteration on ASHRAE Eq. 5 / 6
    (below 0 degC this is the frost point). ``p`` is accepted for a uniform
    signature but does not affect the result.
    """
    _pressure(p)
    return psychrolib.GetTDewPointFromRelHum(_check_T(T), _check_RH(RH))


def wet_bulb(T: float, RH: float, p: float | None = None) -> float:
    """Thermodynamic wet-bulb temperature [degC] from T [degC], RH [0-1], p [Pa].

    ASHRAE Eq. 33 (water) / Eq. 35 (ice) solved for t* by bisection.
    """
    return psychrolib.GetTWetBulbFromRelHum(_check_T(T), _check_RH(RH), _pressure(p))


def humidity_ratio_from_wet_bulb(T: float, Twb: float, p: float | None = None) -> float:
    """Humidity ratio W [kg/kg dry air] from T [degC], wet bulb Twb [degC], p [Pa].

    ASHRAE Eq. 33 (Twb > 0 degC) / Eq. 35 (Twb <= 0 degC).
    """
    _check_T(T)
    _check_T(Twb, "Twb")
    if Twb > T:
        raise ValueError(f"Twb={Twb} degC is above T={T} degC")
    return psychrolib.GetHumRatioFromTWetBulb(T, Twb, _pressure(p))


def moist_air_density(T: float, W: float, p: float | None = None) -> float:
    """Moist-air density [kg moist air / m3] from T [degC], W [kg/kg], p [Pa].

    rho = (1 + W) / v, ASHRAE Eq. 11, with v from Eq. 26.
    """
    return psychrolib.GetMoistAirDensity(_check_T(T), _check_W(W), _pressure(p))


def specific_volume(T: float, W: float, p: float | None = None) -> float:
    """Moist-air specific volume [m3 / kg dry air] from T [degC], W [kg/kg], p [Pa].

    v = R_da (T + 273.15) (1 + 1.607858 W) / p, ASHRAE Eq. 26.
    """
    return psychrolib.GetMoistAirVolume(_check_T(T), _check_W(W), _pressure(p))


# --------------------------------------------------------------------------
# Saturation and inverse relations (used by the DX coil model, core.unit)
# --------------------------------------------------------------------------

# Root-finding tolerances for the inverse relations (numerics, not physics).
_XTOL_T = 1e-9  # K


def sat_humidity_ratio(T: float, p: float | None = None) -> float:
    """Humidity ratio of saturated air W_s [kg/kg dry air] at T [degC], p [Pa].

    W_s = 0.621945 pws / (p - pws), ASHRAE Eq. 20 with pw = pws (Eq. 5 / 6).
    """
    return psychrolib.GetSatHumRatio(_check_T(T), _pressure(p))


def sat_air_enthalpy(T: float, p: float | None = None) -> float:
    """Enthalpy of saturated air [J/kg dry air] at T [degC], p [Pa] (ASHRAE Eq. 30 with W_s)."""
    return psychrolib.GetSatAirEnthalpy(_check_T(T), _pressure(p))


def sat_temperature_from_enthalpy(h: float, p: float | None = None) -> float:
    """Temperature [degC] of saturated air with enthalpy h [J/kg dry air] at p [Pa].

    Solves sat_air_enthalpy(T) = h by Brent's method on T_MIN..T_MAX
    (EnergyPlus PsyTsatFnHPb equivalent).
    """
    p = _pressure(p)
    lo, hi = sat_air_enthalpy(T_MIN, p), sat_air_enthalpy(T_MAX, p)
    if not lo <= h <= hi:
        raise ValueError(f"h={h} J/kg is outside the saturation range {lo:.0f}..{hi:.0f} J/kg")
    return brentq(lambda T: psychrolib.GetSatAirEnthalpy(T, p) - h, T_MIN, T_MAX, xtol=_XTOL_T)


def dry_bulb_from_enthalpy(h: float, W: float) -> float:
    """Dry-bulb temperature [degC] from enthalpy h [J/kg dry air] and W [kg/kg].

    ASHRAE Eq. 30 solved for T: T = (h - 2501e3 W) / (1006 + 1860 W).
    """
    T = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h, _check_W(W))
    return _check_T(T)


def humidity_ratio_from_enthalpy(T: float, h: float) -> float:
    """Humidity ratio [kg/kg dry air] from T [degC] and h [J/kg dry air] (ASHRAE Eq. 30 solved for W)."""
    return psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, _check_T(T))


def wet_bulb_from_humidity_ratio(T: float, W: float, p: float | None = None) -> float:
    """Thermodynamic wet-bulb temperature [degC] from T [degC], W [kg/kg], p [Pa].

    Solves humidity_ratio_from_wet_bulb(T, Twb) = W (ASHRAE Eq. 33 / 35) by
    Brent's method to 1e-9 K, giving a smooth result for ODE right-hand sides.
    """
    _check_T(T)
    _check_W(W)
    p = _pressure(p)
    W_sat = psychrolib.GetSatHumRatio(T, p)
    if W > W_sat * (1 + 1e-12):
        raise ValueError(f"W={W} exceeds saturation {W_sat:.6g} kg/kg at T={T} degC")
    if W >= W_sat:
        return T
    return brentq(lambda Twb: psychrolib.GetHumRatioFromTWetBulb(T, Twb, p) - W,
                  T_MIN, T, xtol=_XTOL_T)
