"""Virtual sensors: what an instrument reports when it looks at the true model value.

Sensor types and their evidence (all assumption until datasheets are found):

    kind         instrument (assumed type)         accuracy  resolution  tau    sampling
    temperature  PT100 class A                     E-058 K   E-096 K     E-097  E-098
    humidity     capacitive RH sensor              E-059 -   E-099 -     E-100  E-101
    dew_point    chilled-mirror hygrometer         E-102 K   E-103 K     E-104  E-105
    airflow      nozzle / flow meter (relative)    E-060 -   E-106 -     E-107  E-108

Reading chain (``SensorSet.read``):

    true value -> first-order lag (time constant tau) -> + fixed calibration
    error (one per probe, uniform within +/- accuracy, seeded) -> rounded to
    the resolution -> held between sampling instants (sampling period)

Airflow accuracy and resolution are relative to the reading.
``SensorSet.ideal()`` has no lag, error, resolution or sampling.

Uncertainty of a derived quantity q(readings) is propagated by perturbing
each CHANNEL by its bound (accuracy + resolution / 2), worst case, linear sum:

    U = sum_c |q(bias_c = +bound_c) - q(0)|

A channel with several probes counts once (the mean of probes each within
+/- bound is within +/- bound). The bound covers calibration and resolution
errors; it does NOT cover the dynamic lag error, so measurements must be
taken at (near) steady state -- see docs, "Control and sensors".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from evidence import registry as _evidence
from evidence.registry import Registry

SENSOR_EVIDENCE: dict[str, dict[str, tuple[str, str]]] = {
    "temperature": {"accuracy": ("E-058", "K"), "resolution": ("E-096", "K"),
                    "tau_s": ("E-097", "s"), "sample_s": ("E-098", "s")},
    "humidity": {"accuracy": ("E-059", "-"), "resolution": ("E-099", "-"),
                 "tau_s": ("E-100", "s"), "sample_s": ("E-101", "s")},
    "dew_point": {"accuracy": ("E-102", "K"), "resolution": ("E-103", "K"),
                  "tau_s": ("E-104", "s"), "sample_s": ("E-105", "s")},
    "airflow": {"accuracy": ("E-060", "-"), "resolution": ("E-106", "-"),
                "tau_s": ("E-107", "s"), "sample_s": ("E-108", "s")},
}
RELATIVE_KINDS = ("airflow",)


@dataclass(frozen=True)
class SensorType:
    """Metrological model of one kind of sensor."""

    kind: str
    accuracy: float = 0.0       # +/- (absolute, or relative for airflow)
    resolution: float = 0.0     # smallest step (absolute, or relative for airflow)
    tau_s: float = 0.0          # first-order time constant [s]
    sample_s: float = 0.0       # sampling period [s] (0 = continuous)

    @property
    def bound(self) -> float:
        """Static error bound per probe: accuracy + resolution / 2."""
        return self.accuracy + self.resolution / 2

    @property
    def is_ideal(self) -> bool:
        return self.accuracy == self.resolution == self.tau_s == self.sample_s == 0.0

    def lag(self, t: np.ndarray, x: np.ndarray, y0: float | None = None) -> np.ndarray:
        """First-order response y' = (x - y) / tau, exact for piecewise-linear x."""
        if self.tau_s <= 0 or len(x) < 2:
            return np.array(x, dtype=float)
        y = np.empty(len(x))
        y[0] = x[0] if y0 is None else y0
        for k in range(len(x) - 1):
            dt = t[k + 1] - t[k]
            a = np.exp(-dt / self.tau_s)
            slope = (x[k + 1] - x[k]) / dt if dt > 0 else 0.0
            y[k + 1] = x[k + 1] - self.tau_s * slope + (y[k] - x[k] + self.tau_s * slope) * a
        return y

    def quantize(self, x: np.ndarray, scale: np.ndarray | float = 1.0) -> np.ndarray:
        if self.resolution <= 0:
            return x
        step = self.resolution * np.abs(scale) if self.kind in RELATIVE_KINDS else self.resolution
        step = np.where(np.asarray(step) > 0, step, 1.0)
        return np.round(x / step) * step

    def hold(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Zero-order hold at the sampling instants t0, t0 + T_s, ..."""
        if self.sample_s <= 0 or len(t) < 2:
            return y
        t_s = t[0] + np.floor((t - t[0]) / self.sample_s + 1e-9) * self.sample_s
        return np.interp(t_s, t, y)


@dataclass
class SensorSet:
    """Sensor types per kind plus a seeded calibration-error generator."""

    types: dict[str, SensorType]
    seed: int | None = 0
    evidence_ids: list[str] = field(default_factory=list)
    _errors: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def ideal(cls) -> SensorSet:
        return cls({kind: SensorType(kind) for kind in SENSOR_EVIDENCE})

    @classmethod
    def from_evidence(cls, evidence: Registry | None = None, seed: int | None = 0) -> SensorSet:
        number = evidence.number if evidence is not None else _evidence.number
        types, ids = {}, []
        for kind, params in SENSOR_EVIDENCE.items():
            values = {name: number(eid, unit) for name, (eid, unit) in params.items()}
            types[kind] = SensorType(kind, **values)
            ids += [eid for eid, _ in params.values()]
        return cls(types, seed=seed, evidence_ids=ids)

    @property
    def accuracy(self) -> dict[str, float]:
        return {k: t.accuracy for k, t in self.types.items()}

    @property
    def bound(self) -> dict[str, float]:
        return {k: t.bound for k, t in self.types.items()}

    @property
    def is_ideal(self) -> bool:
        return all(t.is_ideal for t in self.types.values())

    def errors(self, channel: str, kind: str, n_probes: int = 1) -> np.ndarray:
        """Fixed calibration error of each probe of ``channel`` (drawn once)."""
        if channel not in self._errors:
            a = self.types[kind].accuracy
            rng = np.random.default_rng(None if self.seed is None
                                        else [self.seed, sum(map(ord, channel))])
            self._errors[channel] = rng.uniform(-a, a, n_probes) if a > 0 else np.zeros(n_probes)
        return self._errors[channel]

    def read(self, channel: str, kind: str, true: np.ndarray | float, n_probes: int = 1,
             t: np.ndarray | None = None) -> np.ndarray:
        """Probe readings, shape (n_probes, len(true)).

        With ``t`` (time series, s) the lag and the sampling hold are applied;
        without it the readings are static (steady state).
        """
        st = self.types[kind]
        x = np.atleast_1d(np.asarray(true, dtype=float))
        if t is not None:
            x = st.lag(np.asarray(t, float), x)
        e = self.errors(channel, kind, n_probes)[:, None]
        y = x * (1 + e) if kind in RELATIVE_KINDS else x + e
        y = st.quantize(y, scale=x)
        if t is not None:
            y = np.vstack([st.hold(np.asarray(t, float), row) for row in y])
        return y


@dataclass(frozen=True)
class Channel:
    """A measurement channel used by a quantity: name, kind, number of probes."""

    name: str
    kind: str
    n_probes: int = 1


@dataclass(frozen=True)
class Measured:
    value: float
    uncertainty: float          # worst-case bound, linear sum over channels
    uncertainty_rss: float      # root-sum-square, for information


def propagate(fn: Callable[[Mapping[str, float]], float], channels: list[Channel],
              sensors: SensorSet) -> Measured:
    """Evaluate ``fn(bias)`` with readings and propagate channel bounds.

    ``fn`` receives a mapping channel -> extra bias (absolute, or relative for
    airflow) added on top of the sensor readings, and returns the quantity.
    """
    value = fn({})
    contributions = []
    for ch in channels:
        b = sensors.types[ch.kind].bound
        if b == 0:
            continue
        contributions.append(abs(fn({ch.name: b}) - value))
    c = np.array(contributions) if contributions else np.zeros(1)
    return Measured(float(value), float(c.sum()), float(np.sqrt((c ** 2).sum())))
