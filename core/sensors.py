"""Virtual sensors: measurement channels with an accuracy bound and a fixed error.

Each channel has a kind (temperature, humidity, airflow) whose accuracy comes
from the evidence registry:

    temperature  +/- K           E-058
    humidity     +/- RH (0-1)    E-059
    airflow      +/- relative    E-060

A run draws one fixed error per probe, uniformly within +/- accuracy (a
calibration offset), from a seeded generator. ``SensorSet.ideal()`` has zero
accuracy and zero error.

Uncertainty of a derived quantity q(readings) is propagated by perturbing
each CHANNEL by its accuracy (worst case, linear sum):

    U = sum_c |q(bias_c = +a_c) - q(0)|

Channels with several probes (e.g. 8 chamber probes) are one channel: the
mean of probes each within +/- a is itself within +/- a. As long as every
probe error lies within its accuracy, the true value lies within
measured +/- U (to first order).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from evidence import registry as _evidence
from evidence.registry import Registry

SENSOR_EVIDENCE: dict[str, tuple[str, str]] = {
    "temperature": ("E-058", "K"),
    "humidity": ("E-059", "-"),
    "airflow": ("E-060", "-"),     # relative
}


@dataclass
class SensorSet:
    """Accuracy per sensor kind plus a seeded error generator."""

    accuracy: dict[str, float]
    seed: int | None = 0
    evidence_ids: list[str] = field(default_factory=list)
    _errors: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def ideal(cls) -> SensorSet:
        return cls({kind: 0.0 for kind in SENSOR_EVIDENCE})

    @classmethod
    def from_evidence(cls, evidence: Registry | None = None, seed: int | None = 0) -> SensorSet:
        number = evidence.number if evidence is not None else _evidence.number
        acc = {kind: number(eid, unit) for kind, (eid, unit) in SENSOR_EVIDENCE.items()}
        return cls(acc, seed=seed, evidence_ids=[eid for eid, _ in SENSOR_EVIDENCE.values()])

    @property
    def is_ideal(self) -> bool:
        return all(a == 0.0 for a in self.accuracy.values())

    def errors(self, channel: str, kind: str, n_probes: int = 1) -> np.ndarray:
        """Fixed error of each probe of ``channel`` (drawn once per channel)."""
        if channel not in self._errors:
            a = self.accuracy[kind]
            rng = np.random.default_rng(None if self.seed is None
                                        else [self.seed, sum(map(ord, channel))])
            self._errors[channel] = rng.uniform(-a, a, n_probes) if a > 0 else np.zeros(n_probes)
        return self._errors[channel]

    def read(self, channel: str, kind: str, true: np.ndarray | float,
             n_probes: int = 1) -> np.ndarray:
        """Probe readings, shape (n_probes, len(true)); airflow errors are relative."""
        true = np.atleast_1d(np.asarray(true, dtype=float))
        e = self.errors(channel, kind, n_probes)[:, None]
        return true * (1 + e) if kind == "airflow" else true + e


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
    """Evaluate ``fn(bias)`` with readings and propagate channel accuracies.

    ``fn`` receives a mapping channel -> extra bias (absolute, or relative for
    airflow) added on top of the sensor's own errors, and returns the quantity.
    """
    value = fn({})
    contributions = []
    for ch in channels:
        a = sensors.accuracy[ch.kind]
        if a == 0:
            continue
        contributions.append(abs(fn({ch.name: a}) - value))
    c = np.array(contributions) if contributions else np.zeros(1)
    return Measured(float(value), float(c.sum()), float(np.sqrt((c ** 2).sum())))
