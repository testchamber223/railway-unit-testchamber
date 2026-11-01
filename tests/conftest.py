"""Shared VERIFICATION-ONLY fixtures.

tests/data/verification_overrides.yaml fills empty evidence rows with labelled
stand-in values (override label "verification-only"); they are maths checks,
not data for any real coach, unit, facility or standard.
"""

from pathlib import Path

import pytest
import yaml

from core.coach import load_coach_definition
from core.unit import load_unit_card
from evidence.registry import Registry

DATA = Path(__file__).parent / "data"
OVERRIDES = yaml.safe_load((DATA / "verification_overrides.yaml").read_text(encoding="utf-8"))


def verification_registry(**extra) -> Registry:
    """Evidence registry with the verification overrides (plus ``extra`` ones)."""
    values = dict(OVERRIDES)
    values.update({k.replace("_", "-"): v for k, v in extra.items()})
    return Registry().with_overrides(values, label="verification-only")


@pytest.fixture(scope="session")
def vreg() -> Registry:
    return verification_registry()


@pytest.fixture(scope="session")
def vcoach(vreg):
    return load_coach_definition(evidence=vreg)


@pytest.fixture(scope="session")
def vcard():
    return load_unit_card(DATA / "verification_unit.yaml")
