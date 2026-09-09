"""Regression tests for batched EM generation."""

import numpy as np
import pandas as pd
import pytest
from fiesta.inference.lightcurve_model import SurrogateModel
from jax import numpy as jnp, random

from ninjax.generation import generate_signals
from ninjax.generation.em import em_lightcurve, em_lightcurve_batch


ERROR_BUDGET = 0.1
DETECTION_LIMIT = 9.5


class FakeSurrogate(SurrogateModel):
    def __init__(self):
        self.name = "fake"
        self.parameter_names = ["log10_mej_dyn", "inclination_EM"]
        self.filters = ["ztfr", "ztfg"]
        self.times = jnp.linspace(0.5, 10.0, 5)

    def project_input(self, x):
        return x

    def compute_output(self, x):
        return jnp.stack([x[0] + x[1], 2.0 * x[0] + x[1]])

    def project_output(self, y):
        return y

    def convert_to_mag(self, y, x):
        mags = y[:, None] + jnp.zeros_like(self.times)
        mags = mags + 5.0 * jnp.log10(x["luminosity_distance"])
        return self.times, dict(zip(self.filters, mags))


@pytest.fixture
def fake_em_model():
    return FakeSurrogate()


def event_params(index):
    return {
        "log10_mej_dyn": -2.0 - 0.1 * index,
        "inclination_EM": 0.3 + 0.05 * index,
        "luminosity_distance": 100.0 + 20.0 * index,
        "redshift": 0.02 + 0.001 * index,
    }


def test_batched_lightcurves_match_scalar(fake_em_model):
    params = [event_params(index) for index in range(4)]
    event_keys = [random.fold_in(random.key(1), index) for index in range(4)]

    batched = em_lightcurve_batch(
        params,
        fake_em_model,
        error_budget=ERROR_BUDGET,
        detection_limit=DETECTION_LIMIT,
        rng_key=event_keys,
    )

    for one, event, event_key in zip(batched, params, event_keys, strict=True):
        scalar = em_lightcurve(
            event,
            fake_em_model,
            error_budget=ERROR_BUDGET,
            detection_limit=DETECTION_LIMIT,
            rng_key=event_key,
        )

        for filt in scalar:
            for field in scalar[filt]:
                np.testing.assert_array_equal(one[filt][field], scalar[filt][field])


@pytest.fixture
def eos_file(tmp_path):
    masses = np.linspace(1.0, 2.2, 40)
    radii = 12.0 - 0.2 * (masses - 1.4)
    path = tmp_path / "eos.dat"
    np.savetxt(path, np.column_stack([radii, masses, 500.0 * (masses / 1.4) ** -6.0]))
    return path


@pytest.fixture
def table():
    n_events = 5
    rng = np.random.default_rng(0)

    return pd.DataFrame(
        {
            "mass_1": np.linspace(1.36, 1.40, n_events),
            "mass_2": np.linspace(1.30, 1.34, n_events),
            "luminosity_distance": np.linspace(100.0, 200.0, n_events),
            "ra": rng.uniform(0.0, 2 * np.pi, n_events),
            "dec": rng.uniform(-1.0, 1.0, n_events),
            "psi": rng.uniform(0.0, np.pi, n_events),
            "theta_jn": rng.uniform(0.0, np.pi, n_events),
            "phase": rng.uniform(0.0, 2 * np.pi, n_events),
            "geocent_time": np.full(n_events, 1187008882.43),
            "chi_1": np.zeros(n_events),
            "chi_2": np.zeros(n_events),
        }
    )


def test_batch_size_preserves_em_realisation(table, eos_file, fake_em_model):
    def run(batch_size):
        return generate_signals(
            table,
            eos_file,
            rng_key=random.key(1),
            gw_mode="none",
            em_model=fake_em_model,
            error_budget=ERROR_BUDGET,
            detection_limit=DETECTION_LIMIT,
            batch_size=batch_size,
        )

    whole = run(None)
    batched = run(2)

    assert len(batched) == len(whole)

    for one, other in zip(whole, batched, strict=True):
        for filt in fake_em_model.filters:
            for field in ("mag", "mag_err"):
                np.testing.assert_array_equal(other["em"][filt][field], one["em"][filt][field])