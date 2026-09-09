"""Tests for batch generation behaviour."""

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp, random

from ninjax.generation import generate_signals


N_EVENTS = 5
FREQUENCIES = jnp.arange(20.0, 40.0, 1.0)
STRAIN_RTOL = 1e-10

@pytest.fixture
def eos_file(tmp_path):
    masses = np.linspace(1.0, 2.2, 40)
    radii = 12.0 - 0.2 * (masses - 1.4)

    path = tmp_path / "eos.dat"
    np.savetxt(path, np.column_stack([radii, masses, 500.0 * (masses / 1.4) ** -6.0]))
    return path


@pytest.fixture
def table():
    rng = np.random.default_rng(0)

    return pd.DataFrame(
        {
            "mass_1": np.linspace(1.36, 1.40, N_EVENTS),
            "mass_2": np.linspace(1.30, 1.34, N_EVENTS),
            "luminosity_distance": np.linspace(100.0, 200.0, N_EVENTS),
            "ra": rng.uniform(0.0, 2 * np.pi, N_EVENTS),
            "dec": rng.uniform(-1.0, 1.0, N_EVENTS),
            "psi": rng.uniform(0.0, np.pi, N_EVENTS),
            "theta_jn": rng.uniform(0.0, np.pi, N_EVENTS),
            "phase": rng.uniform(0.0, 2 * np.pi, N_EVENTS),
            "geocent_time": np.full(N_EVENTS, 1187008882.43),
            "chi_1": np.zeros(N_EVENTS),
            "chi_2": np.zeros(N_EVENTS),
        }
    )


def run(table, eos_file, batch_size):
    return generate_signals(
        table,
        eos_file,
        rng_key=random.key(1),
        gw_mode="polarizations",
        frequencies=FREQUENCIES,
        batch_size=batch_size,
    )


def test_batch_size_preserves_generated_signals(table, eos_file):
    """Batching must preserve event ordering and generated signals."""
    whole = run(table, eos_file, None)
    batched = run(table, eos_file, 2)  # 2 + 2 + 1

    assert len(batched) == len(whole) == N_EVENTS

    for one, other in zip(whole, batched, strict=True):
        for polarization in ("p", "c"):
            np.testing.assert_allclose(
                other["gw"][polarization],
                one["gw"][polarization],
                rtol=STRAIN_RTOL,
                atol=0.0)

def test_batch_size_preserves_noise_realisation(table, eos_file, tmp_path):
    """An event's RNG must depend on its global index, not batch position."""
    asd = tmp_path / "asd.txt"
    frequencies = np.arange(0.0, 200.0, 0.125)
    np.savetxt(
        asd,
        np.column_stack([frequencies, np.full_like(frequencies, 1e-23)]))

    gw_kwargs = {
        "detectors": ["H1"],
        "duration": 4.0,
        "sampling_frequency": 256.0,
        "f_min": 20.0,
        "f_max": 100.0,
        "asd_files": {"H1": str(asd)},
    }

    def run_strain(batch_size):
        return generate_signals(
            table,
            eos_file,
            rng_key=random.key(1),
            gw_mode="strain",
            gw_kwargs=gw_kwargs,
            batch_size=batch_size,
        )

    whole = run_strain(None)
    batched = run_strain(2)

    for one, other in zip(whole, batched, strict=True):
        np.testing.assert_array_equal(other["gw"]["H1"]["strain"],one["gw"]["H1"]["strain"])


def test_batch_size_below_one_is_rejected(table, eos_file):
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        run(table, eos_file, 0)