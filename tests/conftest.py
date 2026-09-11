"""Test configuration for ninjax test suite."""

import numpy as np
import pandas as pd
import pytest


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
