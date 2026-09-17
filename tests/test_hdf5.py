"""Tests for streaming HDF5 output."""

import sys

import h5py
import numpy as np
import pytest
from jax import numpy as jnp, random

from ninjax.cli.generate import main
from ninjax.generation import generate_signals, generate_signals_hdf5
from ninjax.generation.hdf5 import SignalWriter


FREQUENCIES = jnp.arange(20.0, 40.0, 1.0)
GW_RTOL = 1e-10


def run(table, eos_file, *, path=None, batch_size=2):
    options = {
        "rng_key": random.key(1),
        "gw_mode": "polarizations",
        "frequencies": FREQUENCIES,
        "batch_size": batch_size,
    }
    if path is None:
        return generate_signals(table, eos_file, **options)
    return generate_signals_hdf5(table, eos_file, path, **options)


def test_hdf5_matches_generated_records(table, eos_file, tmp_path):
    records = run(table, eos_file)
    path = run(table, eos_file, path=tmp_path / "signals.h5")

    with h5py.File(path) as file:
        assert file.attrs["n_requested"] == 5
        assert file.attrs["n_written"] == 5
        assert file.attrs["complete"]
        np.testing.assert_array_equal(file["gw/frequencies"][()], FREQUENCIES)

        for index, record in enumerate(records):
            np.testing.assert_allclose(
                file["parameters/mass_1"][index], record["parameters"]["mass_1"]
            )
            for polarization in ("p", "c"):
                np.testing.assert_allclose(
                    file[f"gw/{polarization}"][index],
                    record["gw"][polarization],
                    rtol=GW_RTOL,
                    atol=0.0,
                )


def test_writer_stores_em_output(tmp_path):
    path = tmp_path / "signals.h5"
    batch = [
        {
            "parameters": {"mass_1": 1.4},
            "em": {
                "ztfg": {"time": np.array([0.5, 1.0]), "mag": np.array([20.0, 21.0])}
            },
        }
    ]

    with SignalWriter(path, 1) as writer:
        writer.write(0, batch)

    with h5py.File(path) as file:
        assert set(file) == {"parameters", "em"}
        np.testing.assert_array_equal(file["em/ztfg/time"][0], [0.5, 1.0])
        np.testing.assert_array_equal(file["em/ztfg/mag"][0], [20.0, 21.0])


def test_writer_stores_strain_output(tmp_path):
    """Strain records share the frequency grid and PSD but stack everything else."""
    path = tmp_path / "signals.h5"
    frequencies = np.arange(4.0)

    def record(index):
        return {
            "parameters": {"mass_1": 1.4 + index},
            "gw": {
                "H1": {
                    "frequencies": frequencies,
                    "psd": np.ones(4),
                    "strain": frequencies + index,
                    "optimal_snr": np.float64(10.0 + index),
                    "match_filtered_snr": np.complex128(9.0 + index),
                }
            },
        }

    with SignalWriter(path, 2) as writer:
        writer.write(0, [record(0)])
        writer.write(1, [record(1)])

    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["gw/H1/frequencies"][()], frequencies)
        np.testing.assert_array_equal(file["gw/H1/psd"][()], np.ones(4))
        assert file["gw/H1/strain"].shape == (2, 4)
        np.testing.assert_array_equal(file["gw/H1/strain"][1], frequencies + 1)
        assert file["gw/H1/optimal_snr"].shape == (2,)
        assert file["gw/H1/match_filtered_snr"].shape == (2,)
        np.testing.assert_array_equal(file["gw/H1/optimal_snr"][()], [10.0, 11.0])


def test_failed_run_leaves_partial_file(tmp_path):
    final = tmp_path / "signals.h5"
    partial = tmp_path / "signals.partial.h5"

    with pytest.raises(RuntimeError, match="generation failed"):
        with SignalWriter(final, 2) as writer:
            writer.write(0, [{"parameters": {"mass_1": 1.4}}])
            raise RuntimeError("generation failed")

    assert partial.exists()
    assert not final.exists()

    with h5py.File(partial) as file:
        assert file.attrs["n_written"] == 1
        assert not file.attrs["complete"]


def test_cli_output_format(table, eos_file, tmp_path, monkeypatch):
    params = tmp_path / "binaries.csv"
    table.head(2).to_csv(params, index=False)

    base = [
        "ninjax-generate",
        "--eos",
        str(eos_file),
        "--params-file",
        str(params),
        "--gw.f-min",
        "20",
        "--gw.f-max",
        "40",
        "--gw.delta-f",
        "1",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base,
            "--outdir",
            str(tmp_path / "h5"),
            "--output-format",
            "hdf5",
            "--batch-size",
            "1",
        ],
    )
    main()

    monkeypatch.setattr(sys, "argv", [*base, "--outdir", str(tmp_path / "files")])
    main()

    assert sorted(p.name for p in (tmp_path / "h5").iterdir()) == ["signals.h5"]
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == [
        "0_gw.npz",
        "1_gw.npz",
        "parameters.csv",
    ]
