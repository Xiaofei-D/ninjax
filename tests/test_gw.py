"""Regression tests for GW waveform compilation and detector reuse."""

import numpy as np
import pytest
from jax import numpy as jnp, random
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

from ninjax.generation import gw_polarizations, gw_strain
from ninjax.generation.gw import gw_polarizations_batch


F_REF = 20.0
FREQUENCIES = jnp.arange(20.0, 120.0, 0.5)

PARAMS = {
    "M_c": 1.1976,
    "eta": 0.2493,
    "s1_z": 0.01,
    "s2_z": -0.02,
    "lambda_1": 400.0,
    "lambda_2": 450.0,
    "d_L": 150.0,
    "phase_c": 1.3,
    "iota": 0.4,
    "ra": 2.1,
    "dec": -0.4,
    "psi": 0.7,
    "t_c": 0.0,
}


def test_compiled_waveform_matches_eager():
    eager = RippleIMRPhenomD_NRTidalv2(f_ref=F_REF)(FREQUENCIES, PARAMS)
    compiled = gw_polarizations(PARAMS, FREQUENCIES, f_ref=F_REF)

    assert compiled.keys() == eager.keys()
    for polarization in eager:
        np.testing.assert_allclose(
            compiled[polarization], eager[polarization], rtol=1e-10, atol=0.0
        )


def test_batched_polarizations_match_scalar_calls():
    events = 3
    varied = {
        "M_c": [1.18, 1.1976, 1.21],
        "d_L": [100.0, 150.0, 200.0],
        "iota": [0.2, 0.4, 0.6],
        "lambda_1": [380.0, 400.0, 420.0],
    }
    batch = {
        name: jnp.array(varied.get(name, [value] * events))
        for name, value in PARAMS.items()
    }

    batched = gw_polarizations_batch(batch, FREQUENCIES, f_ref=F_REF)

    for polarization in ("p", "c"):
        assert batched[polarization].shape == (events, FREQUENCIES.size)

    for index in range(events):
        scalar = gw_polarizations(
            {name: value[index] for name, value in batch.items()},
            FREQUENCIES,
            f_ref=F_REF,
        )
        for polarization in scalar:
            np.testing.assert_allclose(
                batched[polarization][index],
                scalar[polarization],
                rtol=1e-10,
                atol=0.0,
            )


@pytest.fixture
def asd_file(tmp_path):
    frequencies = np.arange(0.0, 200.0, 0.125)
    path = tmp_path / "asd.txt"
    np.savetxt(path, np.column_stack([frequencies, np.full_like(frequencies, 1e-23)]))
    return str(path)


def test_reused_detector_does_not_contaminate_events(asd_file):
    other = {**PARAMS, "M_c": 1.3, "d_L": 300.0, "ra": 0.4, "iota": 1.1}

    def strain(params):
        return gw_strain(
            params,
            detectors=("H1",),
            duration=4.0,
            sampling_frequency=256.0,
            trigger_time=1187008882.43,
            f_min=20.0,
            f_max=100.0,
            f_ref=F_REF,
            asd_files={"H1": asd_file},
            rng_key=random.key(3),
        )

    first_a = strain(PARAMS)
    strain(other)
    second_a = strain(PARAMS)

    for field in first_a["H1"]:
        np.testing.assert_array_equal(
            np.asarray(second_a["H1"][field]),
            np.asarray(first_a["H1"][field]),
            err_msg=f"H1/{field} was contaminated by the event in between",
        )
