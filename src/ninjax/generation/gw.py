"""Gravitational-wave signals for a binary neutron star."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import jax
from jax import Array
from jaxtyping import Key
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import GroundBased2G, get_detector_preset
from jimgw.core.single_event.transform_utils import m1_m2_to_Mc_eta
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2


def to_jim_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Map table columns onto the names jim's tidal waveform and detectors expect."""
    M_c, eta = m1_m2_to_Mc_eta(params["mass_1"], params["mass_2"])
    return {
        "M_c": M_c,
        "eta": eta,
        "s1_z": params["chi_1"],
        "s2_z": params["chi_2"],
        "lambda_1": params["lambda_1"],
        "lambda_2": params["lambda_2"],
        "d_L": params["luminosity_distance"],
        "phase_c": params["phase"],
        # exact for the aligned-spin waveform used here
        "iota": params["theta_jn"],
        "ra": params["ra"],
        "dec": params["dec"],
        "psi": params["psi"],
        "t_c": params["t_c"],
    }


def _detector_instances(names: Sequence[str]) -> list[GroundBased2G]:
    preset = get_detector_preset()
    instances = []
    for name in names:
        entry = preset[name]
        instances.extend(entry if isinstance(entry, list) else [entry])
    return instances


class _CompiledWaveform:
    """JIT-compiled scalar and batched waveform evaluation, 
    reused across events with the same ``f_ref``."""

    def __init__(self, f_ref: float) -> None:
        self.model = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)
        self._call = jax.jit(self.model.__call__)
        # The grid is shared by every event, so only the parameters are mapped.
        self._batched = jax.jit(jax.vmap(self.model.__call__, in_axes=(None, 0)))

    def __call__(
        self, frequencies: Array, params: Mapping[str, Any]
    ) -> dict[str, Array]:
        return self._call(frequencies, dict(params))

    def batched(
        self, frequencies: Array, params: Mapping[str, Any]
    ) -> dict[str, Array]:
        """Evaluate one shared grid against a batch of parameter sets."""
        return self._batched(frequencies, dict(params))

@cache
def _waveform(f_ref: float) -> _CompiledWaveform:
    """Return a cached compiled waveform for a reference frequency."""
    return _CompiledWaveform(f_ref)


@cache
def _detectors_with_psd(
    names: tuple[str, ...],
    psd_items: tuple[tuple[str, str], ...],
    asd_items: tuple[tuple[str, str], ...],
) -> tuple[tuple[GroundBased2G, ...], tuple[PowerSpectrum, ...]]:
    """Cache detector instances and their original PSDs for a configuration."""
    # Cached detectors are mutable; callers must reset them before reuse.
    psd_files, asd_files = dict(psd_items), dict(asd_items)
    instances = tuple(_detector_instances(names))
    psds = tuple(
        ifo.load_and_set_psd(
            psd_file=psd_files.get(ifo.name, ""),
            asd_file=asd_files.get(ifo.name, ""),
        )
        for ifo in instances
    )
    return instances, psds

def gw_polarizations(
    params: Mapping[str, Any],
    frequencies: Array,
    *,
    f_ref: float = 20.0,
) -> dict[str, Array]:
    """Plus and cross polarizations on a frequency grid for one binary, keyed ``"p"`` and ``"c"``."""
    return _waveform(f_ref)(frequencies, params)


def gw_polarizations_batch(
    params: Mapping[str, Any],
    frequencies: Array,
    *,
    f_ref: float = 20.0,
) -> dict[str, Array]:
    """Generate GW polarizations for a batch of binaries on a shared frequency grid.

    Parameter leaves have shape ``(n_events,)`` and outputs have shape
    ``(n_events, n_frequencies)``.
    """
    return _waveform(f_ref).batched(frequencies, params)


def gw_strain(
    params: Mapping[str, Any],
    *,
    detectors: Sequence[str] = ("H1", "L1", "V1"),
    duration: float,
    sampling_frequency: float,
    trigger_time: float,
    f_min: float = 20.0,
    f_max: float = 1024.0,
    f_ref: float = 20.0,
    psd_files: Mapping[str, str] | None = None,
    asd_files: Mapping[str, str] | None = None,
    zero_noise: bool = False,
    rng_key: Key | None = None,
) -> dict[str, dict[str, Any]]:
    """Project the waveform onto each detector and add coloured Gaussian noise.

    Without ``psd_files`` or ``asd_files`` jim downloads the GWTC-2 ASD, which
    needs network access and only covers H1, L1 and V1. Results are keyed by
    detector name, so ``"ET"`` expands to its three components.

    Detector and PSD setup is cached and reused across calls.
    """
    waveform = _waveform(f_ref)
    instances, psds = _detectors_with_psd(
        tuple(detectors),
        tuple(sorted((psd_files or {}).items())),
        tuple(sorted((asd_files or {}).items())),
    )

    signals = {}
    for ifo, psd in zip(instances, psds, strict=True):
        # Detectors are mutable, so restore a clean state and the original PSD
        # before each injection.
        ifo.clear_data_and_psd()
        ifo.set_psd(psd)
        ifo.inject_signal(
            duration=duration,
            sampling_frequency=sampling_frequency,
            trigger_time=trigger_time,
            waveform_model=waveform,
            parameters=dict(params),
            f_min=f_min,
            f_max=f_max,
            zero_noise=zero_noise,
            rng_key=rng_key,
        )
        signals[ifo.name] = {
            "frequencies": ifo.sliced_frequencies,
            "strain": ifo.sliced_fd_data,
            "psd": ifo.sliced_psd,
            "optimal_snr": ifo.optimal_snr,
            "match_filtered_snr": ifo.match_filtered_snr,
        }
    return signals
