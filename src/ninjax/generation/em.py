"""Electromagnetic light curves from a fiesta surrogate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

# A surrogate's ``convert_to_mag`` reads these on top of its own parameters: the
# distance for every model, the redshift for the flux models.
EXTRA_SURROGATE_INPUTS = ("luminosity_distance", "redshift")


def _add_noise(
    times: Array,
    mags: Mapping[str, Array],
    key: Key,
    *,
    error_budget: float,
    detection_limit: float | None,
) -> dict[str, dict[str, Array]]:
    """Add Gaussian noise to one event's light curve, one subkey per filter.

    Iterates ``mags`` in the order the surrogate produced it, which is
    ``model.filters`` order, so each filter always draws the same subkey.
    """
    lightcurves = {}
    for filt, mag in mags.items():
        key, subkey = jax.random.split(key)
        mag = mag + error_budget * jax.random.normal(subkey, mag.shape)
        mag_err = jnp.full(mag.shape, error_budget)

        if detection_limit is not None:
            missed = mag > detection_limit
            mag = jnp.where(missed, detection_limit, mag)
            mag_err = jnp.where(missed, jnp.inf, mag_err)

        lightcurves[filt] = {"time": times, "mag": mag, "mag_err": mag_err}
    return lightcurves


def _surrogate_inputs(
    params: Sequence[Mapping[str, Any]], model: Any
) -> dict[str, Array]:
    """Stack the parameters needed by Fiesta into batched ``(n_events,)`` arrays."""
    names = list(model.parameter_names)
    names += [name for name in EXTRA_SURROGATE_INPUTS if name not in names]
    return {
        name: jnp.asarray([float(event[name]) for event in params]) for name in names
    }


def em_lightcurve(
    params: Mapping[str, Any],
    model: Any,
    *,
    error_budget: float | None = None,
    detection_limit: float | None = None,
    rng_key: Key | None = None,
) -> dict[str, dict[str, Array]]:
    """Apparent AB magnitudes per filter from a Fiesta surrogate.

    Returns the noiseless curve unless ``error_budget`` is given, in which case
    Gaussian scatter of that width is added. Points fainter than
    ``detection_limit`` are returned at the limit with an infinite error, which is
    how fiesta's ``EMLikelihood`` marks a non-detection.
    """
    times, mags = model.predict(dict(params))

    if error_budget is None:
        return {filt: {"time": times, "mag": mag} for filt, mag in mags.items()}

    return _add_noise(times, mags, 
        jax.random.key(0) if rng_key is None else rng_key,
        error_budget=error_budget, detection_limit=detection_limit)

def em_lightcurve_batch(
    params: Sequence[Mapping[str, Any]],
    model: Any,
    *,
    error_budget: float | None = None,
    detection_limit: float | None = None,
    rng_key: Sequence[Key] | None = None,
) -> list[dict[str, dict[str, Array]]]:
    """Generate EM light curves for a batch using Fiesta ``vpredict``.

    Returns one light curve mapping per event. If scatter is applied, each event
    uses its own RNG key so the result is independent of batch split.
    """
    times, mags = model.vpredict(_surrogate_inputs(params, model))

    # Preserve Fiesta's filter order so noise uses the same per-filter subkeys
    lightcurves = []
    for index in range(len(params)):
        event_mags = {filt: mag[index] for filt, mag in mags.items()}
        if error_budget is None:
            lightcurves.append({filt: {"time": times[index], "mag": mag}
                                for filt, mag in event_mags.items()})
        else:
            lightcurves.append(
                _add_noise(
                    times[index],
                    event_mags,
                    jax.random.key(0) if rng_key is None else rng_key[index],
                    error_budget=error_budget,
                    detection_limit=detection_limit,
                )
            )
    return lightcurves