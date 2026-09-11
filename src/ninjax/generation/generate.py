"""Driver turning a table of binary properties into GW and EM signals."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence, Iterator
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy import units
from astropy.cosmology import Planck18, z_at_value
from astropy.time import Time
from fiesta.utils import write_event_data
from jax import Array
from jaxtyping import Key
from jesterTOV.tov.data_classes import FamilyData
from jimgw.core.prior import Prior
from jimgw.core.single_event.transform_utils import Mc_eta_to_m1_m2, Mc_q_to_m1_m2

from ninjax.generation.ejecta import binary_to_ejecta
from ninjax.generation.em import em_lightcurve_batch
from ninjax.generation.eos import EOSLike, resolve_family
from ninjax.generation.gw import gw_polarizations_batch, gw_strain, to_jim_params
from ninjax.generation.hdf5 import SignalWriter

# astropy exports its realizations lazily, so Planck18 carries no usable static type
COSMOLOGY: Any = Planck18

REQUIRED_COLUMNS = (
    "luminosity_distance",
    "ra",
    "dec",
    "psi",
    "theta_jn",
    "phase",
    "geocent_time",
    "chi_1",
    "chi_2",
)


def sample_parameters(prior: Prior, n_samples: int, rng_key: Key) -> pd.DataFrame:
    """Draw ``n_samples`` from a jim prior into a table."""
    samples = prior.sample(rng_key, n_samples)
    return pd.DataFrame({name: np.asarray(v) for name, v in samples.items()})


def _to_table(
    source: pd.DataFrame | Prior, n_samples: int, rng_key: Key | None
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        table = source.copy()
    elif rng_key is None:
        raise ValueError("rng_key is required when sampling from a prior")
    else:
        table = sample_parameters(source, n_samples, rng_key)

    if "mass_1" not in table:
        if "M_c" not in table:
            raise ValueError("need either mass_1/mass_2 or M_c with q or eta")
        M_c = jnp.asarray(table["M_c"].to_numpy())
        if "q" in table:
            mass_1, mass_2 = Mc_q_to_m1_m2(M_c, jnp.asarray(table["q"].to_numpy()))
        else:
            mass_1, mass_2 = Mc_eta_to_m1_m2(M_c, jnp.asarray(table["eta"].to_numpy()))
        table["mass_1"] = np.asarray(mass_1)
        table["mass_2"] = np.asarray(mass_2)

    missing = [name for name in REQUIRED_COLUMNS if name not in table]
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    return table


def _redshift(params: Mapping[str, Any]) -> float:
    if "redshift" in params:
        return float(params["redshift"])
    distance = float(params["luminosity_distance"]) * units.Mpc
    return float(z_at_value(COSMOLOGY.luminosity_distance, distance))


def _flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{key}_"))
        else:
            flat[f"{prefix}{key}"] = np.asarray(value)
    return flat


def _write_record(
    outdir: Path, index: int, record: Mapping[str, Any], params: Mapping[str, Any]
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if record.get("em"):
        mjd = Time(float(params["geocent_time"]), format="gps").mjd
        data = {
            filt: np.column_stack(
                [
                    mjd + np.asarray(lightcurve["time"]),
                    np.asarray(lightcurve["mag"]),
                    np.asarray(lightcurve.get("mag_err", 0.0))
                    * np.ones_like(lightcurve["mag"]),
                ]
            )
            for filt, lightcurve in record["em"].items()
        }
        write_event_data(str(outdir / f"{index}_em.dat"), data)

    if record.get("gw"):
        np.savez(outdir / f"{index}_gw.npz", **_flatten(record["gw"]))


def _generate_one(
    row: Mapping[str, Any],
    event_key: Key,
    *,
    family: FamilyData,
    gw_mode: Literal["polarizations", "strain", "none"],
    gw_opts: Mapping[str, Any],
    alpha: float,
    ratio_zeta: float,
) -> dict[str, Any]:
    """Prepare one event and generate its GW strain when requested.
    Batched GW polarizations EM generation is added later at the batch level.
    """
    params: dict[str, Any] = dict(row)
    redshift = _redshift(params)
    params["redshift"] = redshift
    params.setdefault(
        "inclination_EM", min(params["theta_jn"], np.pi - params["theta_jn"])
    )
    params["mass_1_source"] = params["mass_1"] / (1 + redshift)
    params["mass_2_source"] = params["mass_2"] / (1 + redshift)
    params.update(
        binary_to_ejecta(
            params["mass_1_source"],
            params["mass_2_source"],
            family,
            alpha=params.get("alpha", alpha),
            ratio_zeta=params.get("ratio_zeta", ratio_zeta),
        )
    )

    trigger_time = float(gw_opts.get("trigger_time", params["geocent_time"]))
    params["t_c"] = float(params["geocent_time"]) - trigger_time
    record: dict[str, Any] = {"parameters": params}
    if gw_mode == "strain":
        record["gw"] = gw_strain(
            to_jim_params(params),
            rng_key=event_key,
            **{**gw_opts, "trigger_time": trigger_time},
        )

    return record


def _stack_params(params: Sequence[Mapping[str, Any]]) -> dict[str, Array]:
    """Turn one mapping per event into columns of shape ``(n_events,)``."""
    return {
        name: jnp.stack([jnp.asarray(event[name]) for event in params])
        for name in params[0]
    }


def _generate_batches(
    table: pd.DataFrame,
    family: FamilyData,
    *,
    rng_key: Key | None,
    gw_mode: Literal["polarizations", "strain", "none"],
    frequencies: Array | None,
    gw_kwargs: Mapping[str, Any] | None,
    em_model: Any,
    error_budget: float | None,
    detection_limit: float | None,
    alpha: float,
    ratio_zeta: float,
    batch_size: int | None,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield ``(start, records)`` for each batch of the population in order."""
    gw_opts = dict(gw_kwargs or {})
    base_key = jax.random.key(0) if rng_key is None else rng_key

    rows = table.to_dict(orient="records")
    if batch_size is None:
        batch_size = max(len(rows), 1)
    elif batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        event_keys = [
            jax.random.fold_in(base_key, start + offset) for offset in range(len(chunk))
        ]
        batch = [
            _generate_one(
                row,
                event_key,
                family=family,
                gw_mode=gw_mode,
                gw_opts=gw_opts,
                alpha=alpha,
                ratio_zeta=ratio_zeta,
            )
            for row, event_key in zip(chunk, event_keys, strict=True)
        ]

        # One vectorised waveform call for the whole batch.
        if gw_mode == "polarizations":
            if frequencies is None:
                raise ValueError("gw_mode='polarizations' needs frequencies")
            polarizations = gw_polarizations_batch(
                _stack_params(
                    [to_jim_params(record["parameters"]) for record in batch]
                ),
                frequencies,
            )
            for index, record in enumerate(batch):
                record["gw"] = {
                    name: value[index] for name, value in polarizations.items()
                }

        # One surrogate call for the whole batch, then per-event noise.
        if em_model is not None:
            lightcurves = em_lightcurve_batch(
                [record["parameters"] for record in batch],
                em_model,
                error_budget=error_budget,
                detection_limit=detection_limit,
                rng_key=event_keys,
            )
            for record, lightcurve in zip(batch, lightcurves, strict=True):
                record["em"] = lightcurve

        yield start, batch


def generate_signals(
    source: pd.DataFrame | Prior,
    eos: EOSLike,
    *,
    n_samples: int = 1,
    rng_key: Key | None = None,
    gw_mode: Literal["polarizations", "strain", "none"] = "polarizations",
    frequencies: Array | None = None,
    gw_kwargs: Mapping[str, Any] | None = None,
    em_model: Any = None,
    error_budget: float | None = None,
    detection_limit: float | None = None,
    alpha: float = 0.0,
    ratio_zeta: float = 0.15,
    batch_size: int | None = None,
    outdir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate GW and EM signals for every binary in ``source``.

    ``source`` is either a table of binary properties or a jim prior to sample.
    ``eos`` is a macroscopic table path, a dict of jester parameters, or a
    ready-made family. ``gw_kwargs`` is forwarded to :func:`gw_strain`, which
    needs at least ``duration`` and ``sampling_frequency``.

    ``batch_size`` splits the population into chunks that are generated. The default
    processes the whole population as a single batch.
    """
    table = _to_table(source, n_samples, rng_key)
    family = resolve_family(eos)

    records = []
    for start, batch in _generate_batches(
        table,
        family,
        rng_key=rng_key,
        gw_mode=gw_mode,
        frequencies=frequencies,
        gw_kwargs=gw_kwargs,
        em_model=em_model,
        error_budget=error_budget,
        detection_limit=detection_limit,
        alpha=alpha,
        ratio_zeta=ratio_zeta,
        batch_size=batch_size,
    ):
        for offset, record in enumerate(batch):
            if outdir is not None:
                _write_record(
                    Path(outdir), start + offset, record, record["parameters"]
                )
            records.append(record)

    return records


def generate_signals_hdf5(
    source: pd.DataFrame | Prior,
    eos: EOSLike,
    path: str | os.PathLike[str],
    *,
    n_samples: int = 1,
    rng_key: Key | None = None,
    gw_mode: Literal["polarizations", "strain", "none"] = "polarizations",
    frequencies: Array | None = None,
    gw_kwargs: Mapping[str, Any] | None = None,
    em_model: Any = None,
    error_budget: float | None = None,
    detection_limit: float | None = None,
    alpha: float = 0.0,
    ratio_zeta: float = 0.15,
    batch_size: int,
) -> Path:
    """Generate signals batch by batch and stream them into one HDF5 file.
    Generated signal memory is bounded by the batch size.
    """
    table = _to_table(source, n_samples, rng_key)
    writer = SignalWriter(
        path,
        len(table),
        frequencies=frequencies if gw_mode == "polarizations" else None,
    )
    family = resolve_family(eos)

    with writer:
        for start, batch in _generate_batches(
            table,
            family,
            rng_key=rng_key,
            gw_mode=gw_mode,
            frequencies=frequencies,
            gw_kwargs=gw_kwargs,
            em_model=em_model,
            error_budget=error_budget,
            detection_limit=detection_limit,
            alpha=alpha,
            ratio_zeta=ratio_zeta,
            batch_size=batch_size,
        ):
            writer.write(start, batch)

    return writer.path
