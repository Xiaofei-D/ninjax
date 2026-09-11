"""Stream generated signals into one HDF5 file, one batch at a time."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self, cast

import h5py
import numpy as np


SHARED_STRAIN_FIELDS = ("frequencies", "psd")


class SignalWriter:
    """Stream batches of generated records into a HDF5 file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        n_events: int,
        *,
        frequencies: Any = None,
    ) -> None:
        self.path = Path(path)
        self.partial = self.path.with_name(
            f"{self.path.stem}.partial{self.path.suffix}"
        )
        for existing in (self.path, self.partial):
            if existing.exists():
                raise FileExistsError(f"refusing to overwrite {existing}")

        self.n_events = n_events
        self.frequencies = None if frequencies is None else np.asarray(frequencies)
        self._file: h5py.File | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.partial, "w")
        self._file.attrs["n_requested"] = self.n_events
        self._file.attrs["n_written"] = 0
        self._file.attrs["complete"] = False
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._file is not None

        if exc_type is None:
            # Check the number written matches that is requested
            n_written = int(cast(Any, self._file.attrs["n_written"]))
            if n_written != self.n_events:
                self._file.close()
                raise RuntimeError(f"expected {self.n_events} events, wrote {n_written}")

            self._file.attrs["complete"] = True
            self._file.close()
            os.replace(self.partial, self.path)
        else:
            self._file.close()

    def write(self, start: int, batch: Sequence[Mapping[str, Any]]) -> None:
        """Write one batch to its event slice and flush it to disk."""
        file = self._file
        assert file is not None, "use SignalWriter as a context manager"

        expected_start = int(cast(Any, file.attrs["n_written"]))
        if start != expected_start:
            raise ValueError(f"expected batch to start at {expected_start}, got {start}")

        if "parameters" not in file:
            self._create(batch[0])

        stop = start + len(batch)
        first = batch[0]

        for name in first["parameters"]:
            cast(h5py.Dataset,file[f"parameters/{name}"])[start:stop] = np.stack([np.asarray(record["parameters"][name]) for record in batch])

        for name, value in first.get("gw", {}).items():
            if isinstance(value, Mapping):
                for field in value:
                    if field not in SHARED_STRAIN_FIELDS:
                        cast(h5py.Dataset,file[f"gw/{name}/{field}"])[start:stop] = np.stack([np.asarray(record["gw"][name][field]) for record in batch])
            else:
                cast(h5py.Dataset,file[f"gw/{name}"])[start:stop] = np.stack([np.asarray(record["gw"][name]) for record in batch])

        for filt, curve in first.get("em", {}).items():
            for field in curve:
                cast(h5py.Dataset,file[f"em/{filt}/{field}"])[start:stop] = np.stack([np.asarray(record["em"][filt][field]) for record in batch])

        file.attrs["n_written"] = stop
        file.flush()

    def _create(self, record: Mapping[str, Any]) -> None:
        """Preallocate datasets from the structure of one record."""
        file = self._file
        assert file is not None
        n = self.n_events

        for name, value in record["parameters"].items():
            array = np.asarray(value)
            if array.shape != () or array.dtype.kind not in "biufc":
                raise TypeError(
                    f"parameter {name!r} is {array.dtype} with shape {array.shape}; "
                    "only scalars can be stored"
                )
            file.create_dataset(f"parameters/{name}", shape=(n,), dtype=array.dtype)

        if "gw" in record:
            if self.frequencies is not None:
                file.create_dataset("gw/frequencies", data=self.frequencies)

            for name, value in record["gw"].items():
                if isinstance(value, Mapping):
                    for field, entry in value.items():
                        array = np.asarray(entry)

                        if field in SHARED_STRAIN_FIELDS:
                            file.create_dataset(f"gw/{name}/{field}", data=array)

                        else:
                            file.create_dataset(f"gw/{name}/{field}", shape=(n, *array.shape), dtype=array.dtype)

                else:
                    array = np.asarray(value)
                    file.create_dataset(f"gw/{name}", shape=(n, *array.shape), dtype=array.dtype)

        if "em" in record:
            for filt, curve in record["em"].items():
                # Avoid create a wrong name in the dataset silently, otherwise will cause trouble during reading the dataset.
                if "/" in filt:
                    raise ValueError(f"filter name {filt!r} cannot contain '/'")

                for field, value in curve.items():
                    array = np.asarray(value)
                    file.create_dataset(f"em/{filt}/{field}", shape=(n, *array.shape), dtype=array.dtype)