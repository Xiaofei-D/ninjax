ninjax documentation
==========================

*SHORT TAGLINE FOR YOUR PACKAGE*

``ninjax`` does [DESCRIBE WHAT YOUR PACKAGE DOES].


What's in ``ninjax``?
============================

``ninjax`` provides [DESCRIBE THE KEY COMPONENTS].

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: :doc:`Module A <api/ninjax>`

       Description of module A

    .. grid-item:: :doc:`Module B <api/ninjax>`

       Description of module B


Getting started
===============

* Check out the :doc:`examples/getting_started` to get familiar with ``ninjax``.
* Dive into the code itself in the API reference of :doc:`api/ninjax`.


Installation
=============

Install the latest version by cloning the repository::

    git clone https://github.com/nuclear-multimessenger-astronomy/ninjax

We recommend using ``uv`` for managing the Python environment::

   uv venv --python=3.12
   source .venv/bin/activate

The package can then be installed directly::

    cd ninjax
    uv pip install -e .                # Basic install
    uv pip install -e ".[dev]"         # For developers (tests, docs)
    uv pip install -e ".[cuda13]"      # NVIDIA GPU, CUDA 13 wheels (Linux only)
    uv pip install -e ".[cuda12]"      # NVIDIA GPU, CUDA 12 wheels (Linux only)

Or using ``uv sync``::

    uv sync
    uv sync --extra dev         # For developers
    uv sync --extra cuda13      # NVIDIA GPU, CUDA 13 wheels (Linux only)
    uv sync --extra cuda12      # NVIDIA GPU, CUDA 12 wheels (Linux only)

GPU support
-----------

The default install is CPU-only. The two GPU extras are Linux-only and mutually
exclusive: which one you want depends on the NVIDIA driver installed on the machine,
not on any CUDA toolkit you may have loaded, since the wheels ship their own. Check
the driver with::

    nvidia-smi --query-gpu=driver_version --format=csv,noheader

.. list-table::
   :header-rows: 1

   * - Driver version
     - Extra
   * - 580 or newer
     - ``cuda13``
   * - 525 to 579
     - ``cuda12``
   * - older than 525
     - Neither. Ask your system administrator to update the driver

Prefer ``cuda13`` where the driver allows it, since jax intends to discontinue the
CUDA 12 wheels. Note that the "CUDA Version" printed in the ``nvidia-smi`` header is
the highest CUDA release the driver supports, so it is an upper bound rather than the
version to match.

Confirm that jax sees the GPU afterwards::

    uv run python -c "import jax; print(jax.devices())"


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/ninjax

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   developer_guide/contributing

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
