"""Test configuration for ninjax test suite."""

import jax


# jimgw.core.single_event.time_utils raises at import time when x64 is off, and
# ninjax.generation imports it, so this has to run before any ninjax import.
jax.config.update("jax_enable_x64", True)
