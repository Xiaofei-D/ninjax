"""Tests for the BNS ejecta fitting formulas."""

import jax

import ninjax
from ninjax.generation.ejecta import GEOM_MSUN_KM, dynamic_mass_KrFo, log10_disk_mass


def test_version_is_a_string():
    assert isinstance(ninjax.__version__, str)


def test_importing_ninjax_enables_float64():
    assert jax.config.jax_enable_x64


def test_dynamic_mass_is_symmetric_under_swapping_the_two_stars():
    heavier_first = dynamic_mass_KrFo(1.6, 1.2, 0.18, 0.15)
    lighter_first = dynamic_mass_KrFo(1.2, 1.6, 0.15, 0.18)
    assert heavier_first == lighter_first


def test_dynamic_mass_of_a_typical_binary_is_a_few_thousandths_of_a_solar_mass():
    assert 1e-3 < dynamic_mass_KrFo(1.4, 1.4, 0.16, 0.16) < 1e-1


def test_dynamic_mass_of_very_compact_stars_is_clamped_to_zero():
    assert dynamic_mass_KrFo(1.4, 1.4, 0.25, 0.25) == 0.0


def test_disk_mass_falls_as_the_binary_approaches_prompt_collapse():
    r16_geom = 12.0 / GEOM_MSUN_KM
    light = log10_disk_mass(2.7, 0.9, 2.1, r16_geom)
    heavy = log10_disk_mass(3.5, 0.9, 2.1, r16_geom)
    assert heavy < light


def test_disk_mass_never_falls_below_the_floor():
    r16_geom = 12.0 / GEOM_MSUN_KM
    assert log10_disk_mass(8.0, 0.9, 2.1, r16_geom) >= -3.0
