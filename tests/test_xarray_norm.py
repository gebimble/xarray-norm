import pytest

import numpy as np

from toys import __version__
from toys.xarray_norm import radial_norm, axial_norm


def test_version():
    assert __version__ == '0.1.0'


def test_radial_norm(datasets):
    """Tests that integral of the data in two datasets is the same once passed
    through the radial integral normalisation routine."""
    # GIVEN two :py:class:`xarray.Dataset`s with 2D information
    # WHEN fed through the :py:func:`radial_norm` routine
    # THEN return two datasets whose integrals are equal
    dataset1, dataset2 = [x.sel(z=0, t=0, method='nearest') for x in datasets]

    assert all([len(x['value'].shape) == 2 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, ratio = radial_norm(dataset1, dataset2, 'value')

    assert dataset1 == normed_ds1
    assert dataset1['value'].sum() == normed_ds1['value'].sum()
    assert dataset2['value'].sum() != normed_ds2['value'].sum()
    assert dataset1['value'].sum().item() \
           == pytest.approx(normed_ds2['value'].sum().item())


def test_axial_norm_relative_integral(datasets):
    """Tests that integral of the data in two datasets is the same once passed
    through the radial integral normalisation routine.

    Single location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose integrals are equal

    dataset1, dataset2 = [
        x.sel(x=0, y=0, t=0, method='nearest') for x in datasets
    ]

    assert all([len(x['value'].shape) == 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2, 'value', 'z')

    assert normed_ds1['value'].integrate('z').item() == pytest.approx(
        np.ptp(dataset1['z'].values))

    assert normed_ds2['value'].integrate('z').item() == pytest.approx(
        np.ptp(dataset2['z'].values))
