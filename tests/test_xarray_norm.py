import pytest

import numpy as np

import matplotlib.pyplot as plt

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


def test_axial_norm_relative_integral_single_loc(datasets):
    """Tests that integral of the data, as calculated by `axial_norm`,
    for two datasets is equal to the range of the direction of integration
    once passed through the radial integral normalisation routine.

    Single location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose integrals are equal to the range of the
    # direction of integration

    dataset1, dataset2 = [
        x.isel(t=0).sel(x=np.random.choice(x['x']),
                        y=np.random.choice(x['y']),) for x in datasets
    ]

    assert all([len(x['value'].shape) == 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='relative-integral')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert nds1_vals.integrate('z').item() \
           == pytest.approx(np.ptp(dataset1['z'].values))

    assert nds2_vals.integrate('z').item() \
           == pytest.approx(np.ptp(dataset2['z'].values))


def test_axial_norm_relative_integral_multiple_loc(datasets):
    """Tests that integral of the data, as calculated by `axial_norm`,
    for two datasets is equal to the range of the direction of integration
    once passed through the radial integral normalisation routine.

    Multiple location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose integrals at each radial location are
    # equal to the range of the direction of integration

    dataset1, dataset2 = [
        x.isel(t=0).sel(x=np.random.choice(x['x'], 2),
                        y=np.random.choice(x['y'], 2)) for x in datasets
    ]

    assert all([len(x['value'].shape) != 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='relative-integral')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert (nds1_vals.integrate('z')
            == pytest.approx(np.ptp(dataset1['z'].values))).all()

    assert (nds2_vals.integrate('z')
            == pytest.approx(np.ptp(dataset2['z'].values))).all()


def test_axial_norm_peak_integral_single_loc(datasets):
    """Tests that integral of the data, as calculated by `axial_norm`,
    for two datasets is equal to the range of the direction of integration
    once passed through the radial integral normalisation routine.

    Identical tests to `test_axial_norm_relative_integral_single_loc`,
    with identical results, as this is at a single location.

    Single location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose integrals are equal to the range of the
    # direction of integration

    dataset1, dataset2 = [
        x.isel(t=0).sel(x=np.random.choice(x['x']),
                        y=np.random.choice(x['y']),) for x in datasets
    ]

    assert all([len(x['value'].shape) == 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='peak-integral')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert nds1_vals.integrate('z').item() \
           == pytest.approx(np.ptp(dataset1['z'].values))

    assert nds2_vals.integrate('z').item() \
           == pytest.approx(np.ptp(dataset2['z'].values))


def test_axial_norm_peak_integral_multiple_loc(datasets):
    """Tests that integral of the data, as calculated by `axial_norm`,
    for two datasets is equal to the range of the direction of integration
    once passed through the radial integral normalisation routine.

    Multiple location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose integrals at each radial location are
    # equal to or less than the range of the dimension of integration

    dataset1, dataset2 = [
        x.isel(t=0) for x in datasets
    ]

    assert all([len(x['value'].shape) != 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='peak-integral')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    nds1_int = nds1_vals.integrate('z')
    ds1_range = np.ptp(dataset1['z'].values)

    nds2_int = nds2_vals.integrate('z')
    ds2_range = np.ptp(dataset2['z'].values)

    # values can either be less than or equal to the range in the direction
    # of integration; sadly there is no 'approximately less-than-or-equal-to
    # function in numpy, so we have to use a combination of <= and `np.isclose`
    assert np.logical_or(nds1_int <= ds1_range,
                         np.isclose(nds1_int, ds1_range)).all()

    assert np.logical_or(nds2_int <= ds2_range,
                         np.isclose(nds2_int, ds2_range)).all()


def test_axial_norm_peak_single_loc(datasets):
    """Tests that values of the data, as calculated by `axial_norm`,
    for two datasets has values no bigger than 1, and that a value at any
    position is equal to its value in the original dataset divided by its
    original maximum value.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose values are fractions of the maximum
    # value of the original dataset

    dataset1, dataset2 = [
        x.isel(t=0).sel(x=np.random.choice(x['x']),
                        y=np.random.choice(x['y']),) for x in datasets
    ]

    assert all([len(x['value'].shape) == 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='peak')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert nds1_vals.max() == 1
    assert nds2_vals.max() == 1

    assert all(nds1_vals == dataset1['value']/dataset1['value'].max())
    assert all(nds2_vals == dataset2['value']/dataset2['value'].max())


def test_axial_norm_peak_multiple_loc(datasets):
    """Tests that values of the data, as calculated by `axial_norm`,
    for two datasets has values no bigger than 1, and that a value at any
    position is equal to its value in the original dataset divided by its
    original maximum value.

    Multiple locations.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose values are fractions of the maximum
    # value of the original dataset

    dataset1, dataset2 = [
        x.isel(t=0) for x in datasets
    ]

    assert all([len(x['value'].shape) != 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='peak')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert nds1_vals.max() == 1
    assert nds2_vals.max() == 1

    assert (nds1_vals == dataset1['value']/dataset1['value'].max()).all()
    assert (nds2_vals == dataset2['value']/dataset2['value'].max()).all()


def test_axial_norm_relative_peak_single_loc(datasets):
    """Tests that values of the data, as calculated by `axial_norm`,
    for two datasets has values no bigger than 1, and that a value at any
    position is equal to its value in the original dataset divided by its
    original maximum value.

    Identical tests to `test_axial_norm_peak_single_loc`, with identical
    results, as this is at a single location.

    Single location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose values are fractions of the maximum
    # value of the original dataset

    dataset1, dataset2 = [
        x.isel(t=0).sel(x=np.random.choice(x['x']),
                        y=np.random.choice(x['y']),) for x in datasets
    ]

    assert all([len(x['value'].shape) == 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='relative-peak')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert nds1_vals.max() == 1
    assert nds2_vals.max() == 1

    assert all(nds1_vals == dataset1['value']/dataset1['value'].max())
    assert all(nds2_vals == dataset2['value']/dataset2['value'].max())


def test_axial_norm_relative_peak_multiple_loc(datasets):
    """Tests that values of the data, as calculated by `axial_norm`,
    for two datasets has values no bigger than 1, and that a value at any
    position is equal to its value in the original dataset divided by its
    original maximum value.

    Multiple location.
    """
    # GIVEN two :py:class:`xarray.Dataset`s
    # WHEN fed through the :py:func:`axial_norm` routine
    # THEN return two datasets whose values are fractions of the maximum
    # value of the original dataset

    dataset1, dataset2 = [
        x.isel(t=0) for x in datasets
    ]

    assert all([len(x['value'].shape) != 1 for x in (dataset1, dataset2)])

    normed_ds1, normed_ds2, *_ = axial_norm(dataset1, dataset2,
                                            datavar_name='value',
                                            dim='z',
                                            method='relative-peak')

    nds1_vals, nds2_vals = [x['value'] for x in (normed_ds1, normed_ds2)]

    assert (nds1_vals.max(dim='z') == 1).all()
    assert (nds2_vals.max(dim='z') == 1).all()

    assert (nds1_vals == dataset1['value']/dataset1['value'].max(dim='z')).all()
    assert (nds2_vals == dataset2['value']/dataset2['value'].max(dim='z')).all()