import pytest

import numpy as np
import xarray as xr

from scipy.stats import beta, lognorm

from toys import __version__
from toys.xarray_norm import radial_norm, axial_norm, align


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


def test_align_1d():
    a = b = 0.5
    offset = 0.3

    sl = slice(2, -3)

    x = np.linspace(beta.ppf(0.01, a, b),
                    beta.ppf(0.99, a, b), 100)
    x_trim = x[sl] + offset

    y1 = beta.pdf(x, 0.5, 0.5) + 20 * lognorm.pdf(x * 2, 1)

    def make_noisy_data(y1):
        y2 = np.random.uniform(0, 0.5, (100,)) + y1
        y2 = y2[sl]
        return y2

    y2 = make_noisy_data(y1)

    def make_new_garbled_data(x_trim, y2):
        new_x_trim = np.sort(
            np.random.uniform(x_trim[0], x_trim[-1], int(len(x_trim) / 2)))
        new_y2 = np.interp(new_x_trim, x_trim, y2)
        return new_x_trim, new_y2

    new_x_trim, new_y2 = make_new_garbled_data(x_trim, y2)

    dsa = xr.Dataset(
        data_vars=dict(value=(['x', 'y', 'z', 't'], new_y2.reshape(
            1, 1, new_y2.shape[0], 1))),
        coords=dict(x=(['x'], [0]), y=(['y'], [0]), t=(['t'], [0]),
                    z=(['z'], new_x_trim)))

    dsb = xr.Dataset(data_vars=dict(value=(['x', 'y', 'z', 't'], y1.reshape(
        1, 1, y1.shape[0], 1))),
                     coords=dict(x=(['x'], [0]), y=(['y'], [0]),
                                 t=(['t'], [0]),
                                 z=(['z'], x)))

    dsa, dsb, res = align(dsa, dsb)

    assert -res.x == pytest.approx(offset, rel=1e-1)


def test_align_2d():
    a = b = 0.5
    offset = 0.3

    sl = slice(2, -3)

    x = np.linspace(beta.ppf(0.01, a, b),
                    beta.ppf(0.99, a, b), 100)
    x_trim = x[sl] + offset

    y1 = beta.pdf(x, 0.5, 0.5) + 20 * lognorm.pdf(x * 2, 1)

    def make_noisy_data(y1):
        y2 = np.random.uniform(0, 0.5, (100,)) + y1
        y2 = y2[sl]
        return y2

    y21 = make_noisy_data(y1)
    y22 = make_noisy_data(y1)

    def make_new_xtrim(x_trim):
        return np.sort(np.random.uniform(x_trim[0], x_trim[-1], int(len(x_trim) / 2)))

    def make_new_ytrim(new_x_trim, x_trim, y2):
        return np.interp(new_x_trim, x_trim, y2)

    new_x_trim = make_new_xtrim(x_trim)

    new_y21 = make_new_ytrim(new_x_trim, x_trim, y21)
    new_y22 = make_new_ytrim(new_x_trim, x_trim, y22)

    dsa = xr.Dataset(
        data_vars=dict(
            value=(['x', 'y', 'z', 't'],
                   np.vstack([new_y21, new_y22]).reshape(2, 1, new_y21.shape[0], 1))
        ),
        coords=dict(x=(['x'], [-.5, .5]),
                    y=(['y'], [0]),
                    t=(['t'], [0]),
                    z=(['z'], new_x_trim)))

    dsb = xr.Dataset(
        data_vars=dict(
            value=(['x', 'y', 'z', 't'],
                   np.vstack([y21, y22]).reshape(2, 1, y21.shape[0], 1))
        ),
        coords=dict(x=(['x'], [-.5, .5]),
                    y=(['y'], [0]),
                    t=(['t'], [0]),
                    z=(['z'], x[sl])))

    dsa, dsb, res = align(dsa, dsb)

    assert -res.x == pytest.approx(offset, rel=1e-1)


def test_align_3d():
    a = b = 0.5
    offset = 0.3

    sl = slice(2, -3)

    x = np.linspace(beta.ppf(0.01, a, b),
                    beta.ppf(0.99, a, b), 100)
    x_trim = x[sl] + offset

    y1 = beta.pdf(x, 0.5, 0.5) + 20 * lognorm.pdf(x * 2, 1)

    def make_noisy_data(y1):
        y2 = np.random.uniform(0, 0.5, (100,)) + y1
        y2 = y2[sl]
        return y2

    y21 = make_noisy_data(y1)
    y22 = make_noisy_data(y1)
    y23 = make_noisy_data(y1)
    y24 = make_noisy_data(y1)

    def make_new_xtrim(x_trim):
        return np.sort(np.random.uniform(x_trim[0], x_trim[-1], int(len(x_trim) / 2)))

    def make_new_ytrim(new_x_trim, x_trim, y2):
        return np.interp(new_x_trim, x_trim, y2)

    new_x_trim = make_new_xtrim(x_trim)

    new_y21 = make_new_ytrim(new_x_trim, x_trim, y21)
    new_y22 = make_new_ytrim(new_x_trim, x_trim, y22)
    new_y23 = make_new_ytrim(new_x_trim, x_trim, y23)
    new_y24 = make_new_ytrim(new_x_trim, x_trim, y24)

    dsa = xr.Dataset(
        data_vars=dict(
            value=(['x', 'y', 'z', 't'],
                   np.vstack([new_y21,
                              new_y22,
                              new_y23,
                              new_y24]).reshape(2, 2, new_y21.shape[0], 1))
        ),
        coords=dict(x=(['x'], [-.5, .5]),
                    y=(['y'], [-.5, .5]),
                    t=(['t'], [0]),
                    z=(['z'], new_x_trim)))

    dsb = xr.Dataset(
        data_vars=dict(
            value=(['x', 'y', 'z', 't'],
                   np.vstack([y21,
                              y22,
                              y23,
                              y24]).reshape(2, 2, y21.shape[0], 1))
        ),
        coords=dict(x=(['x'], [-.5, .5]),
                    y=(['y'], [-.5, .5]),
                    t=(['t'], [0]),
                    z=(['z'], x[sl])))

    dsa, dsb, res = align(dsa, dsb)

    assert -res.x == pytest.approx(offset, rel=1e-1)
