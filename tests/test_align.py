import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, lognorm, chisquare

from toys.xarray_norm import align


def test_align():
    a = b = 0.5

    x = np.linspace(beta.ppf(0.01, a, b),
                    beta.ppf(0.99, a, b), 100)
    x_trim = x[2:-3] + 0.3

    y1 = beta.pdf(x, 0.5, 0.5) + 20 * lognorm.pdf(x * 2, 1)

    y2 = np.random.uniform(0, 0.5, (100,)) + y1
    y2 = y2[2:-3]

    new_x_trim = np.sort(
        np.random.uniform(x_trim[0], x_trim[-1], int(len(x_trim) / 2)))
    new_y2 = np.interp(new_x_trim, x_trim, y2)

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

    align(dsa, dsb)
    assert False
