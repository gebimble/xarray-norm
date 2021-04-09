import numpy as np
import pandas as pd
import xarray as xr

from scipy.optimize import minimize
from scipy.stats import chisquare

import matplotlib.pyplot as plt


def radial_norm(ds1, ds2, datavar_name):
    ds1_sum, ds2_sum = [x[datavar_name].sum() for x in (ds1, ds2)]
    ratio = ds1_sum / ds2_sum

    return ds1, ds2.assign(**{datavar_name: ratio * ds2[datavar_name]}), ratio


def axial_norm(ds1, ds2, datavar_name, dim, method='relative-integral'):

    if method == 'relative-integral':
        ds1_norm, ds2_norm = [
            x[datavar_name].integrate(dim) / np.ptp(x[dim].values)
            for x in (ds1, ds2)
        ]

    if method == 'peak':
        ds1_norm, ds2_norm = [
            x[datavar_name].max() for x in (ds1, ds2)
        ]

    if method == 'max-integral':
        NotImplemented


    ratio = ds1_norm / ds2_norm

    new_ds1 = ds1.assign(**{datavar_name: ds1[datavar_name] / ds1_norm})
    new_ds2 = ds2.assign(**{datavar_name: ds2[datavar_name] / ds2_norm})

    return new_ds1, new_ds2, ds1_norm, ds2_norm, ratio

