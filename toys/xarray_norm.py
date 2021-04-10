from functools import reduce

import numpy as np

from scipy.optimize import minimize
from scipy.stats import chisquare

import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt


def radial_norm(ds1, ds2, datavar_name):
    ds1_sum, ds2_sum = [x[datavar_name].sum() for x in (ds1, ds2)]
    ratio = ds1_sum / ds2_sum

    return ds1, ds2.assign(**{datavar_name: ratio * ds2[datavar_name]}), ratio


def axial_norm(ds1, ds2,
               datavar_name='value', dim='z',
               method='relative-integral'):
    if method == 'relative-integral':
        ds1_norm, ds2_norm = [
            x[datavar_name].integrate(dim) / np.ptp(x[dim].values)
            for x in (ds1, ds2)
        ]

    if method == 'peak-integral':
        ds1_norm, ds2_norm = [
            x[datavar_name].integrate(dim).max() / np.ptp(x[dim].values)
            for x in (ds1, ds2)
        ]

    if method == 'relative-peak':
        ds1_norm, ds2_norm = [
            x[datavar_name].max(dim=dim) for x in (ds1, ds2)
        ]

    if method == 'peak':
        ds1_norm, ds2_norm = [
            x[datavar_name].max() for x in (ds1, ds2)
        ]

    ratio = ds1_norm / ds2_norm

    new_ds1 = ds1.assign(**{datavar_name: ds1[datavar_name] / ds1_norm})
    new_ds2 = ds2.assign(**{datavar_name: ds2[datavar_name] / ds2_norm})

    return new_ds1, new_ds2, ds1_norm, ds2_norm, ratio


def align(exp, theory,
          exp_trim=[None, None], theory_trim=[None, None],
          datavar_name='value', dim='z', method='relative-integral'):

    exp, theory = [ds.isel(**{dim: slice(*sl)})
                   for ds, sl in zip((exp, theory),
                                     (exp_trim, theory_trim))]

    theory = theory.interp(exp.drop(['t', dim]).coords)  # get theory at the
                                                         # exp locations

    # reset exp heights by colocating the bottom of their z-axis
    # exp = exp.assign(
    #     **{
    #         dim: exp[dim] - (exp[dim][0] - theory[dim][0])
    #     }
    # )

    def min_chisq(ht, exp, th, datavar_name='value', dim='z'):
        new_exp = exp.assign(
            **{dim: exp[dim] + ht}
        )

        new_exp = new_exp.interp_like(th).where(np.isfinite(th[datavar_name]))

        new_exp = reduce(lambda y, x: y.dropna(x, how='all'),
                         ('x', 'y', 'z'),
                         new_exp[datavar_name])
        th = reduce(lambda y, x: y.dropna(x, how='all'),
                    ('x', 'y', 'z'),
                    th[datavar_name])

        th = th.sel(z=new_exp['z'].values)

        stack_dict = {'xy': ('x', 'y')}

        chi_results = chisquare(new_exp.stack(stack_dict),
                                th.stack(stack_dict),
                                axis=0)[0]


        chi_res_sum = chi_results[np.isfinite(chi_results)].sum()
        print(chi_res_sum)
        return chi_res_sum


    res = minimize(min_chisq, [0.], args=(exp, theory, datavar_name, dim),
                   bounds=(np.array([-0.25, 0.25])*np.ptp(exp[dim].values),))

    return res
