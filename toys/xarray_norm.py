import numpy as np

from scipy.optimize import minimize


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


def trim_datasets(datasets, bounds, dim='z'):
    return [ds.isel(**{dim: slice(*bd)}) for ds, bd in zip(datasets, bounds)]


def align(to_align, align_to, var='value', dim='z'):

    def min_chisq(shift, to_align, align_to, datavar='value', dim='z'):
        new_exp = to_align.assign(**{dim: to_align[dim] + shift})

        new_exp = new_exp.interp(**{dim: align_to[dim].values})

        return np.power(align_to[datavar] - new_exp[var], 2).sum()

    res = minimize(
        min_chisq, [0.], args=(to_align, align_to, var, dim),
        bounds=(np.array([-0.5, 0.5]) * np.ptp(to_align[dim].values),)
    )

    aligned = to_align.assign_coords(**{dim: to_align[dim] + res.x})

    return aligned, align_to, res
