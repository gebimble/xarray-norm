import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope='session')
def datasets():
    dataset1, dataset2 = [
        xr.Dataset.from_dataframe(
            pd.read_parquet(f'../SpatialET/data/gaussian_0'
                            f'{x}.parquet').set_index(['x', 'y', 'z', 't']))
        for x in (1, 2)
    ]

    return dataset1, dataset2
