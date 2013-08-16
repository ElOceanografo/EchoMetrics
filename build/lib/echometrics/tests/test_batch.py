import numpy as np
from echometrics import *
from fixtures import data, make_simple_data
from nose import with_setup

def test_calc_metrics():
    file = 'data/Sept17_DVM_int_cope38.csv'
    echo = read_flat(file, ['Lat_M', 'Lon_M'], 'Layer', 'Sv_mean')
    metrics_list = [depth_integral, sv_avg, center_of_mass, proportion_occupied,
                    aggregation_index, equivalent_area]
    metrics = calc_metrics(echo, metrics_list)
    assert type(metrics) == pandas.DataFrame
    assert all(metrics.index == echo.index)
    assert metrics.shape[0] == len(echo.index)
    assert len(metrics.columns) == len(metrics_list)