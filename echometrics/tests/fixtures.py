import numpy as np

simple_data_max = -50
simple_data_min = -100
data = {}

def make_simple_data():
    data['data'] = np.zeros((5, 7)) - simple_data_min
    data['data'][2, :] = simple_data_max
    data['depth'] = np.arange(5)
    data['index'] = np.arange(7)
    data['bad_data'] = np.zeros((5, 7))
    data['bad_data'][2, 2] = 1
