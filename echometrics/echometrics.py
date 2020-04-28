import string
import numpy as np
from scipy.stats import mode
from scipy import io
import scipy.ndimage as ndi
import pandas
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print "Warning: could not import from matplotlib. Echogram.show() will not work."

class Echogram(object):
    '''
    echogram(data, depth, index, scale=decibel, threshold=[-80, 0], bad_data=None):)

    Representation of an echogram.  A wrapper class for a
    depth-time masked array.

    Parameters
    ----------
    data : array-like
        2-d array of backscatter values (an echogram).  Depth should be along
        the vertical axis, with time/distance on the horizontal one.
    index : array-like
        Array of indexing values (e.g. time, distance, lat/lon) locating the
        pings in the echogram.  May be any type.
    scale: string
        Indicates whether echogram is in log or linear units.
        Must be either "log" or "linear".
    threshold: [lower, upper]
        Defaults to [-80, 0].  Values above and below which a mask is
        applied to the data matrix.
    bad_data : bool array
        Array the same size as data, indicating which cells are marked as bad data.
        NOTE: This is not currently implemented and will be ignored in calculations.
    '''
    def __init__(self, data, depth, index, scale='decibel', threshold=[0, -110], bad_data=None):
        self.data = data
        self.depth = depth.flatten()
        self.index = index
        self.bad_data = bad_data
        self.scale = scale
        self.threshold = threshold
        # Make sure everything is the right size before proceeding
        self.__check_dimensions()
        self.dz = abs(self.depth[1] - self.depth[0])
        self.data = np.ma.masked_outside(self.data, threshold[0], threshold[1])

    def __check_dimensions(self):
        assert len(self.depth) == self.data.shape[0], \
            "Size of depth array does not match data."
        assert len(self.index) == self.data.shape[1], \
            "Size of index array does not match data."
        assert len(self.threshold) == 2, \
            "Threshold must be sequence of length 2."
        dz = np.diff(self.depth)
        assert all(np.round(dz, 7) == np.round(dz[0], 7)), \
            "Depth bins must be equally spaced."
        if self.bad_data is not None:
            assert self.bad_data.shape == self.data.shape, \
                "data and bad_data must have same shape."

    def show(self):
        '''
        Plot an image of the echogram.
        '''
        if plt is None:
            print "matplotlib was not imported, canot plot"
            return
        plt.imshow(self.data, aspect='auto')
        # make axes ticks be depth and time/location

    def set_scale(self, scale):
        '''
        Set the scale of the echogram (decibel or linear units).

        Parameters
        ----------
        scale : string
            Scale to which the echogram will be converted.  Must be either "dB" or
            "linear".  If the echogram is already in that form, nothing is changed.
        '''
        if scale != self.scale:
            if scale not in ['linear', 'decibel']:
                raise ValueError('Scale must be either "linear" or "decibel."')
            self.scale = scale
            if scale == 'linear':
                self.data = 10**(self.data / 10.)
                # TODO: set threshold scale
            elif scale == 'decibel':
                self.data = 10 * np.log10(self.data)

    def set_threshold(self, threshold=None):
        '''
        Set the threshold of the echogram, for display and metric calculations.

        Parameters
        ----------
        threshold : sequence
            Any length-2 sequence containing the upper and lower threshold. The
            order ([hi, lo] or [lo, hi]) does not matter.
        '''
        if len(threshold) != 2:
            raise ValueError("Threshold must be a sequence of length 2.")
        else:
            self.threshold = threshold
            self.data = np.ma.masked_outside(self.data,
                min(threshold), max(threshold))

    def flip(self):
        self.data = np.flipud(self.data)
        self.depth = np.flipud(self.depth)

    # def show(self, display_range=20, *args, **kwargs):
    #         '''
    #         Plots the echogram.
    #         '''
    #         tmin, tmax = min(self.datetimes), max(self.datetimes)
    #         zmin, zmax = min(self.z), max(self.z)
    #         display_threshold = min(self.threshold)
    #         ax = imshow(self.data, aspect='auto',
    #                     vmin=display_threshold,
    #                     vmax=display_threshold + display_range,
    #                     *args, **kwargs)
    #         locs, labels = xticks()
    #         #xticks(locs[np.logical_and(locs >= 0, locs < len(self.datetimes))],
    #         #        self.datetimes[::np.diff(locs)[0]])
    #         return ax

def read_flat(file, index, depth, value, sep=',', **kwargs):
    '''
    Create an echogram object from a text file in "flat" format--i.e., with
    one sample (interval and depth) per row.

    Parameters
    ----------
    file : string
        Name of file to read in.
    index : string, or list of strings
        Name or names of column(s) containing the indexing values.
    depth : string
        Name of the column containing the depth values
    value : string
        Name of the column
    sep : string
        Delimiter separating columns.  Default is a comma.
    **kwargs : additional arguments passed to Echogram()

    Returns
    -------
    An echometrics.Echogram object.
    '''
    data = pandas.read_table(file, sep=sep)
    idx = data[index]
    idx = [string.join([str(x) for x in idx.ix[i]]) for i in range(data.shape[0])]
    data['index'] = idx
    data_pivot = data.pivot(depth, 'index', value)
    return Echogram(np.array(data_pivot), np.array(data_pivot.index).astype('float'),
                    np.array(data_pivot.columns), **kwargs)

def read_mat(file, names, **kwargs):
    '''
    Create an echogram object from a .MAT file exported from Echoview.

    Parameters
    ----------
    file : string
        Name of file to read in.
    names : dictionary
        Dictionary matching the names of the fields in the .MAT file
        to the names expected by the Echogram constructor.  Must have entries
        for keys 'data', 'depth', and 'index'
    **kwargs
        Other names arguments to Echogram() (e.g. bad_data, scale, threshold).

    Returns
    -------
    An echometrics.Echogram object.
    '''
    # TODO: check `names` dict for having correct keys
    mat_data = io.loadmat(file)
    return Echogram(mat_data[names['data']].T, mat_data[names['depth']],
                    mat_data[names['index']], **kwargs)


def to_linear(x):
    '''
    Utility function to convert values in linear units to decibels (10 * log10(x))

    Parameters
    ----------
    x : array_like
        Input data.

    Examples
    --------
    to_linear(-10) --> 0.1
    to_linear(-50) --> 1.0e-5
    '''
    return 10**(x / 10.0)

def to_dB(x):
    '''
    Utility to covert linear values to decibels (10**(x / 10.))

    Parameters
    ----------
    x : array_like
        Input data
    Examples
    --------
    to_dB(0.1) --> -10.0
    to_db(1.0e-5) --> -50.0
    '''
    return 10 * np.log10(x)

def remove_outliers(echo, percentile, size):
    '''
    Masks all data values that fall above the given percentile value
    for the area around each pixel given by size.
    '''
    percentile_array = ndi.percentile_filter(echo.data, percentile, size=size)
    return np.ma.masked_where(echo.data > percentile_array, echo.data)


def depth_integral(echo, range=None, dB_result=True):
    '''
    Returns the depth-integrated Sv of the echogram (Sa) as a vector.

	Parameters
	----------
	echo : Echogram
		Input echogram object.
	range : tuple
		maximum, minimum depths over which to integrate.
	dB_result : bool
	    If true (the default), return result as backscattering strength
	    (i.e. in decibel form)
    '''
    if range is None:
        range = max(echo.depth), min(echo.depth)
    sample = np.logical_and(min(range) < echo.depth, echo.depth <= max(range))
    integral = np.sum(to_linear(echo.data[sample, :]), axis=0) * echo.dz
    if dB_result:
        integral = to_dB(integral)
    return integral

def sv_avg(echo, dB_result=True):
    '''
    Returns the depth-averaged volumetric backscatter from an echogram.

    Parameters
    ----------
    echo : echometrics.Echogram
        Input echogram object.
    dB_result : bool
	    If true (the default), return result as mean volume backscattering strength
	    (i.e. in decibel form)
    '''
    avg = np.mean(to_linear(echo.data), axis=0)
    if dB_result:
        avg = to_dB(avg)
    return avg

def center_of_mass(echo):
    '''
    Returns the center of mass of an Echogram (the expected value
    of depth in each ping weighted by the sv at each depth).

    Parameters
    ----------
    echo : ecometrics.Echogram
        Input Echogram object.
    '''
    linear = to_linear(echo.data)
    return np.ma.dot(linear.T, echo.depth) / linear.sum(axis=0)

def inertia(echo):
    '''
    Calulate the inertia (a measure of spread or dispersion around the center of
    mass) for an Echogram.

    Parameters
    ----------
    echo : ecometrics.Echogram
        Input Echogram object.
    '''
    depth_bins, time_bins = echo.data.shape
    depth_grid = np.tile(echo.depth, (time_bins, 1)).T
    diff = depth_grid - center_of_mass(echo).reshape(1, time_bins)
    linear = to_linear(echo.data)
    return np.sum((diff**2 * linear), axis=0) / (np.sum(linear, axis=0))

def proportion_occupied(echo):
    '''
    Returns the proportion of each ping that is above the echogram's threshold.

    Parameters
    ----------
    echo : ecometrics.Echogram
        Input Echogram object.
    '''
    num_bins = len(echo.depth)
    pa = np.sum(echo.data.mask == False, axis=0)
    return pa / float(num_bins)

def aggregation_index(echo):
    '''
    Calculate Bez and Rivoirard's (2002) Index of Aggregation.

    Parameters
    ----------
    echo : ecometrics.Echogram
        Input Echogram object.
    '''
    linear = to_linear(echo.data)
    return np.sum(linear**2, axis=0) * echo.dz / np.sum(linear*echo.dz, axis=0)**2

def equivalent_area(echo):
    '''
    Calculate the equivalent area for each ping in an echogram (the area that would
    be occupied if all cells had the same density.)

    Parameters
    ----------
    echo : ecometrics.Echogram
        Input Echogram object.
    '''
    linear = to_linear(echo.data)
    return np.sum(linear * echo.dz, axis=0)**2 / np.sum(linear**2, axis=0) * echo.dz


def layer_mask(echo, gauss_size=(19,19), med_size=(19, 19), dilate_size=(19, 19), slope_threshold=0.02, echo_threshold=-110):
    '''
    Detect layers on an echogram.
    Returns a boolean array the same size as echo.data
    '''
    def padded_diff(a):
        empty_row = np.zeros(a.shape[1]) + np.nan
        return np.vstack((empty_row, np.diff(a, axis=0)))

    old_threshold = echo.threshold
    echo.set_threshold([echo_threshold - 1, 0])
    zerodata = np.copy(echo.data.data)
    zerodata[echo.data.mask] = min(echo.threshold)
    smoothed = ndi.gaussian_filter(zerodata, gauss_size)
    d1 = padded_diff(smoothed)
    d2 = padded_diff(d1)
    level = ((np.absolute(d1) < slope_threshold) & (d2 < 0) &
          (zerodata > echo_threshold) & (echo.data.mask==False))
    level =  ndi.grey_dilation(ndi.median_filter(level, med_size), size=dilate_size)
    level = np.ma.array(level, mask=echo.bad_data)
    echo.set_threshold(old_threshold)
    return level

def nlayers(echo, layer=None, gauss_size=(19,19), med_size=(19, 19), dilate_size=(19, 19), slope_threshold=0.02, echo_threshold=-110):
    '''
    Counts the layers present in an echogram.

    Parameters
    ----------
    echo: Echogram object
    layer : 2-d array
        Boolean array the same size as echo.data, designating which parts of the
        echogram are considered to be in a layer.
    gauss_size, med_size, dilate_size : tuple
        Length-2 sequences giving the height and width of convolution filters
        used in the layer-detection algorithm.  All default to (19, 19).
    slope_threshold : float
        The threshold below which the first derivative of Sv with respect to depth
        is considered "flat" for the purposes of the detection algorithm.
        Defaults to 0.02.
    echo_threshold : float
        Acoustic threshold to use as the floor in the layer detection. Defaults
        to -80.
    Notes
    -----
    The returned array is set equal to nan where the entire water column is
    marked as bad data.

    Depending on the resolution of the data and the particular system, you may
    need to futz with the parameters to get good results.  For one (possibly
    excessive) approach to futzing, see:

    Urmy, S.S., Horne, J.K., and Barbee, D.H., 2012.  Measuring the vertical
    distributional variability of pelagic fauna in Monterey Bay. ICES Journal of
    Marine Science 69 (2): 184-196.

    As a general starting point, the convolution filters should be set to about
    half the width of the layer features you are interested in.  The slope
    threshold should in general be as small as it can before it starts missing
    layers.
    '''
    if layer is None:
        layer = layer_mask(echo, gauss_size, med_size, dilate_size, slope_threshold,
                            echo_threshold)
    layer = np.where(layer, 1., 0.)
    edges = np.nansum(abs(np.diff(layer, axis=0)), axis=0)
    edges[edges % 2 == 1] += 1
    edges[np.all(layer.mask, axis=0)] = np.nan
    return edges / 2

def calc_metrics(echo, metric_funcs):
    '''
    Applies multiple metric functions to an echogram, returning all values in an
    array with the same index as the echogram.

    Parameters
    ----------
    echo : Echogram object
        Echogram from which the metrics are calculated.
    metric_funcs : list
        List of functions to apply to the echogram.

    Returns
    -------
    A pandas DataFrame holding the metrics, with the same index as the echogram.

    Notes
    -----
    This function currently can't do either of the layer-detection functions.
    '''
    metrics = np.array([f(echo) for f in metric_funcs])
    metrics = pandas.DataFrame(metrics.T)
    metrics.index = echo.index
    metrics.columns = [f.func_name for f in metric_funcs]
    return metrics
