===========
EchoMetrics
===========

This python package implements a set of metrics intended to concisely describe the vertical distribution of acoustic backscatter in the water column.  Given an echogram (a 2-D array of acoustic backscatter values indexed by depth on the vertical axis and time and/or location on the horizontal), these metrics will measure various characteristics of the vertical distribution of backscatter (i.e., animal density) ping-by-ping.

The original version of this code was written to help analyze a large set of acoustic data collected from an ocean observing system in Monterey Bay, CA.  A more detailed description of the algorithms can be found in:
 
Urmy, S.S., Horne, J.K., and Barbee, D.H., 2012.  Measuring the vertical distributional variability of pelagic fauna in Monterey Bay. ICES Journal of Marine Science 69 (2): 184-196.  http://icesjms.oxfordjournals.org/cgi/content/full/fsr205?ijkey=tSU0noNUWz4bj57&keytype=ref

Installation
============
Depends
-------
This package depends on a few other Python packages, which can all be found on PyPi:

* NumPy and SciPy, for array operations

* Pandas, for data-reshaping purposes

* Matplotlib for plotting.  This is not strictly necessary, but the "show" method of the Echogram class won't work without it.

Installing
----------
Download the package, then extract it and ``cd`` into the directory:

    tar -xf EchoMetrics-x.x.x.tar.gz
    cd EchoMetrics-x.x.x

then install it manually using distutils:

    python setup.py install

Metrics
=======

Eight metric functions are included in the package at this time.  They are:

* ``sv_avg``: The (linear) average value of volumetric backscatter in the water column.

* ``depth_integral``: Summation of backscatter over the whole water column or a specified depth range.

* ``center_of_mass``: The mean location of backscatter in the water column (the centroid)

* ``inertia``: A measure of spatial dispersion around the center of mass.

* ``aggregation_index``: Measures the degree of aggregation.  Higher when high densities are concentrated in small areas.

* ``equivalent_area``: Measure of evenness (it is the reciprocal of the aggregation index).  Represents the area of the water column that would be occupied if all cells had the mean density.

* ``proportion_occupied``: The proportion of the water column with density above the echogram's threshold value.

* ``nlayers``: Number of distinct scattering layers detected by an image-analysis algorithm.

Usage
=====

An example of usage is as follows::

    import echometrics
    import matplotlib.pyplot as plt
    
    filename = "path/to/data/My_EchoView_export.csv"
    index = ["Lon_M", "Lat_M"]
    depth = "Depth_M"
    value = "Sv_mean"
    echo = echometrics.read_flat(fileame, index, depth, value)
    echo.show()
    cm = echometrics.center_of_mass(echo)
    plt.figure()
    plt.plot(cm)
    plt.show()


