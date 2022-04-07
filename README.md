
SaturnX
========

SaturnX is a Python software package to perform timing 
analysis of (not exclusively) X-ray data.

What SaturnX can do:

- read event FITS files (photon counts over time);
- read GTIs (Good Time Intervals) from event FITS files, GTI "cleaning" (sorting and removing overlapping intervals), filtering according to duration;
- filter events (photon counts) according to user settings and split events according to GTIs;
- create binned lightcurves from events and perform a large variety of operations on these products (linear and logarithmic time binning, normalization, bin by bin arithmetical operations, splitting according to segment and/or GTIs, mean/variance/RMS computation, plotting);
- create power spectra from binned lightcurves and perform standard timing analysis operations such us power spectra normalization, average, linear and logaritmic frequency binning, Poisson noise subtraction, fractional and absolute RMS computation in different frequency bands, plotting;
- fit interactively power spectra with Lorentzian functions or user defined models;
- perform **wavelet** analysis of binned lightcurves using default or user defined wavelets;
- compute timing analysis standard products (basic information about the target and observational conditions, count rate plots, hardness intensity diagram, energy spectra, average power spectra in different energy bands and per GTI) and arrange them in bookmarked well organized PDF pages.

SaturnX design "philosophy"
---------------------------

SaturnX core consists of four main classes: Event (and EventList), Gti (and GtiList), Lightcurve (and LightcurveList), and PowerSpectrum (and PowerList). Each of these classes is an extension of the pandas.DataFrame class, so that you can make full use of all those pandas amazing features you are familiar with and, at the same time, access a long list of functionalities that I specifically implemented for each class. Furthermore, because of their "pandasian" nature, SaturnX core objects can be easily displayed in jupyter-notebooks.
Each core class comes also with metadata used to keep track of each step of data reduction and to fetch, when needed, information about the original FITS file.
Plotting methods of core classes allow quick visualization and check of science products at each level of the data reduction. These methods are designed so that default SaturnX plots can be modified according to user own settings and needs (presentations, papers, reports, etc).

Installation and Testing
------------------------

Work in progress

Documentation
-------------

Work in Progress. I am currently writing the documentation using Sphinx.



