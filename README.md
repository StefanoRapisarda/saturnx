
SaturnX
========

SaturnX is a Python software package to perform timing 
analysis of (not exclusively) X-ray data.

What SaturnX can do:

- read event FITS files (photon counts over time) from different X-ray observatories;
- read GTIs (Good Time Intervals) from event FITS files. GTIs can be "cleaned" (sorted and merged if overlapping) and filtered according to duration or any other user criteria;
- compute binned lightcurves from events and perform a large variety of operations on these products (linear and logarithmic time binning, normalization, bin by bin arithmetical operations, merging, splitting according to segment and/or GTIs, mean/variance/RMS computation, plotting);
- compute power spectra from binned lightcurves and perform standard timing analysis operations such us power spectra normalization, average, linear and logarithmic frequency binning, Poisson noise subtraction, fractional and absolute RMS computation in different frequency bands, and plotting;
- fit interactively power spectra with Lorentzian functions or user defined models. A GUI allows the user to "draw" Lorentzians on the power spectrum plot and adjust its parameters on the spot;
- perform **wavelet** analysis of binned lightcurves using default or user defined wavelets;
- compute timing analysis standard products (basic information about the target and observational conditions, count rate plots, hardness intensity diagram, energy spectra, average power spectra in different energy bands and per GTI). The timing standard products are arranged in bookmarked, well organized, PDF pages to serve as a reference for deeper data analysis;
- provide several handy utilities for analysing data from X-ray observatories.

Features to be implemented:

- CrossSpectrum (and CrossList) and BiSpectrum (and BiList) core classes;
- GUI to analyse data for all the available X-ray observatories, from data reduction to standard product creation and interactive fitting;

On SaturnX design
-----------------

SaturnX core consists of four main classes: Event (and EventList), Gti (and GtiList), Lightcurve (and LightcurveList), and PowerSpectrum (and PowerList). All the classes (except the Event class) are independent from the X-ray observatory providing the data. 
Each of the core classes is an extension of the pandas.DataFrame class, so that the user can make full use of all those pandas amazing features he/she is familiar with and, at the same time, access a long list of functionalities that I specifically implemented for each class. Furthermore, because of their "pandasian" nature, SaturnX core objects can be easily displayed in jupyter-notebooks.

Core classes also come with metadata used to keep track of each step of data reduction and to fetch, when needed, information about the original FITS file. This and other features are meant to provide to the user the possibility to easely check every step of data analysis. Data reduction control is a basic principle of SaturnX design.

Plotting methods of core classes allow quick visualization of science products. These methods are designed so that default SaturnX plots can be modified according to user own settings and needs (presentations, papers, reports, etc).

Where does SaturnX come from?
-----------------------------

SaturnX is the result of more than ten years of personal experience on X-ray data reduction and analysis. This journey started at the University of Amsterdam where I was using a legacy code written in Fortran, *xana*. This code allowed several generations of Amsterdam scientists to significantly improve our knowledge of X-ray binaries. Despite its successfull outcomes, the code was just terribly written, it had a lot of GOTO statements, pieces of code grafted by different users, and almost no comments or documentation. After my PhD, I invested quite some time learning python and from the ashes (and frustration) of my experience with the Amsterdam legacy code SaturnX was born. SaturnX started as a humble attempt to implement all the features that I personally wish I could use analysing X-ray time series, but I hope that other astronomers working on X-ray binaries will find its functionalities helpful for their research.

Installation and Testing
------------------------

Work in progress

Documentation
-------------

Work in Progress. I am currently writing the documentation using Sphinx.



