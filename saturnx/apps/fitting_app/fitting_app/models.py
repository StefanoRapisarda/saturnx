import pathlib
import numpy as np

class PlotFields:
    def __init__(self):
        self._data_dir = '/Volumes/BigBoy/NICER_data/MAXI_J1820+070/analysis/qpoCB_transition'
        self._file = None
        self._sel_gti = ''
        self._n_segs = None,
        self._poi_low_freq = 0
        self._poi_level = 0
        self._rebin_str = ''
        self._power_norm = ''
        self._bkg = 0
    

class FitFields:
    def __init__(self):
        self._freq_range = [0,np.inf]
        self._fitting_funcs = {}
        self._fit_result = None