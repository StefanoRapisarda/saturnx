from pathlib import Path
from datetime import datetime
import os

from .constants import FieldTypes as FT

fields = {
    "Time Resolution":{'req':True,'type':FT.double},
    "Time Segment":{'req':True,'type':FT.double},
    "Frequency Resolution":{'req':True,'type':FT.double},
    "Nyquist Frequency":{'req':True,'type':FT.double},
    "Time bins":{'req':True,'type':FT.integer},
    "Frequency bins":{'req':True,'type':FT.integer},
    "Low energy":{'req':True,'type':FT.float},
    "High energy":{'req':True,'type':FT.float},
    "Timing modes":{'req':True,'type':FT.string_list},
    "Selected Energy bands":{'req':True,'type':FT.string_list},
    "Raw data dir":{'req':True,'type':FT.string},
    "Reduced data dir":{'req':True,'type':FT.string},
    "Event file extension":{'req':True,'type':FT.string},
    "Event file identifier":{'req':True,'type':FT.string},
    "Reduced products suffix":{'req':True,'type':FT.string},
    "Fix time and frequency bins":{'req':True,'type':FT.boolean},
    "Split events into GTI":{'req':True,'type':FT.boolean},
    "Compute Lightcurves":{'req':True,'type':FT.boolean},
    "Read Lightcurves":{'req':True,'type':FT.boolean},
    "Compute Power Spectra":{'req':True,'type':FT.boolean},
    "Read Lightcurves":{'req':True,'type':FT.boolean},
    "Override":{'req':True,'type':FT.boolean}
}