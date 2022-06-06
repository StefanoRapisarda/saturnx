import tkinter as tk
from tkinter import ttk

from .widgets import (
    FileBox, PlotArea, GtiIndexBox, PoissonBox, RebinBox, 
    NormalizationBox, InputDirBox
)   

class View(ttk.Frame):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._init_upper_panel_row1()
        self._init_upper_panel_row2()
        self._init_plot_area()
        self._init_input_dir_panel()

    def _init_upper_panel_row1(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=0,sticky='nswe')
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_columnconfigure(3, weight=1)
        frame.grid_columnconfigure(4, weight=1)

        self._file_box = FileBox(parent=frame)
        self._file_box.grid(column=0,row=0,sticky='nswe')

        self._gti_box = GtiIndexBox(parent=frame)
        self._gti_box.grid(column=1,row=0,sticky='nswe')
        self._gti_box._disable()

        self._plot_button = tk.Button(frame,text='PLOT')
        self._plot_button.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        
        self._fit_button = tk.Button(frame,text='FIT')
        self._fit_button.grid(column=3,row=0,padx=5,pady=5,sticky='nswe')

        self._reset_button = tk.Button(frame,text='RESET')
        self._reset_button.grid(column=4,row=0,padx=5,pady=5,sticky='nswe')  

    def _init_upper_panel_row2(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=1,sticky='we')
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

        self._poi_box = PoissonBox(parent=frame)
        self._poi_box.grid(column=0,row=0,sticky='we') 

        self._rebin_box = RebinBox(parent=frame)
        self._rebin_box.grid(column=1,row=0,sticky='we')

        self._norm_box = NormalizationBox(parent=frame)
        self._norm_box.grid(column=2,row=0,sticky='we')

    def _init_plot_area(self):

        self._plot_area = PlotArea(parent=self)
        self._plot_area.grid(column=0,row=2,sticky='nswe')

    def _init_input_dir_panel(self):

        self._input_dir_box = InputDirBox(parent=self)
        self._input_dir_box.grid(column=0,row=3,pady=5,sticky='nswe')
    

