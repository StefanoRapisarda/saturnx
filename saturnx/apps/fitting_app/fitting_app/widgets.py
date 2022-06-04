import tkinter as tk
from tkinter import ttk

import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PlotArea(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_canvas()
        self._init_coor_panel()

    def _init_canvas(self):

        self._fig = Figure(figsize=(5,3.5),dpi=150)
        self._ax  = self._fig.add_subplot()

        self._canvas = FigureCanvasTkAgg(self._fig, master = self)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(column=0,row=0,sticky='nswe')

    def _init_coor_panel(self):

        coor_frame = ttk.Frame(self)
        coor_frame.grid(column=0,row=1,pady=5,sticky='nswe')
        coor_frame.grid_columnconfigure(0,weight=1)
        coor_frame.grid_columnconfigure(1,weight=1)
        coor_frame.grid_columnconfigure(2,weight=1)
        coor_frame.grid_columnconfigure(3,weight=1)

        labelx = ttk.Label(coor_frame,text='x coor: ')
        labelx.grid(column=0,row=0,pady=5,padx=5,sticky='nswe')
        self._x_pos = ttk.Label(coor_frame,text=' ')
        self._x_pos.grid(column=1,row=0,pady=5,padx=5,sticky='nswe')
        labely = ttk.Label(coor_frame,text='y coor: ')
        labely.grid(column=2,row=0,pady=5,padx=5,sticky='nswe')
        self._y_pos = ttk.Label(coor_frame,text=' ')
        self._y_pos.grid(column=3,row=0,pady=5,padx=5,sticky='nswe') 


class FileBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):

        box = ttk.LabelFrame(self,text='File')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        box.tag = 1

        self._file = tk.StringVar() # <--- PowerSpectrum or PowerList file
        self._file.set(' ') 

        self._file_menu = ttk.OptionMenu(box, self._file)
        self._file_menu.config(width=30)
        self._file_menu.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

class GtiIndexBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):
        box = ttk.LabelFrame(self,text='GTI selection')  
        box.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        # This will be initialized whhen self._file is changed
        self._gti_sel_string = tk.StringVar()
        self._gti_sel_string.set('')
        self._gti_entry = tk.Entry(box, textvariable=self._gti_sel_string,width=13)
        self._gti_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._gti_entry.to_change = 1

class PoissonBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):

        box = ttk.LabelFrame(self, text='Poisson level')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        # Frequeny boundary 
        self._low_freq = tk.DoubleVar()
        self._low_freq.set(3000)
        freq_label1 = tk.Label(box,text='>')
        freq_label1.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        freq_entry = tk.Entry(box,textvariable=self._low_freq,
            width=5)
        freq_entry.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        freq_label2 = tk.Label(box,text='Hz')
        freq_label2.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')

        # Poisson level
        self._poi_level = tk.DoubleVar()
        self._poi_level.set(0)
        self._est_poi_button = tk.Button(box,text='Estimate')
        self._est_poi_button.grid(column=3,row=0,padx=5,pady=5,sticky='nswe')
        self._poi_level_entry = tk.Entry(box,textvariable=self._poi_level,width=5)
        self._poi_level_entry.grid(column=4,row=0,padx=5,pady=5,sticky='nswe')

        self._sub_poi_button = tk.Button(box,text='Subtract Poi')
        self._sub_poi_button.grid(column=0,row=1,columnspan=5,padx=5,pady=5,sticky='nswe')
        self._sub_poi_button.config(state='disabled')

class RebinBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):

        box  = ttk.LabelFrame(self, text='Rebinning')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')  

        self._rebin_factor = tk.StringVar()
        self._rebin_factor.set(-30)
        self._rebin_entry = tk.Entry(box,textvariable=self._rebin_factor,width=9)
        self._rebin_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._rebin_button = tk.Button(box,text='Rebin')
        self._rebin_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')


class NormalizationBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):    

        box = ttk.LabelFrame(self, text='Normalization')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='nswe') 

        self._xy_flag = tk.IntVar()
        xy_checkbox = tk.Checkbutton(box,text='X Vs. XY',var=self._xy_flag)
        xy_checkbox.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        self._norm = tk.StringVar()
        self._norm.set('Leahy')
        norm_menu = ttk.OptionMenu(box,self._norm,*tuple(['','None','Leahy','RMS']))
        norm_menu.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')

        bkg_label = tk.Label(box,text='BKG [c/s]')
        bkg_label.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._bkg = tk.DoubleVar()
        self._bkg.set(0)
        bkg_entry = tk.Entry(box,textvariable=self._bkg,width=10)
        bkg_entry.grid(column=1,row=1,padx=5,pady=5,sticky='nswe')

class InputDirBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 

        self._input_dir = tk.StringVar()
        self._dir_entry = tk.Entry(self,textvariable=self._input_dir)
        self._dir_entry.grid(column=0,row=0,sticky='nswe')
        self._set_dir_button = ttk.Button(self,text='SET')
        self._set_dir_button.grid(column=1,row=0,sticky='nswe')

class InfoBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 
        pass
