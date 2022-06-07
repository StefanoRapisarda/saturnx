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
        self._fig.tight_layout()
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

        labelx = ttk.Label(coor_frame,text='x coor: ',width=10)
        labelx.grid(column=0,row=0,pady=5,padx=5,sticky='nswe')
        self._x_pos = ttk.Label(coor_frame,text=' ',width=10)
        self._x_pos.grid(column=1,row=0,pady=5,padx=5,sticky='nswe')
        labely = ttk.Label(coor_frame,text='y coor: ',width=10)
        labely.grid(column=2,row=0,pady=5,padx=5,sticky='nswe')
        self._y_pos = ttk.Label(coor_frame,text=' ',width=10)
        self._y_pos.grid(column=3,row=0,pady=5,padx=5,sticky='nswe') 


class FileBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):

        self._box = ttk.LabelFrame(self,text='File')
        self._box.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        self._file = tk.StringVar() # <--- PowerSpectrum or PowerList file
        self._file.set(' ') 

        self._file_menu = ttk.OptionMenu(self._box, self._file)
        self._file_menu.config(width=30)
        self._file_menu.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

    def _reset(self):

        self._box.configure(text='File')
        self._file_menu['menu'].delete(0,'end')
        self._file.set(' ')
        

class GtiIndexBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self._parent = parent

        self._init_box()

    def _init_box(self):
        self._box = ttk.LabelFrame(self,text='GTI selection')  
        self._box.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        # This will be initialized whhen self._file is changed
        self._gti_sel_string = tk.StringVar()
        self._gti_sel_string.set('')
        self._gti_entry = tk.Entry(self._box, textvariable=self._gti_sel_string,width=13)
        self._gti_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

    def _enable(self):
        for child in self._box.winfo_children():
            child.configure(state='normal')

    def _disable(self):
        for child in self._box.winfo_children():
            child.configure(state='disable')        


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

        self._poi_flag = tk.IntVar()
        checkbox = tk.Checkbutton(box,text='Subtract',var=self._poi_flag)
        checkbox.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')     


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
        norm_menu = ttk.OptionMenu(box,self._norm,*tuple(['','None ','Leahy','RMS  ']))
        norm_menu.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')

        bkg_label = tk.Label(box,text='BKG [c/s]')
        bkg_label.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        self._bkg = tk.DoubleVar()
        self._bkg.set(0)
        bkg_entry = tk.Entry(box,textvariable=self._bkg,width=10)
        bkg_entry.grid(column=3,row=0,padx=5,pady=5,sticky='nswe')

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


class FrequencyRangeBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 

        box = ttk.LabelFrame(self,text='Frequency range')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='we')

        self._freq_range = tk.StringVar()
        self._freq_range.set('0-100')
        freq_range_entry = tk.Entry(box,textvar=self._freq_range, width=20)
        freq_range_entry.grid(column=0,row=0,padx=5,pady=5,sticky='we')   

        freq_label = tk.Label(box,text='Hz')
        freq_label.grid(column=1,row=0,padx=5,pady=5,sticky='we')


class FitFunctionsBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 

        box = ttk.LabelFrame(self, text='Fitting functions')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='we') 

        # Left column
        # --------------------------------------------------------------
        left_col = ttk.Frame(box)
        left_col.grid(column=0,row=0,sticky='nswe')

        self._fit_func_listbox = tk.Listbox(
            left_col,selectmode='multiple',height=12)
        self._fit_func_listbox.grid(
            column=0,row=0,padx=5,pady=5,sticky='nsew')   
        # --------------------------------------------------------------  

        # Right column
        # --------------------------------------------------------------
        right_col = ttk.Frame(box)
        right_col.grid(column=1,row=0,sticky='nswe')

        # Fitting function menu
        self._fit_func = tk.StringVar()
        fit_func_box = ttk.OptionMenu(right_col,self._fit_func)
        fit_func_box.grid(column=0,row=0, columnspan=2,\
            sticky='we',padx=5,pady=5)

        # Add and delete buttons
        add_button = ttk.Button(right_col, text='ADD')
        add_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        del_button = ttk.Button(right_col, text='DEL')
        del_button.grid(column=0,row=2,padx=5,pady=5,sticky='we') 

        # Fit and clear button
        fit_button = ttk.Button(right_col, text='FIT')
        fit_button.grid(column=0,row=3,padx=5,pady=5,sticky='nswe')        
        clear_button = ttk.Button(right_col, text='RESET')
        clear_button.grid(column=0,row=4,padx=5,pady=5,sticky='nsew')      
        # --------------------------------------------------------------


class ComputeErrorBox(ttk.Frame):
    pass

class SaveLoadBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 

        box = ttk.LabelFrame(self,text='Fit Output name')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='we')

        self._output_name = tk.StringVar()
        name_entry = tk.Entry(box,textvariable=self._output_name)
        name_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        save_button = ttk.Button(box, text='SAVE')
        save_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')        
        load_button = ttk.Button(box, text='LOAD')
        load_button.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')       


class FitParametersBox(ttk.Frame):

    def __init__(self,parent,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_columnconfigure(0,weight=1)

        self._parent = parent

        self._init_box()

    def _init_box(self): 

        box = ttk.LabelFrame(self,text='Fitting Parameters')
        box.grid(column=0,row=0,padx=5,pady=5,sticky='we')

        row1 = ttk.Frame(box)
        row1.grid(column=0,row=0,padx=5,pady=5,sticky='we')
        row2 = ttk.Frame(box)
        row2.grid(column=0,row=1,padx=5,pady=5,sticky='we')
        row3 = ttk.Frame(box)
        row3.grid(column=0,row=2,padx=5,pady=5,sticky='we')

        # First row
        self._fit_pars_listbox = tk.Listbox(row1, selectmode='multiple',width=25)
        self._fit_pars_listbox.grid(column=0,row=0,padx=5,pady=5,sticky='ew')     

        # Second row
        self._par_val = tk.StringVar()
        reset_entry = tk.Entry(row2, textvariable=self._par_val,width=10)
        reset_entry.grid(column=1,row=0,padx=5,pady=5,sticky='nsew')

        label = ttk.Label(row2,text='Par value')
        label.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')  

        # Third row
        self._reset_button = ttk.Button(row3, text='SET')
        self._reset_button.grid(column=0,row=0,padx=5,pady=5,sticky='nsew')        

        self._freeze = ttk.Button(row3, text='FREEZE')
        self._freeze.grid(column=1,row=0,padx=5,pady=5,sticky='senw')
        
        self._free = tk.Button(row3, text='FREE')
        self._free.grid(column=2,row=0,padx=5,pady=5,sticky='senw')


if __name__ == '__main__':
    print('Launching app')
    app = tk.Tk()
    frame = tk.Frame(app)
    frame.pack()
    box = FitFunctionsBox(parent=frame)
    box.pack()
    app.mainloop()
        

