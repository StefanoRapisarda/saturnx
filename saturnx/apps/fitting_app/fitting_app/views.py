import tkinter as tk
from tkinter import ttk

from .widgets import (
    FileBox, PlotArea, GtiIndexBox, PoissonBox, RebinBox, 
    NormalizationBox, InputDirBox, FrequencyRangeBox, FitFunctionsBox,
    SaveLoadBox, FitParametersBox, FitInfoBox, ResidualPlotBox
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
    

class FitView(ttk.Frame):

    def __init__(self,*args,controller,**kwargs):
        super().__init__(*args,**kwargs)

        self._controller = controller

        self.grid_columnconfigure(0, weight=1)

        self._init_freq_panel()
        self._init_fitting_function_panel()
        self._init_fitting_parameters_panel()
        self._init_save_panel()

    def _init_freq_panel(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=0,sticky='we')  

        self._freq_range_box = FrequencyRangeBox(parent=frame)
        self._freq_range_box.grid(column=0,row=0,sticky='we')  

    def _init_fitting_function_panel(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=1,sticky='we')

        self._fit_function_box = FitFunctionsBox(parent=frame)
        self._fit_function_box.grid(column=0,row=0,sticky='we')

        # Init fitting functions
        menu = self._fit_function_box._fit_func_box
        menu["menu"].delete(0,'end')
        for func in self._controller._model_func_list:
            menu["menu"].add_command(
                label=func,
                command=lambda value=func:self._fit_function_box._fit_func.set(value)
            )
        
        self._fit_function_box._fit_func_listbox.bind(
            '<<ListboxSelect>>',
            self._controller._activate_draw_function)
        
        # Configuring buttons
        self._fit_function_box._add_button.configure(
            command=self._controller._add_func
        )
        self._fit_function_box._del_button.configure(
            command=self._controller._del_func
        )
        self._fit_function_box._fit_button.configure(
            command=self._controller._comp_fit
        )

    def _init_fitting_parameters_panel(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=2,sticky='we')

        self._fit_parameters_box = FitParametersBox(parent=frame)
        self._fit_parameters_box.grid(column=0,row=0,sticky='we') 

    def _init_save_panel(self):

        frame = ttk.Frame(self)
        frame.grid(column=0,row=3,sticky='we')

        self._save_box = SaveLoadBox(parent=frame)
        self._save_box.grid(column=0,row=0,sticky='we')

class FitResultView(ttk.Frame):

    def __init__(self,*args,controller,**kwargs):
        super().__init__(*args,**kwargs)

        self._controller = controller

        self.grid_columnconfigure(0, weight=1)

        self._residual_box = ResidualPlotBox(parent=self)
        self._residual_box.grid(column=0,row=0,sticky='we')

        self._stats_box = FitInfoBox(parent=self)
        self._stats_box.grid(column=0,row=1,sticky='we') 

if __name__ == '__main__':
    print('Launching app')
    app = tk.Tk()
    frame = tk.Frame(app)
    frame.pack()
    box = FitResultView(parent=frame)
    box.pack()
    app.mainloop()
        