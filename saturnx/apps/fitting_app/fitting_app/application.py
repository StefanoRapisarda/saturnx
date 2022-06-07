import tkinter as tk
from tkinter import ttk

from .views import View, FitView
from .models import PlotFields, FitFields
from .controllers import Controller

class FitApp(tk.Tk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._model = {
            'plot_fields' : PlotFields(),
            'fit_fields' : FitFields()
        }

        self.title('Power spectrum fitting application')
        self.columnconfigure(0, weight = 1)

        ttk.Label(
            self,
            text = 'Power Spectrum Fitting Application',
            font = ("TkDefaultFont",16)
            ).grid(row=0)

        self._view = View()
        self._view.grid(row=1, padx=10, sticky=(tk.W + tk.E))

        self._cotroller = Controller(model=self._model, view=self._view)


class FitAppTest(tk.Tk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._view = FitView()
        self._view.grid(row=0, padx=10, sticky=(tk.W + tk.E))

