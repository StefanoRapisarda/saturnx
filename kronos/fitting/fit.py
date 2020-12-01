import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Menu

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np 

import pickle

import os
import sys
sys.path.append('/Volumes/Samsung_T5/software/')
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from timing.utilities import *
from fitting_functions import lorentzian
sys.setrecursionlimit(10000)
import timing as tg


class PlotWindow:

    def __init__(self,parent):

        self.parent = parent
        self.parent.title('Plot Window')
        self.fig = Figure(figsize=(10,5))

        self.data = None

        self._set_main_frame()
        self._set_canvas()
        self._set_menu()

    def _set_main_frame(self):
        self.main_frame = tk.Frame(self.parent,width=1100,height=550)
        self.main_frame.grid(column=0,row=1,padx=5,pady=5,sticky='NSEW')
    
    def _set_canvas(self, fig=None):
        self.canvas = FigureCanvasTkAgg(self.fig,master = self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0,row=0,sticky='NSWE')

    def _set_menu(self):

        menu_bar = Menu(self.parent)
        self.parent.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        window_menu = Menu(menu_bar, tearoff=0)

        file_menu.add_command(label='Load Power',command=self._load_power)
        file_menu.add_command(label='Quit',command=quit)

        window_menu.add_command(label='Data Info',command=self._new_window(DataInfo))
        window_menu.add_command(label='')       
            
        menu_bar.add_cascade(label='File', menu=file_menu) 
        menu_bar.add_cascade(label='Windows', menu=window_menu)

    def _plot_data(self):

        if self.data['name'] == 'power': self._plot_power()

    def _plot_power(self):
        self.fig = Figure(figsize=(10,5))
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        ax1.set_xlabel('Frequency [Hz]')
        ax2.set_xlabel('Frequency [Hz]')
        if self.data['norm'] == 'rms':
            ax1.set_ylabel('Power [(rms/mean)$^2$]')
            ax2.set_ylabel('Power [(rms/mean)$^2$/Hz]')
        elif self.data['norm'] == 'leahy':
            ax1.set_ylabel('Leahy Power')
            ax2.set_ylabel('Leahy Power')
        else:
            ax1.set_ylabel('Power')  
            ax2.set_ylabel('Power') 

        if not self.data['yerr'] is None:
            ax1.errorbar(self.x_to_plot,
                         self.x_to_plot*self.y_to_plot,
                         self.x_to_plot*self.yerr_to_plot,
                         fmt='-k')
            ax2.errorbar(self.x_to_plot,
                         self.y_to_plot,
                         self.yerr_to_plot,
                         fmt='-k')   
        else:        
            ax1.plot(self.x_to_plot,self.x_to_plot*self.y_to_plot,'-k')
            ax2.plot(self.x_to_plot,self.y_to_plot,'-k')  

        ax1.set_xscale('log') 
        ax1.set_yscale('log')
        ax1.grid()            

        ax2.set_xscale('log') 
        ax2.set_yscale('log')
        ax2.grid()    

        self._set_canvas()   

    def _load_power(self):
        '''
        Loads a power spectrum object, storing the unpbinned arrays into class
        attrbutes and initializing the self.data keys corresponding to binned
        data
        '''
        filename = filedialog.askopenfilename(initialdir=os.getcwd(),
        title = 'Select a power spectrum')

        data = {}

        # I need data to be plotted and fitted
        power = tg.PowerSpectrum.load(filename)
        x = power.freq.to_numpy()
        y = power.power.to_numpy()

        data['x'] = x[x>0]
        data['y'] = y[x>0]
        if 'spower' in power.columns:
            yerr = power.spower.to_numpy()
            data['yerr'] = yerr[x>0]
        else:
            data['yerr']= None

        # I need the normalization to write the plot labels accordingly
        if not power.rms_norm is None: 
            data['norm'] = 'rms'
        elif not power.leahy_norm is None: 
            data['norm'] = 'leahy'
        elif (power.rms_norm is None) and (power.leahy_norm is None):
            data['norm'] = None
        data['history'] = power.history
        data['name'] = 'power'

        self.data = data
        self._init_plot_data()
        self._plot_data()

    def _init_plot_data(self):
        self.x_to_plot = self.data['x']
        self.y_to_plot = self.data['y']
        self.yerr_to_plot = self.data['yerr']
        self._plot_data()


class Fit:

    def __init__(self):

        self.root = tk.Tk()
        PlotWindow(self.root)

if __name__ == '__main__':
    window = Fit()
    window.root.mainloop()

