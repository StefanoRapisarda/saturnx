import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from kronos.core.power import PowerSpectrum,PowerList

# Fitting data
import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult

import random
import uuid

import os
import pandas as pd

from tkinter import ttk
import tkinter as tk
from tkinter import filedialog

import glob

from functools import partial

__all__ = ['FittingTab']

class FittingTab:
    def __init__(self, parent, controller, fit_window):
        self._parent = parent
        self._controller = controller
        self._fit_window = fit_window

        # When self._to_plot is None, nothing has been plotted yet
        self._to_plot = None
        self._first_plot = True
        self._leahy = None

        # Frames
        # -------------------------------------------------------------
        self._frame1 = ttk.LabelFrame(self._parent,text='Plot buttons',
            style=self._controller._head_style)
        self._frame1.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')

        self._frame2 = ttk.LabelFrame(self._parent,text='To Fit',
            style=self._controller._head_style)
        self._frame2.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')

        #self._frame3 = ttk.LabelFrame(self._parent,text='Comments',
        #    style=self._controller._head_style)
        #self._frame3.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        # -------------------------------------------------------------

        # Populating frames
        # -------------------------------------------------------------
        self._init_buttons(self._frame1)
        self._init_plot_area(self._frame2)
        # -------------------------------------------------------------

    def _init_buttons(self,frame):

        row1 = tk.Frame(frame)
        row1.grid(column=0,row=0,sticky='nswe')
        row2 = tk.Frame(frame)
        row2.grid(column=0,row=1,sticky='nswe')

        # File scrollbox
        box11 = ttk.LabelFrame(row1,text='File')
        box11.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        box11.tag = 1
        self._file = tk.StringVar()
        self._file.set(' ')
        # The number of GTIs should be updated depending on the
        # selected file
        self._file.trace_add('write',self._update_label)
        if len(self._controller._pickle_files) != 0:
            self._file.set(self._controller._pickle_files[0])
        self._file_menu = ttk.OptionMenu(box11, self._file, \
            *self._controller._pickle_files)
        self._file_menu.config(width=27)
        self._file_menu.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')


        # GTI index box
        box12 = ttk.LabelFrame(row1,text='GTI selection')  
        box12.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        # This will be initialized whhen self._file is changed
        self._gti_sel_string = tk.StringVar()
        self._gti_sel_string.set('')
        gti_entry = tk.Entry(box12, textvariable=self._gti_sel_string)
        gti_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        gti_entry.to_change = 1


        # PLOT button
        #plot_frame = tk.Frame(row1)
        #plot_frame.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        #plot_frame.grid_rowconfigure(0,weight=1)
        #plot_frame.grid_rowconfigure(1,weight=1)
        #plot_frame.grid_columnconfigure(0,weight=1)
        plot_button = ttk.Button(row1,text='PLOT',
            command=lambda:self._plot())
        plot_button.grid(column=2,row=0,padx=5,pady=5,sticky='nswe')
        #plot_button = ttk.Button(plot_frame,text='RESET',
        #    command=self._reset_plot)
        #plot_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')


        # Poisson buttons
        box21 = ttk.LabelFrame(row2, text='Poisson level')
        box21.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        est_poi_button = tk.Button(box21,text='Est. Poi Level',
            command=self._estimate_poi)
        est_poi_button.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._poi_level = tk.DoubleVar()
        self._poi_level.set(0)
        poi_level_entry = tk.Entry(box21,textvariable=self._poi_level,
            width=10)
        poi_level_entry.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        sub_poi_button = tk.Button(box21,text='Subtract Poi',
            command=lambda: self._update_plot('','',''))
        sub_poi_button.grid(column=0,row=1,columnspan=2,padx=5,pady=5,
            sticky='nswe')

        
        # Rebin buttons
        box22 = ttk.LabelFrame(row2, text='Rebinning')
        box22.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')  
        self._rebin_factor = tk.StringVar()
        self._rebin_factor.set(-30)
        rebin_entry = tk.Entry(box22,textvariable=self._rebin_factor,
            width=10)
        rebin_entry.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        rebin_button = tk.Button(box22,text='Rebin',
            command=lambda: self._update_plot('','',''))
        rebin_button.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')


        
        # Normalization
        box23 = ttk.LabelFrame(row2, text='Normalization')
        box23.grid(column=2,row=0,padx=5,pady=5,sticky='nswe') 
        self._xy_flag = tk.IntVar()
        self._xy_flag.trace_add('write', self._update_plot)
        xy_checkbox = tk.Checkbutton(box23,text='X Vs. XY',
            var=self._xy_flag)
        xy_checkbox.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._norm = tk.StringVar()
        self._norm.set('Leahy')
        self._norm.trace_add('write', self._update_plot)
        norm_menu = ttk.OptionMenu(box23,self._norm,'Leahy','RMS')
        norm_menu.grid(column=1,row=0,padx=5,pady=5,sticky='nswe')
        bkg_label = tk.Label(box23,text='bkg value')
        bkg_label.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._bkg = tk.DoubleVar()
        self._bkg.set(0)
        bkg_entry = tk.Entry(box23,textvariable=self._bkg,width=10)
        bkg_entry.grid(column=1,row=1,padx=5,pady=5,sticky='nswe')

    def _update_label(self,var,indx,mode):
        selection = self._file.get()
        print('selecting',selection)

        data_file = os.path.join(self._controller._output_dir.get(),\
                                'analysis',\
                                self._controller._obs_id,
                                selection)
        self._power_list = PowerList.load(data_file)
        if 'LIST' in selection.upper():
            n_gti = self._power_list[0].history['N_GTIS']
            for child in self._frame1.winfo_children():
                for gchild in child.winfo_children():

                    for ggchild in gchild.winfo_children():
                        if hasattr(ggchild,'to_change') and ggchild['state']=='disabled':
                            ggchild.configure(state='normal')                  

                    if hasattr(gchild,'tag'):
                        gchild.configure(text='File (GTIs ({}): 0-{})'.\
                            format(n_gti,n_gti-1))

        else:
            self._power = pd.read_pickle(data_file)
            for child in self._frame1.winfo_children():
                for gchild in child.winfo_children():

                    for ggchild in gchild.winfo_children():
                        if hasattr(ggchild,'to_change'):
                            ggchild.configure(state='disabled')

                    if hasattr(gchild,'tag'):
                        gchild.configure(text='File')



    def _update_file_menu(self):
        # Reset var and delete all old options
        self._file.set('')
        self._file_menu["menu"].delete(0, "end")

        # Inserting new choices
        for f in self._controller._pickle_files:
            self._file_menu["menu"].add_command(label=f, 
                command=tk._setit(self._file,f))

    def _plot(self):
        if self._first_plot:
            self._ax.clear()
            self._first_plot = False
        else:
            self._ax.clear()
            self._poi_level.set(0)
            self._rebin_factor.set(-30)
            self._xy_flag.set(0)
            self._bkg.set(0)
            self._norm.set('Leahy')
            self._first_plot = True

        if 'LIST' in self._file.get().upper():
            if self._gti_sel_string.get() != '':     
                gti_indices = self.eval_gti_str(self._gti_sel_string.get())
                power_list = PowerList([p for p in self._power_list \
                    if p.history['GTI_INDEX'] in gti_indices])
                power = power_list.average_leahy()
            else:
                power = self._power_list.average_leahy()
            self._leahy = power
        else:
            self._leahy = self._power.leahy()

        self._to_plot = self._leahy.rebin(-30)
        self._to_plot.plot(ax=self._ax,lfont=14)
        self._canvas.draw()
        self._canvas.mpl_connect('motion_notify_event',self._update_cursor)

    def eval_gti_str(self,gti_str):
        if ',' in gti_str:
            div = gti_str.split(',')
        else:
            div = [gti_str]

        gti_indices = []
        for d in div:
            if '-' in d:
                start = int(d.split('-')[0])
                stop  = int(d.split('-')[1]) + 1
                gti_indices += [i for i in range(start,stop)]
            else:
                gti_indices += [int(d)]

        return sorted(list(set(gti_indices)))


    def _update_plot(self,var,indx,mode):

        poi_level = self._poi_level.get()
        bkg = self._bkg.get()
        norm = self._norm.get()

        poi = self._leahy.sub_poi(value=poi_level)

        if norm == 'RMS':
            to_rebin = poi.rms(bkg_cr=bkg)
        elif norm == 'Leahy':
            to_rebin = poi

        rebin_str = self._rebin_factor.get()
        first = True
        if ',' in rebin_str:
            rfcs = rebin_str.split(',')
            for rfc in rfcs:
                if first:
                    rebin = to_rebin.rebin(eval(rfc))
                    first = False
                else:
                    rebin = rebin.rebin(eval(rfc))
        else:
            rebin = to_rebin.rebin(eval(rebin_str))

        self._to_plot = rebin
        self._ax.clear()
        if self._xy_flag.get():
            self._to_plot.plot(ax=self._ax,xy=True,lfont=14)
        else:
            self._to_plot.plot(ax=self._ax,lfont=14)
        self._canvas.draw()
        

    def _rebin(self):
        pass

    def _reset_plot(self):
        self._poi_level.set(0)
        self._norm.set('Leahy')
        self._xy_flag.set(0)
        self._rebin_factor.set('-30')
        self._update_plot('','','')

    def _estimate_poi(self):
        if not self._leahy is None:
            mask = self._leahy.freq>=3000
            value = self._leahy.power[mask].mean()
            self._poi_level.set(value)

    def _init_plot_area(self,frame):

        self._fig = Figure(figsize=(6.3,5),dpi=100) 
        self._ax = self._fig.add_subplot() 

        self._canvas = FigureCanvasTkAgg(self._fig, master = frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(column=0,row=0,sticky='nswe')

        coor_frame = tk.Frame(frame)
        coor_frame.grid(column=0,row=1,pady=5,sticky='nswe')
        coor_frame.grid_columnconfigure(0,weight=1)
        coor_frame.grid_columnconfigure(1,weight=1)
        coor_frame.grid_columnconfigure(2,weight=1)
        coor_frame.grid_columnconfigure(3,weight=1)
        labelx = tk.Label(coor_frame,text='x coor: ')
        labelx.grid(column=0,row=0,pady=5,padx=5,sticky='nswe')
        self._x_pos = tk.Label(coor_frame,text=' ')
        self._x_pos.grid(column=1,row=0,pady=5,padx=5,sticky='nswe')
        labely = tk.Label(coor_frame,text='y coor: ')
        labely.grid(column=2,row=0,pady=5,padx=5,sticky='nswe')
        self._y_pos = tk.Label(coor_frame,text=' ')
        self._y_pos.grid(column=3,row=0,pady=5,padx=5,sticky='nswe')

        fit_button = tk.Button(frame,text='FIT',command = self._fit,
        height=2)
        fit_button.grid(column=0,row=2,pady=5,sticky='nswe')

    def _update_cursor(self,event):
        self._x_pos.configure(text=str(event.xdata))
        self._y_pos.configure(text=str(event.ydata))

    def _fit(self):
        self._new_child_window(self._fit_window)

    def _new_window(self, newWindow):
        self.new = tk.Toplevel(self._controller)
        newWindow(self.new, self._controller)  

    def _new_child_window(self, newWindow):
        new = tk.Toplevel(self._controller)
        newWindow(new, self)




