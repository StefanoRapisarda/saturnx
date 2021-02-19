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

def extractData(fig):
    '''
    Extract x, y, and eventual errors from matplotlib Figure
    '''

    assert isinstance(fig, matplotlib.figure.Figure)

    axes = fig.gca()

    # Extracing x and y data first
    try:
        x = axes.lines[0].get_xdata()
        y = axes.lines[0].get_ydata()
    except Exception as e:
        print(e)
        print('Could not extract x and y data')
        return None, None, None, None

    # Extracing errors
    xerr = None
    yerr = None
    conts = axes.containers
    if len(conts) != 0:

        errorbarcont = conts[0]
        plotlines = errorbarcont[0]
        barlinecols = errorbarcont[2]

        assert np.array_equal(x,plotlines.get_xdata())
        assert np.array_equal(y,plotlines.get_ydata())

        for linecollection in barlinecols:
            segs = linecollection.get_segments()

            # Checking if its an x or y err
            line = segs[0].reshape(1,4)[0]
            unique, counts = np.unique(line, return_counts=True)
            dup = unique[counts>1]
            # test for error on x
            test_x = ((line-dup)[1]==0)&((line-dup)[3]==0)

            upp = []
            low = []
            for i in range(len(segs)):
                seg = segs[i]
                if test_x: 
                    low += [x[i]-seg[0][0]]
                    upp += [seg[1][0]-x[i]]
                else:
                    low += [y[i]-seg[0][1]]
                    upp += [seg[1][1]-y[i]]
            upp = np.array(upp)
            low = np.array(low)

            if not np.allclose(upp,low): print("WARNING: Upper and lower error bars are not equal")
            
            # Checking if its an x or y err
            line = segs[0].reshape(1,4)[0]
            unique, counts = np.unique(line, return_counts=True)
            dup = unique[counts>1]
            # test for error on x
            test_x = ((line-dup)[1]==0)&((line-dup)[3]==0)
            if test_x:
                xerr = upp
            else:
                yerr = upp

    return x,y,xerr,yerr





class PlotWindow:
    def __init__(self,parent):
        self.parent = parent
        self.parent.title('Main Plotting window')

        self.rebin = 0
        self.data = {}
        self.info = {}

        # At first, let's make empty frames
        self._makeFrames()
        self._makeTopButtons()
        self._makeMenu()
        self._plot()

        self.fig1 = Figure(figsize=(5,5))
        self.fig2 = Figure(figsize=(5,5))

    # Frames

    def _makeFrames(self):
        self._makeButtonFrame()
        self._makePlotFrames()
        self._makeInfoFrame()

    def _makeButtonFrame(self):
        self.frameb = tk.Frame(self.parent,height=35)
        self.frameb.grid(column=0,row=0,padx=5,pady=5,columnspan=2)

    def _makeInfoFrame(self):
        self.framei = ttk.LabelFrame(self.parent,height=120)
        self.framei.grid(column=0,row=3,padx=5,pady=5,columnspan=2)

    def _makePlotFrames(self):
        self.frame1 = ttk.LabelFrame(self.parent,text='x Vs x*y',width=550,height=550)
        self.frame1.grid(column=0,row=1,padx=5,pady=5,sticky='NSEW')
        self.frame2 = ttk.LabelFrame(self.parent,text='x Vs y',width=550,height=550)
        self.frame2.grid(column=1,row=1,padx=5,pady=5,sticky='NSEW')      

    # Buttons 

    def _makeTopButtons(self):
        self._makeRebinButtons()
        self._makeButton(3,'Plot',self._plotPower) #
        self._makeButton(4,'Reset',self._reset_plot)
        self._makeButton(5,'Fit',lambda: self._newWindow(FitWindow))
        self._makeButton(6,'Quit',quit)

    def _makeRebinButtons(self):
        frame1 = ttk.LabelFrame(self.frameb,text='Rebin data')
        frame1.grid(column=0,row=0,padx=3,pady=3)
        self.e1 = ttk.Entry(frame1)
        self.e1.grid(column=0,row=0)
        button1 = ttk.Button(frame1,text='rebin',
        command=self._rebin)
        button1.grid(column=1,row=0)

    def _makeButton(self,col=0,text='Text',command=None):
        '''
        Creates a generic button on the button frame
        '''
        button = tk.Button(self.frameb,text=text,height=3,width=7,relief='raised',
        command=command)
        button.grid(column=col,row=0,padx=3,pady=3)  

    # Menu

    def _makeMenu(self):
        menu_bar = Menu(self.parent)
        self.parent.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label='Load Power',command=self._loadPower)
        menu_bar.add_cascade(label='File', menu=file_menu)  


    # Actions
    # Actions triggered by events              
       
    def _reset_plot(self):
        self.rebin = 0
        self._updateData(reset=True)
        self._plotPower()

    def _rebin(self):
        self.rebin = int(self.e1.get())
        self._updateData()
        self._plotPower()

    def _loadPower(self):
        '''
        Loads a power spectrum object, storing the unpbinned arrays into class
        attrbutes and initializing the self.data keys corresponding to binned
        data
        '''
        filename = filedialog.askopenfilename(initialdir=os.getcwd(),
        title = 'Select a power spectrum')
        power = tg.PowerSpectrum.load(filename)
        x = power.freq.to_numpy()
        y = power.power.to_numpy()
        self.x_ori = x[x>0]
        self.y_ori = y[x>0]
        self.data['x_rebin'] = self.x_ori.copy()
        self.data['y_rebin'] = self.y_ori.copy()
        if 'spower' in power.columns:
            yerr = power.spower.to_numpy()
            self.yerr_ori = yerr[x>0]
            self.data['yerr_rebin'] = self.yerr_ori.copy()
        else:
            self.yerr_ori = None
            self.data['yerr_rebin'] = None

        
        if not power.rms_norm is None: 
            self.data['norm'] = 'rms'
        elif not power.leahy_norm is None: 
            self.data['norm'] = 'leahy'
        elif (power.rms_norm is None) and (power.leahy_norm is None):
            self.data['norm'] = None
        self.info = power.history

    def _updateData(self,reset=False):

        if reset:
            self.data['x_rebin'] = self.x_ori.copy()
            self.data['y_rebin'] = self.y_ori.copy()  
            if not self.yerr_ori is None:
                self.data['yerr_rebin'] = self.yerr_ori.copy()
            else:
                self.data['yerr_rebin'] = None
        else:   

            if self.rebin != 0:
                if self.data['yerr_rebin'] is None:
                    self.data['x_rebin'],self.data['y_rebin'],dummy,dummy = \
                    rebin(self.data['x_rebin'],self.data['y_rebin'],rf=self.rebin)
                else:
                    self.data['x_rebin'],self.data['y_rebin'],dummy,self.data['yerr_rebin'] = \
                    rebin(self.data['x_rebin'],self.data['y_rebin'],ye=self.data['yerr_rebin'],rf=self.rebin)

 
    # Plotting

    def _plotPower(self):
        self.fig1 = powerFig(self.data,xy=True).fig
        self.fig2 = powerFig(self.data).fig
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self._plot()


    def _plot(self):
        self.canvas1 = FigureCanvasTkAgg(fig1,master=self.frame1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(column=0,row=0,sticky='NSWE')

        self.canvas2 = FigureCanvasTkAgg(fig2,master=self.frame2)
        self.canvas2.draw()      
        self.canvas2.get_tk_widget().grid(column=0,row=0,sticky='NSWE')  

    def _newWindow(self, newWindow):
        try:
            if self.new_wim.state()=='normal':
                self.new_win.focus()
        except Exception as e:
            self.new_window = tk.Toplevel(self.parent)
            newWindow(self.new_window)

class powerFig:
    def __init__(self,data,xy=False):
        self.fig = plt.figure(figsize=(5,5))

        self.x_binned = data['x_rebin']
        self.y_binned = data['y_rebin']
        self.yerr_binned = data['yerr_rebin']
      
        self.xy = xy
        self.norm_label = data['norm']

        self.set_labels()
        self.plot_data()
        self.set_other_options()

    def set_labels(self):
        plt.xlabel('Frequency [Hz]')
        if self.norm_label == 'rms':
            if self.xy:
                plt.ylabel('Power [(rms/mean)$^2$]')
            else:
                plt.ylabel('Power [(rms/mean)$^2$/Hz]')
        elif self.norm_label == 'leahy':
            plt.ylabel('Leahy Power')
        else:
            plt.ylabel('Power')  

    def plot_data(self):
        if self.xy:
            if self.yerr_binned is None:
                plt.plot(self.x_binned,self.y_binned*self.x_binned,'-k')
            else:
                print(len(self.x_binned),len(self.y_binned),len(self.yerr_binned))
                plt.errorbar(self.x_binned,self.y_binned*self.x_binned,self.yerr_binned*self.x_binned,fmt='-k')
        else:
            if self.yerr_binned is None:
                plt.plot(self.x_binned,self.y_binned,'-k')
            else:
                plt.errorbar(self.x_binned,self.y_binned,self.yerr_binned,fmt='-k')





    def set_other_options(self):
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()


class FitWindow:
    def __init__(self,parent,fig1=None,fig2=None,data=None):
        self.parent = parent
        self.parent.title = 'Fit window'

        self._setFrames()
        self._setFittingFuncs([lorentzian.name])
        self._setListBox()

        self.fig1 = fig1
        self.fig2 = fig2
        self.data = data

        funcs = {lorentzian.name:lorentzian}

    def _setFrames(self):
        self.freqFrame = ttk.LabelFrame(self.parent,text='Frequency boundaries')
        self.freqFrame.grid(column=0,row=0,padx=5,pady=5)

        self.fitFuncFrame = ttk.LabelFrame(self.parent,text='Fitting functions')
        self.fitFuncFrame.grid(column=0,row=1,padx=5,pady=5)

    def _setFreqRange(self):
        self.startFreq = ttk.Entry(self.freqFrame)
        self.startFreq.grid(column=0,row=0,padx=5,pady=5,sticky='W')
        self.startFreq.entry(0,'min')

        self.stopFreq = ttk.Entry(self.freqFrame)
        self.stopFreq.grid(column=1,row=0,padx=5,pady=5,sticky='W')
        self.stopFreq.entry(0,'max')       

    def _setFittingFuncs(self,funcs):
        func = tk.StringVar()
        self.funcs = ttk.Combobox(self.fitFuncFrame,textvariable=funcs)
        self.funcs['values'] = tuple(funcs)
        self.funcs.grid(column=0,row=0)
        self.funcs.current(0)

        add_func = ttk.Button(self.fitFuncFrame, text='add', command=self._clickAdd)
        add_func.grid(column=1,row=0)

        del_func = ttk.Button(self.fitFuncFrame, text='del', command=self._clickDel)
        del_func.grid(column=2,row=0)

    def _setListBox(self):
        frame = tk.Frame(self.fitFuncFrame)
        frame.grid(column=0,row=1,columnspan=3)
        self.listbox = tk.Listbox(frame, selectmode='multiple')
        self.listbox.pack()

    def _clickAdd(self):
        self.listbox.insert(tk.END,self.funcs.get())
        self.fig1,self.fig2 = self.DrawFunc(func=funcs[self.funcs.get()],self.data,self.fig1,self.fig2).draw()

    def _clickDel(self):
        sel = self.listbox.curselection()
        for index in sel[::-1]:
            self.listbox.delete(index)   

    def figs(self):
        return self.fig1, self.fig2


class DrawFunc:

    def __init__(self,func,data,fig1,fig2):

        self.func = func
        self.x = data['x_rebin']
        self.q = 10.and
        self.pars = []
        self.prev_func = 0
        self.first = True
        self.first_click = True

        self.fig1 = fig1
        self.fig2 = fig2

    def draw(self):

        axes1 = self.fig1.add_axes()
        axes2 = self.fig2.add_axes()

        if self.func.name == 'Lorentzian'

            lor1, = axes1.plot(self.x,self.func(self.x,0.01,self.q,1)*self.x,'--')
            lor2, = axes2.plot(self.x,self.func(self.x,0.01,self.q,1),'--')
        
        return self.fig1,self.fig2

if __name__ == '__main__':
    root = tk.Tk()
    window = PlotWindow(root)
    root.mainloop()