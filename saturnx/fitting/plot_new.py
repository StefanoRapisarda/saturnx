import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import Menu

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import numpy as np

import pickle
import pandas as pd

import os
import sys
sys.path.append('/Volumes/Samsung_T5/saturnx')
import saturnx as kr
from saturnx.core.power import PowerSpectrum
from saturnx.utils.my_functions import my_rebin as rebin
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from fitting_functions import lorentzian
sys.setrecursionlimit(10000)

# Fitting data
import lmfit
from lmfit import Model,Parameters
from lmfit.model import save_modelresult,load_modelresult

import uuid

fig1 = Figure(figsize=(10,5),dpi=100)
gs1 = fig1.add_gridspec(1,2)
gs1.tight_layout(fig1)
ax11 = fig1.add_subplot(gs1[0])
ax12 = fig1.add_subplot(gs1[1])

fig2 = Figure(figsize=(10,5),dpi=100)
gs2 = fig2.add_gridspec(1,2)
gs2.tight_layout(fig2)
ax21 = fig2.add_subplot(gs2[0])
ax22 = fig2.add_subplot(gs2[1])

def random_color(n=1,hex=False):
    if n%3 == 1:
        r = 255
        g = int(np.random.randint(0,256//2))
        b = int(np.random.randint(0,256//2))
    elif n%3 == 2:
        r = int(np.random.randint(0,256//2))
        g = 255
        b = int(np.random.randint(0,256//2))   
    elif n%3 == 0:
        r = int(np.random.randint(0,256//2))
        g = int(np.random.randint(0,256//2))
        b = 255           
    rgbl=[r,g,b]
    ran_rgbl=tuple(rgbl)
    if hex:
        return '#%02x%02x%02x' % ran_rgbl
    else:
        return ran_rgbl

class Data:
    '''
    Container for data to plot and fit
    An instance of this must be initialized in the controller
    '''
    def __init__(self, x=None, y=None, yerr=None,
                x_toplot=None, y_toplot=None, yerr_toplot=None,
                info=None, norm=None):

        self.x = x
        self.y = y
        self.yerr = yerr

        self.x_toplot = x_toplot
        self.y_toplot = y_toplot
        self.yerr_toplot = yerr_toplot

        self.info = info
        self.norm = norm

    @classmethod
    def load(cls, filename):

        #power = PowerSpectrum.load(filename)
        power = pd.read_pickle(filename)
        x_ori = power.freq.to_numpy()
        y_ori = power.power.to_numpy()
        x = x_ori[x_ori>0]
        y = y_ori[x_ori>0]
        if 'spower' in power.columns:
            yerr = power.spower.to_numpy()[x_ori>0]
        else:
            yerr = None
        
        if not power.rms_norm is None: 
            norm = 'rms'
        elif not power.leahy_norm is None: 
            norm = 'leahy'
        elif (power.rms_norm is None) and (power.leahy_norm is None):
            norm = None
        info = power.meta_data

        return cls(x=x,y=y,yerr=yerr,
                   x_toplot=x, y_toplot=y, yerr_toplot=yerr,
                   info=info, norm=norm)

    def rebin(self, rebin_factor):
        if self.yerr_toplot is None:
            self.x_toplot, self.y_toplot, dummy, dummy = \
            rebin(self.x_toplot, self.y_toplot,rf=rebin_factor)
        else:
            self.x_toplot, self.y_toplot, dummy, self.yerr_toplot = \
            rebin(self.x_toplot, self.y_toplot,ye=self.yerr_toplot,rf=rebin_factor)  

    def reset(self):
        self.x_toplot = self.x
        self.y_toplot = self.y
        if not self.yerr is None:
            self.yerr_toplot = self.yerr

    def plot_power(self):
        ax11.clear()
        ax12.clear()
        ax11.set_xlabel('Frequency [Hz]')
        ax12.set_xlabel('Frequency [Hz]')
        if self.norm == 'rms':
            ax11.set_ylabel('Power [(rms/mean)$^2$]')
            ax12.set_ylabel('Power [(rms/mean)$^2$/Hz]')
        elif self.norm == 'leahy':
            ax11.set_ylabel('Leahy Power')
            ax12.set_ylabel('Leahy Power')
        else:
            ax11.set_ylabel('Power')  
            ax12.set_ylabel('Power') 

        x_toplot = self.x_toplot
        y_toplot = self.y_toplot
        yerr_toplot = self.yerr_toplot                

        if not yerr_toplot is None:
            ax11.errorbar(x_toplot,
                          x_toplot*y_toplot,
                          x_toplot*yerr_toplot,
                          fmt='-k')
            ax12.errorbar(x_toplot,
                          y_toplot,
                          yerr_toplot,
                          fmt='-k')   
        else:        
            ax11.plot(x_toplot,x_toplot*y_toplot,'-k')
            ax12.plot(x_toplot,y_toplot,'-k')  

        ax11.set_xscale('log') 
        ax11.set_yscale('log')
        ax11.grid()            

        ax12.set_xscale('log') 
        ax12.set_yscale('log')
        ax12.grid()  

class FitApp(tk.Tk):

    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)

        self.windows={}

        self.fit_funcs = {}
        self.fit_func = {}

        container = tk.Frame(self)
        container.grid(column=0,row=0,sticky='nswe')

        # Initializing data
        self.data = Data()

        self.plot_window = PlotWindow2(container, self)
        #tool_window = self._new_window(ToolWindow)

    def _init_menu(self):

        menu_bar = Menu(self)
        tk.Tk.config(self,menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        window_menu = Menu(menu_bar, tearoff=0)

        file_menu.add_command(label='Load Power',
        command=lambda: StartPage._start_session())
        file_menu.add_command(label='Quit',command=quit)

        window_menu.add_command(label='Data Info')
        window_menu.add_command(label='Main Toolbar',
        command=lambda: self._new_window(ToolWindow))      
            
        menu_bar.add_cascade(label='File', menu=file_menu) 
        menu_bar.add_cascade(label='Windows', menu=window_menu)

    def _new_window(self, newWindow):
        self.new = tk.Toplevel(self)
        newWindow(self.new, self)

class PlotWindow2:
    def __init__(self,parent, controller):
        self.parent = parent
        self._controller = controller

        self.plot_frame = tk.Frame(self.parent,width=1000,height=500)
        self.plot_frame.grid(column=0,row=1,padx=5,pady=5,sticky='NSEW')
        self.plot_frame.tkraise()

        self.canvas1 = FigureCanvasTkAgg(fig1,master = self.plot_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(column=0,row=0,sticky='NSWE')

        self.frame = tk.Frame(self.parent,height=35)
        self.frame.grid(column=0,row=0,padx=5,pady=5)

        self._make_rebin_buttons()
        self._make_button(2,'Load',self._load_data)
        self._make_button(3,'Reset',self._reset_data)
        self._make_button(4,'Fit',self._start_fit)
        self._make_button(5 ,'Quit',quit)

    def _make_rebin_buttons(self):
        frame1 = ttk.LabelFrame(self.frame,text='Rebin data',width=50)
        frame1.grid(column=0,row=0,padx=3,pady=3)
        self.e1 = ttk.Entry(frame1)
        self.e1.grid(column=0,row=0)
        button1 = ttk.Button(frame1,text='rebin',
        command = self._rebin_data)
        button1.grid(column=1,row=0)

    def _make_button(self,col=0,text='Text',command=None):
        '''
        Creates a generic button on the button frame
        '''
        button = tk.Button(self.frame,text=text,height=3,width=7,relief='raised',
        command=command)
        button.grid(column=col,row=0,padx=3,pady=3)  

    def _rebin_data(self):
        rebin_factor = int(self.e1.get())  
        self._controller.data.rebin(rebin_factor=rebin_factor)    
        self._reset_plot()
        # To avoid change of axis boundaries, I stored the initial
        # boundaries to use later on
        self._controller.ylim1 = ax11.set_ylim()
        self._controller.ylim2 = ax12.set_ylim()
        self._controller.xlim1 = ax11.set_xlim()
        self._controller.xlim2 = ax12.set_xlim()       


    def _load_data(self):
        filename = filedialog.askopenfilename(initialdir='/Volumes/BigBoy/NICER_data/MAXI_J1820+070/analysis/qpoCB_transition',
        title = 'Select a power spectrum')
        #filename = '/Volumes/Transcend/NICER/Cyg_X1/analysis/0100320101/power_E0.5_10.0_T0.0001220703125_128.0.pkl'
        self._controller.dir = os.path.dirname(filename)
        self._controller.data = Data.load(filename)
        self._reset_plot()

    def _reset_data(self):
        self._controller.data.reset()
        self._reset_plot()


    def _reset_plot(self):
        self._controller.data.plot_power()
        self._controller.plot_window.canvas1.draw()

    def _start_fit(self):
        print('start')
        self._controller._new_window(FitWindow)

    def _fix_axis(self):
        ax11.set_ylim(self._controller.ylim1)
        ax11.set_xlim(self._controller.xlim1)
        ax12.set_ylim(self._controller.ylim2)
        ax12.set_xlim(self._controller.xlim2)

class PlotWindow:
    def __init__(self,parent, controller):
        self.parent = parent
        self._controller = controller

        self.plot_frame = tk.Frame(self.parent,width=1000,height=500)
        self.plot_frame.grid(column=0,row=1,padx=5,pady=5,sticky='NSEW')
        self.plot_frame.tkraise()

        self.canvas1 = FigureCanvasTkAgg(fig1,master = self.plot_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(column=0,row=0,sticky='NSWE')

    def _fix_axis(self):
        ax11.set_ylim(self._controller.ylim1)
        ax11.set_xlim(self._controller.xlim1)
        ax12.set_ylim(self._controller.ylim2)
        ax12.set_xlim(self._controller.xlim2)

class ToolWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self.parent = parent

        self.frame = tk.Frame(self.parent,height=35)
        self.frame.grid(column=0,row=0,padx=5,pady=5)

        self._make_rebin_buttons()
        self._make_button(2,'Load',self._load_data)
        self._make_button(3,'Reset',self._reset_data)
        self._make_button(4,'Fit',self._start_fit)
        self._make_button(5 ,'Quit',quit)

    def _make_rebin_buttons(self):
        frame1 = ttk.LabelFrame(self.frame,text='Rebin data',width=50)
        frame1.grid(column=0,row=0,padx=3,pady=3)
        self.e1 = ttk.Entry(frame1)
        self.e1.grid(column=0,row=0)
        button1 = ttk.Button(frame1,text='rebin',
        command = self._rebin_data)
        button1.grid(column=1,row=0)

    def _make_button(self,col=0,text='Text',command=None):
        '''
        Creates a generic button on the button frame
        '''
        button = tk.Button(self.frame,text=text,height=3,width=7,relief='raised',
        command=command)
        button.grid(column=col,row=0,padx=3,pady=3)  

    def _rebin_data(self):
        rebin_factor = int(self.e1.get())  
        self._controller.data.rebin(rebin_factor=rebin_factor)    
        self._reset_plot()
        # To avoid change of axis boundaries, I stored the initial
        # boundaries to use later on
        self._controller.ylim1 = ax11.set_ylim()
        self._controller.ylim2 = ax12.set_ylim()
        self._controller.xlim1 = ax11.set_xlim()
        self._controller.xlim2 = ax12.set_xlim()       


    def _load_data(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(),
        title = 'Select a power spectrum')
        self._controller.data = Data.load(filename)
        self._reset_plot()

    def _reset_data(self):
        self._controller.data.reset()
        self._reset_plot()


    def _reset_plot(self):
        self._controller.data.plot_power()
        self._controller.plot_window.canvas1.draw()

    def _start_fit(self):
        print('start')
        self._controller._new_window(FitWindow)



class FitWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self.parent = parent
        self.parent.title = 'Fit window'


        self.index = 1
        self._controller.first_fit = True
        self.func_list = {'lorentzian':lorentzian}

        self.frame = tk.Frame(self.parent)
        self.frame.grid(column=0,row=0,padx=5,pady=5)

        self._plot_connected = False
        self._setFrames()


    def _setFrames(self):

        width = 200

        # Frequency bounday boxes
        self.freqFrame = ttk.LabelFrame(self.frame,text='Frequency boundaries',width=width,height=50)
        self.freqFrame.grid(column=0,row=0,padx=5,pady=5,sticky='nswe')
        self._setFreqRange()

        # Fitting functions options and drawing buttons
        self.fitFuncFrame = ttk.LabelFrame(self.frame,text='Fitting functions',width=width)
        self.fitFuncFrame.grid(column=0,row=1,padx=5,pady=5,sticky='nswe')
        self._setFittingFuncsI()
        self._setFittingFuncsII()
        self._setListBox()

        # This is to desplay fitting function parameters
        self.fitInfoFrame = ttk.LabelFrame(self.frame,text='Fitting parameters',width=width)
        self.fitInfoFrame.grid(column=0,row=2,padx=5,pady=5,sticky='nswe')
        self._setFittingPars()

    def _setFittingPars(self):
        self.listboxII = tk.Listbox(self.fitInfoFrame, selectmode='multiple')
        self.listboxII.grid(column=0,row=0,padx=5,pady=5,rowspan=2,sticky='nsew')

        self.par_val = tk.StringVar()
        reset = tk.Entry(self.fitInfoFrame, textvariable=self.par_val, width=10)
        reset.grid(column=1,row=0,padx=5,pady=5,sticky='wn')  

        reset = ttk.Button(self.fitInfoFrame, text='reset', command=lambda: self._reset_par_value())
        reset.grid(column=2,row=0,padx=5,pady=5,sticky='en')        

        self.vII = tk.IntVar()
        self.freeze = tk.Radiobutton(self.fitInfoFrame, text='freeze', 
        variable = self.vII, value = 1, command=lambda: self._reset_par_value(freeze=True))
        self.freeze.grid(column=1,row=1,padx=7,pady=7,sticky='enw')

        self.free = tk.Radiobutton(self.fitInfoFrame, text='free',
        variable = self.vII, value = 0,  command=lambda: self._reset_par_value())
        self.free.grid(column=2,row=1,padx=7,pady=7,sticky='enw')
        self.free.select() 

    def _freeze_par(self):
        par_val = float(self.par_val.get())        

    def _reset_par_value(self,freeze=False):
        # First getting the value then the selected function and parameter
        sel = self.listboxII.curselection()
        items = self.listboxII.get(0,tk.END)
        for index in sel[::-1]:
            item = items[index]
            key = int(item.split(')')[0].split('L')[1])
            par_name = item.split('=')[0].strip().split(')')[1].strip()
            pars = self._controller.fit_funcs[key]['par_name']

            for i in range(len(pars)):
                if par_name == self._controller.fit_funcs[key]['par_name'][i]:

                    if self.par_val.get() == '':
                        par_val = self._controller.fit_funcs[key]['par_value'][i]
                    else:
                        par_val = float(self.par_val.get())

                    self._controller.fit_funcs[key]['par_value'][i] = par_val  
                    if freeze:
                        self._controller.fit_funcs[key]['par_status'][i] = False
                    else:
                        self._controller.fit_funcs[key]['par_status'][i] = True
        self._print_par_value()
        self._plot_func()

    def _print_par_value(self):
        self.listboxII.delete(0,tk.END)
        for key,value in self._controller.fit_funcs.items():
            if 'plots' in value.keys():
                n_pars = len(value['par_value'])
                for i in range(n_pars):
                    line = 'L{:2}) {:4} = {:6.4} ({})'.\
                        format(key,value['par_name'][i],
                                   float(value['par_value'][i]),
                                   ('free' if value['par_status'][i] else 'frozen'))
                    self.listboxII.insert(tk.END,line) 
                    self.listboxII.itemconfig(tk.END,{'fg':value['color']})                 

                    

    def _setFreqRange(self):
        '''
        Sets the two buttons to entry start and stop frequency for fit
        '''
        self.startFreq = tk.Entry(self.freqFrame, width=10)
        self.startFreq.grid(column=0,row=0,padx=5,pady=5,sticky='W')
        self.startFreq.insert(0,0)

        self.stopFreq = tk.Entry(self.freqFrame, width=10)
        self.stopFreq.grid(column=1,row=0,padx=5,pady=5,sticky='W')
        self.stopFreq.insert(0,100)       

    def _setFittingFuncsI(self):
        '''
        Set a combobox with available function to fit and the buttons
        add and del
        '''
        func = tk.StringVar()
        self.funcs = ttk.Combobox(self.fitFuncFrame,textvariable=func)
        self.funcs['values'] = tuple([i for i,j in self.func_list.items()])
        self.funcs.grid(column=0,row=0, sticky='w',padx=5,pady=5)
        self.funcs.current(0)

        add_func = ttk.Button(self.fitFuncFrame, text='add', command=self._clickAdd)
        add_func.grid(column=1,row=0,padx=5,pady=5,sticky='e')

        del_func = ttk.Button(self.fitFuncFrame, text='del', command=self._clickDel)
        del_func.grid(column=2,row=0,padx=5,pady=5,sticky='e')

    def _setFittingFuncsII(self):
        '''
        Set the buttonds fit and close, and the radio buttons hold and draw
        '''
        fit = ttk.Button(self.fitFuncFrame, text='fit', command=self._fit_func)
        fit.grid(column=1,row=1,padx=5,pady=5,sticky='en')        

        close = ttk.Button(self.fitFuncFrame, text='close', command=self._close)
        close.grid(column=2,row=1,padx=5,pady=5,sticky='en')   

        save = ttk.Button(self.fitFuncFrame, text='save', command=self._save)
        save.grid(column=1,row=2,padx=5,pady=5,sticky='en')        

        clear = ttk.Button(self.fitFuncFrame, text='clear', command=self._clear)
        clear.grid(column=2,row=2,padx=5,pady=5,sticky='en')        

        self.v = tk.IntVar()
        self.draw = tk.Radiobutton(self.fitFuncFrame, text='draw', 
        variable = self.v, value = 1, 
        activebackground = 'green',
        command=self._draw_func)
        self.draw.grid(column=2,row=3,padx=7,pady=7,sticky='enw')

        self.hold = tk.Radiobutton(self.fitFuncFrame, variable = self.v, value = 0, text='hold', command=self._hold_func)
        self.hold.grid(column=1,row=3,padx=7,pady=7,sticky='enw')
        self.hold.select()    

    def _clear(self):
        self.listbox.delete(0,tk.END)
        self.listboxII.delete(0,tk.END)
        self._controller.fit_info_box.delete(0,tk.END)
        
        self._controller.first_fit = True
        self.index = 1
        self.fit_funcs = {}
        self.fit_func = {}

        self._controller.data.plot_power()
        self._controller.plot_window.canvas1.draw()


    def _save(self):
        name = 'fit'
        save_modelresult(self._controller.fit_result,'{}/{}.sav'.format(self._controller.dir,name))




    def _setListBox(self):
        '''
        Set the list box containing the additive functions
        '''
        # Here is where fittinf functions are listed
        self.listbox = tk.Listbox(self.fitFuncFrame, selectmode='multiple')
        self.listbox.grid(column=0,row=1,padx=5,pady=5,rowspan=2,sticky='nsew')

    def _clickAdd(self):
        col = random_color(n=self.index,hex=True)
        self._controller.fit_funcs[self.index] = {'name':self.funcs.get(),'color':col}
        self.listbox.insert(tk.END,str(self.index)+') '+self.funcs.get())
        self.listbox.itemconfig(self.index-1,{'fg':col})
        self.index += 1
        self._hold_func()

        for key,value in self._controller.fit_funcs.items():
            print(key,value)
        print('-'*80)

    def _clickDel(self):
        sel = self.listbox.curselection()
        print(sel)
        # the selection index MUST be reversed because self.listbox index changes
        # every time you remove an item
        for index in sel[::-1]:
            self.listbox.delete(index) 
            if 'plots' in self._controller.fit_funcs[index+1].keys():
                ax11.lines.remove(self._controller.fit_funcs[index+1]['plots'][0])
                ax12.lines.remove(self._controller.fit_funcs[index+1]['plots'][1])
            del self._controller.fit_funcs[index+1]
        items = self.listbox.get(0,tk.END)
        if len(items) != 0:
            self._reset_index()
            self._plot_func()
        else:
            self._controller.data.plot_power()
            self._controller.plot_window.canvas1.draw()

        self._print_par_value()
        self._hold_func()

        for key,value in self._controller.fit_funcs.items():
            print(key,value)
        print('-'*80)

    def _reset_index(self):
        items = self.listbox.get(0,tk.END)
        old_func_info = []
        old_items = []
        for i in range(len(items)):
            item = items[i]
            old_items += [item.split(')')[1].strip()]
            old_index = int(item.split(')')[0].strip())
            old_func_info += [self._controller.fit_funcs[old_index]]
        
        self.listbox.delete(0,tk.END)
        self._controller.fit_funcs = {}
        
        for i in range(len(items)):
            self.listbox.insert(tk.END,str(i+1)+') '+old_items[i])
            self.listbox.itemconfig(i,{'fg':old_func_info[i]['color']})
            self._controller.fit_funcs[i+1] = old_func_info[i]

        self.index = len(items) + 1

    def _close(self):
        self.parent.destroy()

    def _draw_func(self):
        self._plot_connected = True
        self._sel_index = int(self.listbox.curselection()[0])
        self._connect()

    def _hold_func(self):
        self._plot_connected = False
        self._disconnect()

        
    def _plot_func(self,reset=False):

        x = self._controller.data.x_toplot
        if not x is None:
            counter = 0
            sum1 = np.zeros(len(x))
            sum2 = np.zeros(len(x))
            for key,value in self._controller.fit_funcs.items():
                if 'par_value' in value.keys():

                    if 'plots' in value.keys():
                        ax11.lines.remove(value['plots'][0])
                        ax12.lines.remove(value['plots'][1])

                    col = value['color']
                    pars = value['par_value']
                    func = self.func_list[value['name']]
                    y = func(x,*pars)
                    lor1, = ax11.plot(x,y*x,'--',color = col)
                    lor2, = ax12.plot(x,y,'--',color = col)
                    self._controller.fit_funcs[key]['plots'] = [lor1,lor2]
                    sum1 += y*x
                    sum2 += y
                    counter +=1
            if 'plot' in self._controller.fit_func.keys():
                ax11.lines.remove(self._controller.fit_func['plot'][0])
                ax12.lines.remove(self._controller.fit_func['plot'][1])           
            if counter > 1:
                all1, = ax11.plot(x,sum1,'r-')
                all2, = ax12.plot(x,sum2,'r-')
                self._controller.fit_func['plot'] = [all1,all2]
            self._controller.plot_window._fix_axis()
            self._controller.plot_window.canvas1.draw()
            self._print_par_value()

    def _connect(self):
        self.cidclick = self._controller.plot_window.canvas1.mpl_connect('button_press_event',self._on_click)
        self.cidscroll = self._controller.plot_window.canvas1.mpl_connect('scroll_event',self.on_roll)

    def _disconnect(self):
        if self._plot_connected:
            self._controller.plot_window.canvas1.mpl_disconnect(self.cidclick)
            self._controller.plot_window.canvas1.mpl_disconnect(self.cidscroll)

    def _on_click(self,event):
        if not event.dblclick:
            print('clicking')
            # Left click or right lick
            if event.button == 1 or event.button == 3:
                # Position of the cursor
                self.xpos = event.xdata
                self.ypos = event.ydata

                if (self.xpos is None) or (self.ypos is None):
                    self.draw.deselect()
                else:

                    if not 'par_value' in self._controller.fit_funcs[self._sel_index+1].keys():
                        q = 10
                    else:
                        q = self._controller.fit_funcs[self._sel_index+1]['par_value'][1]
                    self._controller.fit_funcs[self._sel_index+1]['par_value'] = \
                        [self.ypos/self.xpos,q,self.xpos]
                    self._controller.fit_funcs[self._sel_index+1]['par_status'] = \
                        [True,True,True]    
                    self._controller.fit_funcs[self._sel_index+1]['par_name'] = \
                        ['amp','q','freq']                                       
                    self._plot_func()

            if event.button == 2:
                self._disconnect()
                self.draw.deselect()

    def on_roll(self,event):
        q = self._controller.fit_funcs[self._sel_index+1]['par_value'][1]
        if q > 1:
            step = 1
        else:
            step = 0.1
        if event.button == 'up':
            q -= step
            if q <= 0: q = 0.
            self._controller.fit_funcs[self._sel_index+1]['par_value'][1] = q
            self._plot_func()
        elif event.button == 'down':              
            q += step
            self._controller.fit_funcs[self._sel_index+1]['par_value'][1] = q
            self._plot_func()

              

    def _fit_func(self):  

        if self._controller.first_fit:
            self._controller._new_window(PlotFitWindow)

        self._hold_func()

        x = self._controller.data.x_toplot
        y = self._controller.data.y_toplot
        yerr = self._controller.data.yerr_toplot
        self.fit_mask = (x>= float(self.startFreq.get())) & (x<= float(self.stopFreq.get()))
        self._build_model()
        #init = self.model.eval(self.fit_pars,x=x[self.fit_mask])
        self._controller.fit_result = self.model.fit(y[self.fit_mask],self.fit_pars,x=x[self.fit_mask],
                                    weights=1./(yerr[self.fit_mask]),mthod='leastsq')
        #self.comps = self._controller.fit_result.eval_components(x=x[self.fit_mask])
        self._update_fit_funcs()
        self._plot_func()
        if self._controller.first_fit:
            self._plot_fit()
            self._controller.first_fit = False
        else:
            self._update_fit_plot()
        self._update_info()

    def _update_fit_funcs(self):
        for key, value in self._controller.fit_funcs.items():
            if 'plots' in value.keys():
                par_names = value['par_name']
                n_pars = len(par_names)

                for i in range(n_pars):
                    par_name = 'L{}_{}'.format(key,par_names[i])
                    self._controller.fit_funcs[key]['par_value'][i] = \
                                self._controller.fit_result.best_values[par_name]

    def _update_fit_plot(self):
        x = self._controller.data.x_toplot
        y = self._controller.data.y_toplot
        yerr = self._controller.data.yerr_toplot
        self.line1.set_ydata(self._controller.fit_result.best_fit-y[self.fit_mask])
        self.line2.set_ydata((self._controller.fit_result.best_fit-y[self.fit_mask])**2/yerr[self.fit_mask]**2/self._controller.fit_result.nfree)    
        self._controller.canvas2.draw()


    def _plot_fit(self):
        x = self._controller.data.x_toplot
        y = self._controller.data.y_toplot
        yerr = self._controller.data.yerr_toplot

        self.line1,=ax21.plot(x[self.fit_mask],(self._controller.fit_result.best_fit-y[self.fit_mask]),'-r')
        self.line2,=ax22.plot(x[self.fit_mask],(self._controller.fit_result.best_fit-y[self.fit_mask])**2/yerr[self.fit_mask]**2/self._controller.fit_result.nfree,'-r')

        # Residuals
        max = np.max(abs(self._controller.fit_result.best_fit-y[self.fit_mask]))
        #ax.plot(x[fit_mask],result.best_fit-y,'r')
        ax21.set_xscale('log')
        ax21.set_ylim([-max-max/3,max+max/3])
        ax21.grid()
        ax21.set_xlabel('Frequency [ Hz]')
        ax21.set_ylabel('Residuals [model-data]')
        ax21.set_title('').set_visible(False)
        ax21.set_xlim(self._controller.xlim1)

        ax21bis = ax21.twinx()
        ax21bis.set_ylim(self._controller.ylim1)
        ax21bis.errorbar(x,x*y,x*yerr,alpha=0.3,fmt='-k')
        ax21bis.tick_params(axis='both',which='both',length=0)
        ax21bis.set_yscale('log')  
        ax21bis.set_yticklabels([])     
            

        # Contribution to chi2
        ax22.set_ylabel('Contribution to $\chi^2$')

        ax22.set_xscale('log')
        ax22.set_xlabel('Frequency [ Hz]')
        ax22.grid()
        ax22.set_title('').set_visible(False)
        ax22.set_xlim(self._controller.xlim2)
        ax22.yaxis.set_label_position('right')
        ax22.yaxis.tick_right()

        ax22bis = ax22.twinx()
        ax22bis.set_ylim(self._controller.ylim1)
        ax22bis.errorbar(x,x*y,x*yerr,alpha=0.3,fmt='-k')
        ax22bis.tick_params(axis='both',which='both',length=0)
        ax22bis.set_yscale('log')  
        ax22bis.set_yticklabels([])     
 

        if self._controller.first_fit: self._controller.canvas2.draw()

    def _build_model(self):
        first = True
        for key, value in self._controller.fit_funcs.items():
            print(key)
            if 'plots' in value.keys():
                par_names = value['par_name']
                n_pars = len(par_names)
                func = self.func_list[value['name']]

                print('This one',key,func)
                
                tmp_model = Model(func,prefix='L{}_'.format(key))
                if first:
                    first = False
                    self.fit_pars = tmp_model.make_params()
                    self.model = tmp_model
                else:
                    self.fit_pars.update(tmp_model.make_params())
                    self.model += tmp_model

                par_label = par_names

                for i in range(n_pars):
                    par_val = value['par_value'][i]
                    status = value['par_status'][i]
                    self.fit_pars['L{}_{}'.format(key,par_label[i])].set(value=par_val,vary=status,min=0)

    def _update_info(self):
        self._controller.report = lmfit.fit_report(self._controller.fit_result).split('\n')
        for line in self._controller.report:
            self._controller.fit_info_box.insert(tk.END,line)
        if not self._controller.first_fit:
            self._controller.fit_info_box.insert(tk.END,'='*70+'\n')

class PlotFitWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self.parent = parent

        plot_frame = tk.Frame(self.parent,width=1000,height=500)
        plot_frame.grid(column=0,row=0,padx=5,pady=5,sticky='NSEW')
        plot_frame.tkraise()

        self._controller.canvas2 = FigureCanvasTkAgg(fig2,master = plot_frame)
        self._controller.canvas2.draw()
        self._controller.canvas2.get_tk_widget().grid(column=0,row=0,sticky='NSWE')

        info_frame = tk.Frame(self.parent,width=1000,height=250)
        info_frame.grid(column=0,row=1,padx=5,pady=5,sticky='NSEW')

        self._controller.fit_info_box = tk.Listbox(self.parent, selectmode='multiple')
        self._controller.fit_info_box.grid(column=0,row=1,padx=5,pady=5,sticky='nsew')



class DataInfoWindow:
    def __init__(self,parent,controller):
        self._controller = controller
        self.parent = parent




if __name__ == '__main__':
    app = FitApp()
    app.mainloop()