import os
import glob
import pathlib
from tkinter import filedialog

class Controller:

    def __init__(self,model,view,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self._model = model
        self._view = view

        # Control variables
        # (To check what it is currently plotted on canvas)
        self._data_on_canvas = False
        self._fit_on_canvas = False

        self._init_inputs()
        self._tracing_variables()
        self._init_button_commands()
        self._bind_entries()

    def _init_inputs(self):
        self._view._input_dir_box._input_dir.set(self._model['plot_fields']._data_dir)


    def _tracing_variables(self):
        # File menu
        self._view._file_box._file.trace_add('write',self._update_gti_label)

        # Normalization
        self._view._norm_box._xy_flag.trace_add('write',self._update_plot)
        self._view._norm_box._norm.trace_add('write', self._update_plot)

    def _init_button_commands(self):
        # Command buttons
        self._view._plot_button.configure(command=self._plot)
        self._view._fit_button.configure(command=self._fit)
        self._view._reset_button.configure(command=self._reset)

        # Poisson buttons
        self._view._poi_box._est_poi_button.configure(command=self._est_poi)
        self._view._poi_box._sub_poi_button.configure(command=self._sub_poi)

        # Rebin buttons
        self._view._rebin_box._rebin_button.configure(command=self._rebin)

        # Input dir
        self._view._input_dir_box._set_dir_button.configure(command=self._set_dir)

    def _bind_entries(self):
        self._view._input_dir_box._dir_entry.bind(
            '<Return>',lambda event, flag=False: self._set_dir(flag)
            )

    def _update_file_list(self):
        print('Updating file list')
        menu = self._view._file_box._file_menu
        menu["menu"].delete(0,'end')
        files = self._model['plot_fields']._files
        if len(files) != 0:
            for item in files:
                menu["menu"].add_command(
                    label=item,
                    command=lambda value=item:self._view._file_box._file.set(value)
                )

    def _update_gti_label(self,vat,index,mode):
        print('Updating label')

    def _update_plot(self,var,index,mode):
        print('Updating plot')

    def _update_model(self):
        print('Updating model')
        data_dir = self._view._input_dir_box._input_dir.get()
        self._model['plot_fields']._data_dir = pathlib.Path(data_dir)

        self._model['plot_fields']._files = glob.glob(data_dir+'/*.pkl')

    def _plot(self):
        print('Clicking plot')

    def _fit(self):
        print('Clicking fit')

    def _reset(self):
        print('Clicing reset')

    def _est_poi(self):
        print('Clicking est poi')

    def _sub_poi(self):
        print('Clicking sub poi')

    def _rebin(self):
        print('Clicking rebin')

    def _set_dir(self, choice_flag=True):
        if choice_flag:
            data_dir = filedialog.askdirectory(
                initialdir=self._model['plot_fields']._data_dir,
                title='Select folder for data products')
            self._view._input_dir_box._input_dir.set(data_dir)
            
        self._update_model()
        self._update_file_list()



    