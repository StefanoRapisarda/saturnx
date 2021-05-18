import os
import sys
from typing import overload
import numpy as np
import glob
import logging
import pathlib
from datetime import datetime

import tkinter as tk

sys.path.append('/Volumes/Samsung_T5/kronos')
from kronos.gui.windows import MakePowerWin, LogWindow
from kronos.utils.logging import make_logger

from kronos.scripts.make_lc import make_nicer_lc
from kronos.scripts.make_power import make_power
from kronos.scripts.read_lc_fits import read_lc

class TimingApp:
    def __init__(self):

        self.ui = MakePowerWin()
        self.ui.geometry("925x835")
        self.ui._timing_tab._comp_button['command'] = self._compute
        #self.ui._button1['command'] = self._compute_power
        #self.ui._button2['command'] = self._compute_lightcurve

    def _compute(self):
        if self.ui._timing_tab._comp_lc.get(): self._compute_lightcurve()
        if self.ui._timing_tab._read_lc.get(): self._read_lightcurve()
        if self.ui._timing_tab._comp_pow.get(): self._compute_power()

    def _read_lightcurve(self):
        '''
        Reads lightcurve from a FITS file. This is supposed to be a
        binned lightcurve computed via ftools.
        '''

        # Opening log window
        # The text widget in the opened box will be self.ui._logs
        self.ui._new_window(LogWindow)

        # Reading box parameters
        # -------------------------------------------------------------
        self.ui._timing_tab._read_boxes()

        mission = self.ui._mission.get().strip()

        ext = self.ui._timing_tab._event_ext.get().strip()
        if ext[0] == '.': ext = ext[1:]

        output_suffix = self.ui._timing_tab._output_suffix.get()

        indir = self.ui._timing_tab._input_dir.get()
        outdir = os.path.join(self.ui._timing_tab._output_dir.get())
        # -------------------------------------------------------------

        log_name = make_logger('read_lc',outdir,self.ui._logs)
 
        #logging.info('COMMENTS:')
        #logging.info(self.ui._comments)

        for i,obs_id in enumerate(self.ui._obs_ids):

            logging.info('Processing obs ID {} ({}/{})'.\
                format(obs_id,i+1,len(self.ui._obs_ids)))

            # Looking for specified FITS file according to specified
            # parameters
            if mission == 'NICER':
                root_dir = os.path.join(indir,obs_id,'xti/event_cl')
                fits_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,self.ui._timing_tab._event_str.get().strip(),ext))     
            elif mission == 'RXTE':
                root_dir = os.path.join(indir,'pca')
            elif mission == 'HXMT':
                root_dir = os.path.join(indir,obs_id)
                fits_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,self.ui._timing_tab._event_str.get().strip(),ext))

            # Verifying that there is only a fits file corresponding to
            # selected criteria
            if len(fits_files)==1:
                fits_file = fits_files[0]
            elif len(fits_files)>1:
                msg = 'There is more than one FITS file selected for'\
                    ' obs ID {}. Skipping.'.format(obs_id)
                logging.info(msg)
                continue
            elif len(fits_files)==0:
                msg = 'There are no event file selected for'\
                    ' obs ID {}. Skipping.'.format(obs_id)
                logging.info(msg)
                continue 

            # If energy bands are selected, checking if the FITS file 
            # name has the energy bands in its name
            # !!! This means that energy band selection in this function
            # is perfomed according to FITS file name !!!
            if len(self.ui._en_bands) != 0 and mission != 'HXMT':
                en_bands = [[(e.split('-')[0]),(e.split('-')[1])]\
                    for e in self.ui._en_bands] 
                flag = False
                for en_band in en_bands:
                    low_en = en_band[0]
                    high_en = en_band[1]
                    if (low_en in fits_file and high_en in fits_file):
                        flag = True
                if not flag:
                    msg = 'FITS file does not contain selected energy bands'\
                        ' for obs ID {}. Skipping.'.format(obs_id)
                    continue
                        
            result = read_lc(fits_file,destination=outdir,
                mission=mission,output_suffix=output_suffix,
                log_name=log_name)

            if not result:
                logging.info('Something went wrong with obs ID {}'.\
                    format(obs_id))
        
        logging.info('Everything done!')
        logging.shutdown()   

    def _compute_lightcurve(self):

        # Reading box parameters
        self.ui._timing_tab._read_boxes()

        # Opening log window
        # The Text windger in the opened box will be self.ui._logs
        self.ui._new_window(LogWindow)

        # Reading mission, energy bands, time resolution, and options
        # -------------------------------------------------------------
        mission = self.ui._mission.get().strip()
        en_bands = [[float(e.split('-')[0]),float(e.split('-')[1])]\
                    for e in self.ui._en_bands]
        tres = [np.double(m.split(',')[0].split(':')[1])\
                for m in self.ui._fmodes]
        split_event= self.ui._timing_tab._split_event.get()  
        # -------------------------------------------------------------

        override = self.ui._timing_tab._override.get()

        # Reading input and output dirs
        indir = self.ui._timing_tab._input_dir.get()
        outdir = os.path.join(self.ui._timing_tab._output_dir.get())

        log_name = make_logger('compute_lc',outdir,self.ui._logs) 

        #logging.info('COMMENTS:')
        #logging.info(self.ui._comments)

        logging.info('Computing Lightcurve') 
        
        for i,obs_id in enumerate(self.ui._obs_ids):

            logging.info('Processing obs ID {} ({}/{})'.\
                format(obs_id,i+1,len(self.ui._obs_ids)))

            if mission == 'NICER':
                script = make_nicer_lc
                root_dir = os.path.join(indir,obs_id,'xti/event_cl')
                event_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,
                self.ui._timing_tab._event_str.get().strip(),
                self.ui._timing_tab._event_ext.get().strip()
                ))      
            elif mission == 'RXTE':
                root_dir = os.path.join(indir,'pca')
            elif mission == 'HXMT':
                root_dir = os.path.join(indir,obs_id)
                event_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,
                self.ui._timing_tab._event_str.get().strip(),
                self.ui._timing_tab._event_ext.get().strip()
                ))

            # Checking there is a single event
            if len(event_files)==1:
                event_file = event_files[0]
            elif len(event_files)>1:
                msg = 'There is more than one event file selected for'\
                    ' obs ID {}. Skipping.'.format(obs_id)
                logging.info(msg)
                continue
            elif len(event_files)==0:
                msg = 'There are no event file selected for'\
                    ' obs ID {}. Skipping.'.format(obs_id)
                logging.info(msg)
                continue 


            for i in range(len(tres)):
                logging.info('Computing fmode {}'.\
                                format(self.ui._fmodes[i]))

                test = script(event_file,tres=tres[i],
                    en_bands=en_bands,
                    split_event=split_event,destination=outdir,
                    output_suffix=self.ui._timing_tab._output_suffix.get().strip(),
                    override=override,log_name=log_name)

                if not test:
                    logging.info('Something went wrong with obs ID {}'.\
                        format(obs_id))
                    logging.info('tres: {}'.format(tres))
        
        logging.info('Everything done!')   
        logging.shutdown()         

    def _compute_power(self):

        # Reading box parameters
        # -------------------------------------------------------------
        self.ui._timing_tab._read_boxes()
        mission = self.ui._mission.get().strip()
        en_bands = [[float(e.split('-')[0]),float(e.split('-')[1])]\
                    for e in self.ui._en_bands]
        tres = [np.double(m.split(',')[0].split(':')[1])\
                for m in self.ui._fmodes]
        tseg = [np.double(m.split(',')[1].split(':')[1])\
                for m in self.ui._fmodes]  
        suffix=self.ui._timing_tab._output_suffix.get().strip()
        outdir = self.ui._timing_tab._output_dir.get()
        # -------------------------------------------------------------

        override = self.ui._timing_tab._override.get()

        # Opening log window
        # The Text windger in the opened box will be self.ui._logs
        self.ui._new_window(LogWindow)

        log_name = make_logger('compute_power',outdir,self.ui._logs) 

        #logging.info('COMMENTS:')
        #logging.info(self.ui._comments)

        logging.info('Computing power') 
        for i,obs_id in enumerate(self.ui._obs_ids):

            logging.info('Processing obs ID {} ({}/{})'.\
                format(obs_id,i+1,len(self.ui._obs_ids)))

            wf = os.path.join(outdir,'analysis',obs_id)  
            logging.info('Working folder: {}'.format(wf))    

            for e in range(len(en_bands)):
                low_en = en_bands[e][0]
                high_en = en_bands[e][1]
                logging.info('Computing energy {}'.\
                             format(self.ui._en_bands[e]))

                for i in range(len(tres)):
                    logging.info('Computing fmode {}'.\
                                  format(self.ui._fmodes[i]))

                    # Checking lightcurve list files
                    if suffix == '':
                        file_name = 'lc_list_E{}_{}_T{}.pkl'.\
                            format(low_en,high_en,tres[i])
                    else:
                        file_name = 'lc_list_E{}_{}_T{}_{}.pkl'.\
                            format(low_en,high_en,tres[i],suffix)
                    lc_list_files = glob.glob('{}/{}'.format(wf,file_name))

                    if len(lc_list_files)==1:
                        lc_list_file = lc_list_files[0]
                    elif len(lc_list_files)>1:
                        msg = 'There is more than one lc list file selected for'\
                            ' obs ID {obs_ID}. Skipping.'.format(obs_id)
                        logging.info(msg)
                        continue
                    elif len(lc_list_files)==0:
                        msg = 'There are no lc_list_file file selected for'\
                            ' obs ID {}. Skipping.'.format(obs_id)
                        logging.info(msg)
                        logging.info('Was looking for {}'.format(file_name))
                        continue       

                    test = make_power(lc_list_file,destination=wf,
                        tseg=tseg[i],log_name=log_name,override=override)

                    if not test:
                        logging.info('Something went wrong with obs ID {}'.\
                            format(obs_id))
                        logging.info('e: {}-{} keV, tres: {}'.\
                            format(low_en,high_en,tres[i]))
        
        logging.info('Everything done!')
        logging.shutdown()   

if __name__ == '__main__':
    app = TimingApp()
    app.ui.mainloop()
                    







  

            


