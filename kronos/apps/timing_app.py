import os
import numpy as np
import glob
import logging
from datetime import datetime

import tkinter as tk
from ..gui.windows import MakePowerWin, LogWindow
from ..functions.my_functions import initialize_logger

from ..scripts.make_lc import make_nicer_lc
from ..scripts.make_power import make_nicer_power

class TimingApp:
    def __init__(self):

        self.ui = MakePowerWin()
        self.ui._button1['command'] = self._compute_power
        self.ui._button2['command'] = self._compute_lightcurve

    def _compute_lightcurve(self):

        # Reading box parameters
        self.ui._read_boxes()

        # Opening log window
        # The Text windger in the opened box will be self.ui._logs
        self.ui._new_window(LogWindow)

        mission = self.ui._mission.get().strip()
        en_bands = [[float(e.split('-')[0]),float(e.split('-')[1])]\
                    for e in self.ui._en_bands]
        tres = [np.double(m.split(',')[0].split(':')[1])\
                for m in self.ui._fmodes]

        indir = self.ui._input_dir.get()
        outdir = self.ui._output_dir.get()

        # For logging purposes
        # -------------------------------------------------------------
        now = datetime.now()
        date = ('%d_%d_%d') % (now.day,now.month,now.year)
        time = ('%d_%d') % (now.hour,now.minute)
        log_name = os.path.basename('compute_lc_D{}_T{}'.\
                                    format(date,time))

        initialize_logger(log_name, text_widget=self.ui._logs)

        logging.info('Creating log folder...')
        log_dir = os.path.join(outdir,'logs')
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        # -------------------------------------------------------------  

        logging.info('COMMENTS:')
        logging.info(self.ui._comments)

        logging.info('Computing Lightcurve') 
        
        for i,obs_id in enumerate(self.ui._obs_ids):

            logging.info('Processing obs ID {} ({}/{})'.\
                format(obs_id,i+1,len(self.ui._obs_ids)))

            if mission == 'NICER':
                script = make_nicer_lc
                root_dir = os.path.join(indir,obs_id,'xti/event_cl')
                event_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,
                self.ui._event_str.get().strip(),
                self.ui._event_ext.get().strip()
                ))

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

            elif mission == 'RXTE':
                root_dir = os.path.join(indir,'pca')


            for e in range(len(en_bands)):
                low_en = en_bands[e][0]
                high_en = en_bands[e][1]
                logging.info('Computing energy {}'.\
                             format(self.ui._en_bands[e]))

                for i in range(len(tres)):
                    logging.info('Computing fmode {}'.\
                                  format(self.ui._fmodes[i]))

                    test = script(event_file,destination=outdir,
                    tres=tres[i],low_en=low_en, high_en=high_en,
                    log_dir=log_dir,log_name=log_name,drama=True,
                    output_suffix=self.ui._output_suffix.get().strip())

            if not test:
                logging.info('Something went wrong with obs ID {}'.\
                    format(obs_id))
        
        logging.info('Everything done!')
        os.system(f'mv {log_name}.log {log_dir}')              

    def _compute_power(self):

        # Reading box parameters
        self.ui._read_boxes()

        # Opening log window
        # The Text windger in the opened box will be self.ui._logs
        self.ui._new_window(LogWindow)

        mission = self.ui._mission.get().strip()
        en_bands = [[float(e.split('-')[0]),float(e.split('-')[1])]\
                    for e in self.ui._en_bands]
        tres = [np.double(m.split(',')[0].split(':')[1])\
                for m in self.ui._fmodes]
        tseg = [np.double(m.split(',')[1].split(':')[1])\
                for m in self.ui._fmodes]  

        indir = self.ui._input_dir.get()
        outdir = self.ui._output_dir.get()
        gti_dur = self.ui._gti_dur.get()
        split_event= self.ui._split_event.get()  


        # For logging purposes
        # -------------------------------------------------------------
        now = datetime.now()
        date = ('%d_%d_%d') % (now.day,now.month,now.year)
        time = ('%d_%d') % (now.hour,now.minute)
        log_name = os.path.basename('compute_power_D{}_T{}'.\
                                    format(date,time))

        initialize_logger(log_name, text_widget=self.ui._logs)

        logging.info('Creating log folder...')
        log_dir = os.path.join(outdir,'logs')
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        # -------------------------------------------------------------

        logging.info('COMMENTS:')
        logging.info(self.ui._comments)

        logging.info('Computing power') 
        counter = 1
        
        for i,obs_id in enumerate(self.ui._obs_ids):

            logging.info('Processing obs ID {} ({}/{})'.\
                format(obs_id,i+1,len(self.ui._obs_ids)))

            if mission == 'NICER':
                script = make_nicer_power
                root_dir = os.path.join(indir,obs_id,'xti/event_cl')
                event_files = glob.glob('{}/*{}*.{}*'.format(
                root_dir,
                self.ui._event_str.get().strip(),
                self.ui._event_ext.get().strip()
                ))            

                if len(event_files)==1:
                    event_file = event_files[0]
                elif len(event_files)>1:
                    msg = 'There is more than one event file selected for'\
                        ' obs ID {obs_ID}. Skipping.'.format(obs_id)
                    logging.info(msg)
                    continue
                elif len(event_files)==0:
                    msg = 'There are no event file selected for'\
                        ' obs ID {}. Skipping.'.format(obs_id)
                    logging.info(msg)
                    continue  

            elif mission == 'RXTE':
                root_dir = os.path.join(indir,'pca') 

            for e in range(len(en_bands)):
                low_en = en_bands[e][0]
                high_en = en_bands[e][1]
                logging.info('Computing energy {}'.\
                             format(self.ui._en_bands[e]))

                for i in range(len(tres)):
                    logging.info('Computing fmode {}'.\
                                  format(self.ui._fmodes[i]))

                    test = script(event_file,destination=outdir,
                    tres=tres[i],tseg=tseg[i],gti_dur=gti_dur,
                    low_en=low_en, high_en=high_en,
                    split_event=split_event,drama=True,
                    log_dir=log_dir,log_name=log_name,
                    output_suffix=self.ui._output_suffix.get().strip())

            if not test:
                logging.info('Something went wrong with obs ID {}'.\
                    format(obs_id))
        
        logging.info('Everything done!')
        os.system(f'mv {log_name}.log {log_dir}')

if __name__ == '__main__':
    app = TimingApp()
    app.ui.mainloop()
                    







  

            


