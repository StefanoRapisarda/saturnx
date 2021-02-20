import os
import numpy as np
import pandas as pd 
import pickle
import matplotlib.pyplot as plt

from astropy.io.fits import getdata,getval

from multiprocessing import Pool

from kronos.utils.time_series import my_rebin
from kronos.utils.generic import my_cdate

class Lightcurve(pd.DataFrame):
    '''
    Lightcurve object. It stores binned events in a range of energy

    HISTORY
    -------
    2020 04 ##, Stefano Rapisarda (Uppsala)
    '''

    _metadata = ['_low_en','_high_en','_tres','header','notes']

    def __init__(self, time_array = None, count_array = None,
        low_en_value = None, high_en_value = None,
        tres_value = None, header = None, notes = None):

        if time_array is None:
            column_dict = {'time':np.array([]),'counts':np.array([])}      
        else:
            column_dict = {'time':time_array,'counts':count_array}
            super().__init__(column_dict)

        self._low_en = low_en_value
        self._high_en = high_en_value

        self._tres = tres_value 

        if notes is None:
            self.notes = {}
        else: self.notes = notes
        if history is None:
            self.history = {}   
        else: self.history = history
        if header is None:
            self.header = {}
        else: self.header = header

    def __add__(self, other):
        assert len(self) == len(other),'You cannot add Lightcurves with different dimensions'
        assert self.tres == other.tres,'You cannot add Lightcurves with different time resolution'
        
        if np.array_equal(self.time, other.time):
            time = self.time
        else:
            # Defining a new time ax
            time = self.time - self.time.iloc[0] 
        counts = self.counts + other.counts
        if self.low_en and self.high_en and other.low_en and other.high_en:
            low_en = min(self.low_en,other.low_en)
            high_en = max(self.high_en,other.high_en)
        else:
            low_en = None
            high_en = None
        
        return Lightcurve(time,counts,low_en,high_en,self.tres)

    def __mul__(self,value):
        assert type(value) != str, 'You cannot multiply a lightcurve and a string'
        return Lightcurve(self.time,self.counts*value,
        self.low_en,self.high_en,self.tres,self.notes,self.history)

    def __rmul__(self,value):
        assert type(value) != str, 'You cannot multiply a lightcurve and a string'
        return self*value

    def __truediv__(self,value):
        assert not isinstance(value,str),'You cannot divide a Lightcurve by a string'
        assert value != 0,'Dude, you cannot divide by zero'
        return Lightcurve(self.time,self.counts/value,
        self.low_en,self.high_en,self.tres,self.notes,self.history)  
   
    def split(self,time_seg=16):

        history = self.history.copy()  
        header = self.header.copy()   

        if 'kronos.core.gti.Gti' in str(time_seg.__class__):
            gti = time_seg
            history['GTI_SPLITTING'] = my_cdate()
            header['N_GTIS'] = len(gti)

            lcs = []
            gti_index = 0
            for start,stop in zip(gti.start,gti.stop):
                mask = (self.time>= start) & (self.time<stop)
                time=self.time[mask]
                history_gti = history.copy()
                header_gti = header.copy()
                header_gti['GTI_INDEX'] = gti_index
                counts = self.counts[mask]
                lc = Lightcurve(time,counts,self.low_en,self.high_en,self.tres)
                lc.history = history_gti
                lc.header = header_gti
                lcs += [lc]
                gti_index += 1

        else:

            #print('Splitting in time seg',self.texp)
            if isinstance(time_seg,str): time_seg = eval(time_seg)
            assert time_seg <= self.texp,'Lightcurve duration is less than the specfied segment ({} < {})'.format(time_seg,self.texp)

            history['SEG_SPLITTING'] = my_cdate()
            header['SEG_DUR'] = time_seg

            seg_bins = int(time_seg/self.tres)
            n_segs = int(len(self)/seg_bins)
            header['N_SEGS'] = n_segs
            indices = [i*seg_bins for i in range(1,n_segs+1)]
            # !!! Time intervals must be contigous to use this!!! 
            time_array = np.split(self.time.to_numpy(),indices)[:-1]
            count_array = np.split(self.counts.to_numpy(),indices)[:-1]
            seg_index=0
            lcs = []
            for time,counts in zip(time_array,count_array):
                seg_header = header.copy()
                seg_header['SEG_INDEX'] = seg_index
                lc = Lightcurve(time,counts,self.low_en,self.high_en,self.tres)
                lc.history = history
                lc.header=seg_header
                lcs += [lc]
                seg_index+=1

        return LightcurveList(lcs)

    def rebin(self,factors=-30):
        if type(factors) != list: factors=[factors]

        history = self.history.copy()
        header = self.header.copy()
        history['REBIN'] = my_cdate()
        header['REBIN FACTOR'] = factors

        binned_counts = self.counts.to_numpy()
        binned_time = self.time.to_numpy()
        for f in factors:
            binned_time,binned_counts,dummy,dummy=my_rebin(binned_time,binned_counts,rf = f)
        lc = Lightcurve(binned_time,binned_counts,self.low_en,self.high_en,self.tres)
        lc.history=history
        lc.header = header
        return lc

    def plot(self,ax=False,cr=True,title=False,lfont=16,**kwargs):

        if not 'marker' in kwargs.keys(): kwargs['marker']='o'
        if not 'color' in kwargs.keys(): kwargs['color']='k'

        if ax is False:
            fig, ax = plt.subplots(figsize=(12,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        start = self.time.iloc[0]
        y = self.counts
        if cr: y = self.counts/self.tres
        x = self.time-start
        ax.plot(x,y,**kwargs)

        ax.set_xlabel('Time [{} s]'.format(start),fontsize=lfont)
        ax.set_ylabel('Counts',fontsize=lfont)
        if cr: ax.set_ylabel('Count rate [c/s]',fontsize=lfont)
        ax.grid()

    @staticmethod
    def from_event(events,time_res=1.,user_start_time=None,user_dur=None,low_en=0.,high_en=np.inf):

        if isinstance(time_res,str): time_res = eval(time_res)
    
        assert 'kronos.core.event.Event' in str(events.__class__),'Input must be an Event object'

        history = {}
        header = {}
        history['CREATION_DATE'] = my_cdate()
        history['CREATION_MODE'] = 'Lightcurve computed from Event object'
        header['FILE_NAME'] = events.header['FILE_NAME']
        header['DIR'] = events.header['DIR']
        header['MISSION'] = events.mission

        mission = events.mission

        if user_start_time is None: user_start_time = np.min(events.time)
        if user_dur is None: user_dur = np.max(events.time)-user_start_time
    
        # The following may look messy but it is to ensure that the lightcurve time
        # resolution is the one specified by the user
        length = user_dur
        # To get even the last few photons
        n_bins = int(length/time_res)
    
        start_time = user_start_time
        #stop_time = start_time + user_dur
        stop_time = start_time + n_bins*time_res

        # In this way the resolution is exactly the one specified by the user
        time_bins_edges = np.linspace(start_time-time_res/2.,stop_time+time_res/2.,n_bins+2,dtype=np.double)
        time_bins_center = np.linspace(start_time,stop_time,n_bins+1,dtype=np.double)

        if mission == 'NICER': 
            factor=100.
        elif mission == 'SWIFT':
            factor=100.
        else: 
            factor=1.
        low_ch = low_en*factor
        high_ch = high_en*factor
        mask = (events.pi >= low_ch) & (events.pi < high_ch)
        filt_time = events.time[mask]
                
        # Computing lightcurve
        counts,dummy = np.histogram(filt_time, bins=time_bins_edges)

        lc = Lightcurve(time_bins_center,counts,low_en,high_en,
                          tres_value = time_res)
        lc.history = history 

        return lc

    @staticmethod
    def read_from_fits(fits_file, extname='COUNTS'):
        '''
        Reads lightcurve from FITS file 
        '''

        assert os.path.isfile(fits_file),'FITS file does not exist'
        mission = getval(fits_file,'telescop',1)

        history = {}

        history['CREATION_DATE'] = my_cdate()
        history['CREATION_MODE'] = 'Gti created from fits file'
        history['FILE_NAME'] = os.path.basename(fits_file)
        history['DIR'] = os.path.dirname(fits_file)

        data = getdata(fits_file,extname=extname,header=False,memmap=True)
        col_names = data.columns.names

        assert 'TIME' in col_names, 'FITS file does not have TIME column'
        time = data['TIME']
        if 'COUNTS' in col_names:
            counts = data['COUNTS']
        elif 'RATE' in col_names:
            tres = np.round(np.median(np.ediff1d(time)),12)
            counts = data['RATE']*tres
        
        return Lightcurve(time_array=time,count_array=counts,
            history=history)
        

    @property
    def tot_counts(self):
        if len(self.counts) == 0:
            return 0
        else:
            return self.counts.sum()
       
    @property
    def cr(self):
        if len(self.counts) == 0:
            return 0
        else:
            return self.tot_counts/self.texp

    @property
    def cr_std(self):
        if len(self.counts) == 0:
            return 0
        else:
            return self.counts.std()/self.tres

    @property
    def texp(self):
        if len(self.time) == 0:
            return 0
        elif len(self.time) == 1:
            # This is because time is the center of the bin
            return self.tres
        else:
            return self.time.iloc[-1]-self.time.iloc[0]

    @property
    def tres(self):
        # Computing tres if not specified
        if (self._tres is None) and (not self.time is None):
            if len(self.time) == 0:
                return 0
            elif len(self.time) == 1:
                return None
            else:
                return np.round(np.median(np.ediff1d(self.time)),12)
        else:
            return self._tres

    @tres.setter
    def tres(self,value):
        self._tres = value

    @property
    def rms(self):
        if len(self.counts) == 0:
            return 0
        else:
            return 100*np.sqrt((self.counts**2).mean())

    @property
    def low_en(self):
        return self._low_en

    @low_en.setter
    def low_en(self,value):
        self._low_en = value

    @property
    def high_en(self):
        return self._high_en

    @high_en.setter
    def high_en(self,value):
        self._high_en = value



class LightcurveList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array(['kronos.core.lightcurve.Lightcurve'in str(i.__class__) for i in self]).all():
            raise TypeError('All the elements must be Lightcurve objects')

        self.history = {}
        self.notes = {}


    def __setitem__(self, index, lc):
        if not 'kronos.core.lightcurve.Lightcurve' in str(lc.__class__):
            raise TypeError('The item must be a Lightcurve object')
        self[index] = lc


    def join(self,mask=None):
        if mask is None:
            mask = np.ones(len(self),dtype=bool)
        else:
            assert len(mask) == len(self),'Mask must have the same size of LightcurveList'
        df_list = []
        for i in range(len(self)):
            if mask[i]: 
                df_list += [pd.DataFrame(self[i])]

        cond1 = len(set([self[i].tres for i in range(len(self)) if mask[i]])) != 1
        if cond1:
            print([self[i].tres for i in range(len(self)) if mask[i]])
            raise ValueError('Cannot concatenate Lightcurves with different time res')
        
        df = pd.concat(df_list,ignore_index=True)
        
        valid_index = [i for i in range(len(self)) if mask[i]][0]
        return Lightcurve(df.time,df.counts,
                          self[valid_index].low_en,self[valid_index].high_en,self[valid_index].tres)

    def fill_gaps(self):
        tres_array = np.array([lc.tres for lc in self])
        assert np.all(tres_array==tres_array[0]),'Cannot fill gaps if Lightcurves have different time res'

        new_lcs = []
        for i in range(len(self)):
            if i == 0:
                new_lcs += [self[i]]
                continue
            
            lc = self[i]
            prev_lc = self[i-1]
            tres = lc.tres 

            if (lc.time.iloc[0]-prev_lc.time.iloc[-1]) >  tres:
                filler = np.arange(prev_lc.time.iloc[-1]+tres,lc.time.iloc[0],tres)
                filled_lc = Lightcurve(filler,np.zeros(len(filler)),self[0].low_en,self[0].high_en,tres)
                new_lcs += [filled_lc]
            new_lcs += [self[i]]

        return LightcurveList(new_lcs)


    def split(self,time_seg=16):

        if 'kronos.core.gti.Gti' in str(time_seg.__class__) :
            gti = time_seg
            lc_list = []
            for l in self:
                lc_list += [i for i in l.split(gti)]
            return LightcurveList(lc_list) 

        else:          
            lc_list = []
            for l in self:
                lc_list += [i for i in l.split(time_seg)]
            return LightcurveList(lc_list)        

    def plot(self,ax=False,cr=True,title=False,lfont=16,**kwargs):

        if not 'marker' in kwargs.keys(): kwargs['marker']='o'
        if not 'color' in kwargs.keys(): kwargs['color']='k'

        if ax is False:
            fig, ax = plt.subplots(figsize=(12,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        start = np.array([t.time.iloc[0] for t in self]).min()
        for i in range(len(self)):
            y = self[i].counts
            if cr: y = self[i].cr
            x = (self[i].time.iloc[-1]+self[i].time.iloc[0])/2. - start
            ax.plot(x,y,**kwargs)

        ax.set_xlabel('Time [{} s]'.format(start),fontsize=lfont)
        ax.set_ylabel('Counts',fontsize=lfont)
        if cr: ax.set_ylabel('Count rate [c/s]',fontsize=lfont)
        ax.grid()

        
    def compare(self,lcs='all'):
        fig =plt.figure(figsize=(12,8))
        min_count = np.array([i.counts.min() for i in self]).min()
        max_count = np.array([i.counts.max() for i in self]).max()
        step = (max_count-min_count)/len(self)
        for i in range(len(self)):
            if type(lcs) == list:
                if not i in lcs: continue
            start = self[i].time.iloc[0]
            plt.plot(self[i].time-start,self[i].counts+i*step,label=f'{i}')
        plt.xlabel('Time [{} s]'.format(start))
        plt.ylabel('Counts')
        plt.legend()
        plt.grid()
        return fig

    @property
    def tot_counts(self):
        if len(self) == 0:
            return 0 
        else:
            return np.sum([i.tot_counts for i in self])

    @property
    def texp(self):
        if len(self) == 0:
            return 0
        else:
            return np.sum([i.texp for i in self])

    @property
    def cr(self):
        if len(self) == 0:
            return 0
        else:
            if len(set([i.texp for i in self])) == 1:
                return np.mean([i.cr for i in self])
            else:
                #print([i.tot_counts for i in self])
                return np.sum([i.cr*len(i) for i in self])/np.sum([len(i) for i in self])

    @property
    def cr_std(self):
        if len(self) == 0:
            return 0
        else:
            if len(set([i.texp for i in self])) == 1:
                return np.mean([i.cr_std for i in self])
            else:
                #print([i.cr_std for i in self],np.sum([len(i) for i in self]))
                return np.sqrt( np.sum([i.cr_std**2.*len(i) for i in self]) /np.sum([len(i) for i in self]) )

    @property    
    def mean(self):
        if len(self) == 0:
            return 0
        else:
            if len(set([len(i) for i in self])) != 1:
                raise ValueError('Lightcurves have different dimensions')
            if len(set([i.tres for i in self])) != 1:
                raise ValueError('Lightcurves have different time resolution')      
            first_lc = self[0]  
            time = first_lc.time - first_lc.time.iloc[0] + first_lc.tres/2.
            counts = np.vstack([i.counts.to_numpy() for i in self]).mean(axis=0)
            return Lightcurve(time,counts,first_lc.low_en,first_lc.high_en)

    def save(self,file_name='lightcurve_list.pkl',fold=os.getcwd()):
        try:
            with open(os.path.join(fold,file_name),'wb') as output:
                pickle.dump(self,output)
            print('LightcurveList saved in {}'.format(os.path.join(fold,file_name)))
        except Exception as e:
            print(e)
            print('Could not save LightcurveList')

    @staticmethod
    def load(file_name):
        assert os.path.isfile(file_name),f'{file_name} not found'
        with open(file_name,'rb') as infile:
            lc_list = pickle.load(infile)
        return lc_list
    
