import os
import numpy as np
import pandas as pd 
import pickle
import math
import matplotlib.pyplot as plt

from astropy.io.fits import getdata,getval

from kronos.core.gti import Gti
from kronos.core.event import Event
from kronos.utils.time_series import my_rebin, rebin_arrays
from kronos.utils.generic import is_number, my_cdate

class Lightcurve(pd.DataFrame):
    '''
    Lightcurve object. It stores binned events in a range of energy

    HISTORY
    -------
    2020 04 ##, Stefano Rapisarda (Uppsala)

    NOTE
    ----
    2020 02 23, Stefano Rapisarda (Uppsala)
        When initialising the pandas dataframe with a dictionary, at 
        least one of the items must by an object with __len__, othewise,
        if using scalars, you must specify an integer
    '''

    _metadata = ['_low_en','_high_en','meta_data','notes']

    def __init__(self, time_array = np.array([]), count_array = None,
        low_en_value = None, high_en_value = None,
        rate_array = None,
        meta_data = None, notes = None):

        # Initialisation
        column_dict = {'time':time_array,'counts':count_array,'rate':rate_array}
        super().__init__(column_dict)

        # Rate or count rate initialization
        if len(time_array) != 0:
            if rate_array is None: 
                self.rate = self.counts/self.tres
            elif count_array is None:
                self.counts = self.rate*self.tres

        # Main attributes
        self._low_en = low_en_value
        self._high_en = high_en_value

        if notes is None:
            self.notes = {}
        else: self.notes = notes

        if meta_data is None:
            self.meta_data = {}
        else: self.meta_data = meta_data
        self.meta_data['LC_CRE_DATE'] = my_cdate()

    def __add__(self, other):
        if type(other) == type(Lightcurve()):
            if len(self) != len(other):
                raise TypeError('You cannot add Lightcurves with different dimensions')
            if self.tres != other.tres:
                raise TypeError('You cannot add Lightcurves with different time resolution')
        
            # Initialize time 
            if np.array_equal(self.time, other.time):
                time = self.time
            else:
                # Defining a new time ax 
                time = self.time - self.time.iloc[0] 

            # Initialize counts
            counts = self.counts + other.counts

            # Initialize energy bands
            if self.low_en and self.high_en and other.low_en and other.high_en:
                low_en = min(self.low_en,other.low_en)
                high_en = max(self.high_en,other.high_en)
            else:
                low_en = None
                high_en = None
        
        else:  
            if type(other) == str:
                if is_number(other): 
                    other=eval(other)
                else:
                    raise TypeError('Cannot add string to Lightcurve')
            time = self.time
            counts = self.counts + other
            low_en, high_en = self.low_en, self.high_en

        return Lightcurve(time_array = time,count_array = counts,
            low_en_value = low_en, high_en_value = high_en,
            meta_data = self.meta_data, notes = self.notes)

    def __mul__(self,value):
        
        if type(value) in [list,np.ndarray]:
            if len(value) == 0: return self
            
            print('--->',len(value),value)
            lcs = []
            for item in value:
                if type(item) == str:
                    if is_number(item): 
                        value=eval(item)
                    else:
                        raise TypeError('Cannot multiply string to Lightcurve')                
                else:
                    try:
                        float(item)
                    except Exception:
                        raise TypeError('Array items must be numbers')

                counts = self.counts*item
                lcs += [Lightcurve(self.time,counts,
                    self.low_en,self.high_en,
                    notes=self.notes,meta_data=self.meta_data)]
            return LightcurveList(lcs)

        else:
            if type(value) == str:
                if is_number(value): 
                    value=eval(value)
                else:
                    raise TypeError('Cannot multiply string to Lightcurve')                
            else:
                try:
                    float(value)
                except Exception:
                    raise TypeError('Value must be a number')            
            counts = self.counts*value    

            return Lightcurve(time_array = self.time,count_array = counts,
                low_en_value = self.low_en,high_en_value = self.high_en,
                notes=self.notes,meta_data=self.meta_data)

    def __rmul__(self,value):
        return self*value

    def __truediv__(self,value):
        if type(value) == str:
            if is_number(value): 
                value=eval(value)
            else:
                raise TypeError('Cannot divide Lightcurve by string')                
        else:
            try:
                float(value)
            except Exception:
                raise TypeError('Value must be numbers')         
        if value == 0:
            raise ValueError('Dude, you cannot divide by zero')
        counts = self.counts/value

        return Lightcurve(time_array = self.time, count_array = counts,
            low_en_value = self.low_en, high_en_value = self.high_en,
            notes=self.notes, meta_data=self.meta_data)  
   
    def split(self,time_seg=16):

        meta_data = self.meta_data.copy()   

        if type(time_seg) == type(Gti()):
            print('===> Splitting GTI')
            gti = time_seg
            meta_data['SPLITTING_GTI'] = my_cdate()
            meta_data['N_GTIS'] = len(gti)

            lcs = []
            for gti_index,(start,stop) in enumerate(zip(gti.start,gti.stop)):
                mask = (self.time>= start) & (self.time<stop)
                time=self.time[mask]
                meta_data_gti = meta_data.copy()
                meta_data_gti['GTI_INDEX'] = gti_index
                counts = self.counts[mask]
                lc = Lightcurve(time_array = time, count_array = counts,
                    low_en_value = self.low_en, high_en_value = self.high_en,
                    meta_data = meta_data_gti, notes = self.notes)
                lcs += [lc]

        else:

            #print('Splitting in time seg',self.texp)
            print('===> Splitting Segment')
            if type(time_seg) == str:
                if is_number(time_seg): 
                    time_seg=eval(time_seg)
                else:
                    raise TypeError('Time segment must be a number')                
            else:
                try:
                    float(time_seg)
                except Exception:
                    raise TypeError('Time segment must be a number')         
            if time_seg == 0:
                raise ValueError('Time segment cannot be zero')

            if time_seg >= self.texp:
                print('Lightcurve duration is less than the specfied segment ({} < {})'.format(time_seg,self.texp))
                print('Returning original Lightcurve')
                return LightcurveList([self])

            seg_bins = int(time_seg/self.tres)
            n_segs = int(len(self)/seg_bins)           
            #n_segs = int(self.texp/time_seg)

            meta_data['SPLITTING_SEG'] = my_cdate()
            meta_data['SEG_DUR'] = time_seg
            meta_data['N_SEGS'] = n_segs

            indices = [i*seg_bins for i in range(1,n_segs+1)]
            # np.split performs faster then ad hoc loop
            # The last interval goes from last index to the end of the
            # original array, so it is excluded
            # !!! Time intervals must be contigous to use this!!! 
            time_array = np.split(self.time.to_numpy(),indices)[:-1]
            count_array = np.split(self.counts.to_numpy(),indices)[:-1]
            lcs = []
            for seg_index,(time,counts) in enumerate(zip(time_array,count_array)):
                seg_meta_data = meta_data.copy()
                seg_meta_data['SEG_INDEX'] = seg_index
                lc = Lightcurve(time_array = time, count_array = counts,
                    low_en_value = self.low_en, high_en_value = self.high_en,
                    meta_data = seg_meta_data, notes = self.notes)
                lcs += [lc]

        return LightcurveList(lcs)

    def rebin(self,factors=1):

        # Checking input
        if type(factors) != list: 
            if type(factors) == str:
                if is_number(factors): 
                    factors=eval(factors)
                else:
                    raise TypeError('Rebin factor must be a number')                
            else:
                try:
                    float(factors)
                except Exception:
                    raise TypeError('Rebin factor must be a number')         
            if factors <= 0:
                raise ValueError('Rebin factor cannot be negative')  

            new_factors = [factors] 
        else:
            new_factors = []
            for f in factors:
                if type(f) == str:
                    if is_number(f): 
                        f=eval(f)
                    else:
                        raise TypeError('Rebin factor must be a number')                
                else:
                    try:
                        float(f)
                    except Exception:
                        raise TypeError('Rebin factor must be a number')         
                if f <= 0:
                    raise ValueError('Rebin factor cannot be negative') 
                new_factors += [f]                           

        # Initialising meta_data
        meta_data = self.meta_data.copy()
        meta_data['REBINNING'] = my_cdate()
        meta_data['REBIN_FACTOR'] = new_factors #list

        # Rebinning
        binned_counts = self.counts.to_numpy()
        binned_time = self.time.to_numpy()
        for f in new_factors:
            if f == 1:
                if len(new_factors) == 1:
                    binned_counts = [binned_counts]
            else:
                #binned_time,binned_counts,d,d = my_rebin(binned_time,binned_counts,rf = f,average=False) 
                #binned_counts = my_rebin_single(binned_counts,rf = f, average=False)
                #binned_time = my_rebin_single(binned_time,rf = f)    
                binned_time, binned_counts= rebin_arrays(
                    binned_time,rf=f,tres = self.tres,
                    arrays=binned_counts,bin_mode=['sum'],exps=[1])
                print('===>',len(binned_counts),len(binned_time))    
        lc = Lightcurve(time_array = binned_time, count_array = binned_counts[0],
            low_en_value = self.low_en, high_en_value = self.high_en,
            meta_data = meta_data, notes = self.notes)
        return lc

    def plot(self,norm=None,ax=False,cr=True,title=False,lfont=16,**kwargs):

        if not 'marker' in kwargs.keys(): kwargs['marker']='o'
        if not 'color' in kwargs.keys(): kwargs['color']='k'

        if ax is False:
            fig, ax = plt.subplots(figsize=(12,6))

        if (not title is False) and (not ax is False):
            ax.set_title(title)

        start = self.time.iloc[0]
        y = self.counts

        ylabel = 'Counts'
        if cr: 
            y = self.rate
            ylabel = 'Count rate [c/s]'
        if not norm is None:
            if type(norm) == str: norm = eval(norm)
            y /= norm
        x = self.time-start
        ax.plot(x,y,label='{}_{}'.format(self.low_en,self.high_en),**kwargs)

        ax.set_xlabel('Time [{} s]'.format(start),fontsize=lfont)
        ax.set_ylabel(ylabel,fontsize=lfont)
        ax.grid()
        ax.legend(title='[keV]')

    @staticmethod
    def from_event(events,time_res=1.,user_start_time=None,user_dur=None,low_en=0.,high_en=np.inf):

        if isinstance(time_res,str): time_res = eval(time_res)
    
        if type(events) != type(Event()):
            raise TypeError('Input must be an Event object')

        meta_data = {}
        meta_data['LC_CRE_DATE'] = my_cdate()
        meta_data['LC_CRE_MODE'] = 'Lightcurve computed from Event object'
        # Copying some info from the event file
        keys_to_copy = ['EVT_FILE_NAME','DIR','MISSION','INFO_FROM_HEADER'
                        'MISSION']
        for key in keys_to_copy:
            if key in events.meta_data.keys():
                meta_data[key] = events.meta_data[key]

        if user_start_time is None: user_start_time = np.min(events.time)
        if user_dur is None: user_dur = np.max(events.time)-user_start_time
    
        # The following may look messy but it is to ensure that the lightcurve time
        # resolution is the one specified by the user.
        # Events partially covered by a bin are excluded
        length = user_dur
        n_bins = int(length/time_res)
    
        start_time = user_start_time
        stop_time = start_time + n_bins*time_res

        # In this way the resolution is exactly the one specified by the user
        time_bins_edges = np.linspace(start_time-time_res/2.,stop_time+time_res/2.,n_bins+1,dtype=np.double)
        time_bins_center = np.linspace(start_time,stop_time,n_bins,dtype=np.double)

        # Conversion FROM energy TO channels
        mission = events.meta_data['MISSION']
        if mission == 'NICER': 
            factor=100.
        elif mission == 'SWIFT':
            factor=100.
        else: 
            factor=1.
        low_ch = low_en*factor
        high_ch = high_en*factor

        # Selecting events according to energy
        mask = (events.pi >= low_ch) & (events.pi < high_ch)
        filt_time = events.time[mask]
                
        # Computing lightcurve
        counts,dummy = np.histogram(filt_time, bins=time_bins_edges)

        lc = Lightcurve(time_array = time_bins_center, count_array = counts,
            low_en_value = low_en, high_en_value = high_en,
            meta_data = meta_data, notes = {})

        return lc

    @staticmethod
    def read_from_fits(fits_file, extname='COUNTS'):
        '''
        Reads lightcurve from FITS file 
        '''

        assert os.path.isfile(fits_file),'FITS file does not exist'
        mission = getval(fits_file,'telescop',1)

        meta_data = {}

        meta_data['LC_CRE_DATE'] = my_cdate()
        meta_data['LC_CRE_MODE'] = 'Gti created from fits file'
        meta_data['FILE_NAME'] = os.path.basename(fits_file)
        meta_data['DIR'] = os.path.dirname(fits_file)

        data = getdata(fits_file,extname=extname,meta_data=False,memmap=True)
        col_names = data.columns.names

        assert 'TIME' in col_names, 'FITS file does not have TIME column'
        time = data['TIME']
        if 'COUNTS' in col_names:
            counts = data['COUNTS']
        elif 'RATE' in col_names:
            tres = np.round(np.median(np.ediff1d(time)),12)
            counts = data['RATE']*tres
        
        return Lightcurve(time_array=time,count_array=counts,
            meta_data = meta_data)
        

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
            return self.counts.std()

    @property
    def texp(self):
        # This is the difference between the center of the last and 
        # first photon time bin. This is equal to the time interval 
        # between the edges of the first and last bin minus the width
        # of a single bin (tres*nbins-tres) 
        if len(self.time) > 1:
            return np.round(len(self)*self.tres,
                decimals=int(abs(math.log10(self.tres/1e+6))))
        else:
            return 0

    @property
    def tres(self):
        # Computing tres if not specified
        if len(self.time) > 1:
            #return self.time.iloc[2]-self.time.iloc[1]
            tres = np.median(np.ediff1d(self.time))
            tres = np.round(tres,int(abs(math.log10(tres/1e+6))))
            return tres
        else:
            return 0

    @property
    def rms(self):
        if len(self.counts) == 0:
            return 0
        else:
            return np.sqrt(np.sum(self.counts**2)/len(self.counts))

    @property
    def frac_rms(self):
        if len(self.counts) == 0:
            return 0
        else:
            return np.sqrt(np.var(self.counts)/np.mean(self.counts)**2)

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
    
