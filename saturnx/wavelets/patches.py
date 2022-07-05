import pathlib
import pickle

from shapely.geometry import Polygon, Point
from skimage import measure
from scipy.stats import chi2
from scipy.signal import savgol_filter

import functools
import concurrent

import matplotlib.pyplot as plt

import numpy as np
import pyximport
pyximport.install({"include_dirs":np.get_include()})
from .cextract_mask import extract_mask_cython

from saturnx.utils.generic import my_cdate

class Patches:
    '''
    Object storing wavelet significance "patches" (contours). Patches are
    extracted by a wavelet transform given a SINGLE confidence level

    Each patch is a shapely.geometry.Polygon
    '''

    def __init__(self,patches=None,nf=None,nt=None,
        verdicts=None,flags=None,conf_level=None,patch_info=None,
        meta_data=None,notes=None):
        '''
        PARAMETERS
        ----------
        patches: list
            List of shapely.geometry.Polygon objects
        nf: int or None
            Number of frequency bins in the original wavelet transform
        nt: int or None
            Number of time bins in the original wavelet transform
        '''

        self.patches = patches
        self.nf = nf
        self.nt = nt
        self.verdicts = verdicts
        self.flags = flags

        self.patch_info = patch_info

        self.conf_level = conf_level

        self.notes = notes

        if meta_data is None:
            self.meta_data = {}
        else: self.meta_data = meta_data
        self.meta_data['PATCHES_CRE_DATE'] = my_cdate()

    def __len__(self):
        return len(self.patches)

    def get_single_patch_info(self,patch_i,wavelet,norm):
        patch = self.patches[patch_i]
        info = {}
        info['area'] = patch.area
        info['centroid'] = (patch.centroid.x,patch.centroid.y)
        
        x,y = patch.exterior.xy
        max_x_dist = max(x)-min(x)
        max_y_dist = max(y)-min(y)
        info['max_x_dist'] = max_x_dist
        info['max_y_dist'] = max_y_dist
        
        # First point is repeated
        info['n_points'] = len(x)-1
        
        mask = self.extract_mask(patch_i)
        n_area_points = np.sum(mask)
        info['n_area_points'] = n_area_points
        masked_power = mask*wavelet.norm_power(norm) 
        info['max_power'] = np.max(masked_power)
        info['mean_power'] = np.sum(masked_power)/n_area_points

        return info

    def get_patch_info(self,wavelet,freq_band=[],norm='leahy'):

        if len(freq_band) != 0:
            # This will be interpreted as upper limit
            # Remeber!!! highest index, lowest frequency
            if len(freq_band) == 1:
                upp_index = np.argmin(abs(wavelet.freqs-freq_band[1]))
                low_index = 0
            elif len(freq_band) == 2:
                low_index = np.argmin(abs(wavelet.freqs-freq_band[1]))
                upp_index = np.argmin(abs(wavelet.freqs-freq_band[0]))
            else:
                raise ValueError('freq_band must contain 2 or less values')
        else:
            low_index = 0
            upp_index = len(wavelet.freq) + 1

        print('low_index',low_index)
        print('upp_index',upp_index)

        result = []
        for patch_i in range(len(self.patches)):
            centroid_freq_index = self.patches[patch_i].centroid.y
            print('Centroid freq index',centroid_freq_index)
            if centroid_freq_index >= low_index and centroid_freq_index <= upp_index:
                print('Accepted!')
                result += [self.get_single_patch_info(patch_i=patch_i,wavelet=wavelet,norm=norm)]
        self.patch_info = result
  
    @classmethod
    def extract_patches(cls, wavelet, bkg_power=1., conf_level = None):
        '''
        Extracts significance patches (contours) from wavelet power 
        spectrum given a noise level and a confidence level

        When the noise level is given, this can be either a single 
        number or an array. In the latter case, this is assumed to 
        correspond to increasing frequency and its length must be equal
        to the length of the wavelet frequencies.
        '''

        # Converting eventual string arguments
        if type(bkg_power) == str: power_level = eval(bkg_power)
        if type(conf_level) == str: conf_level = eval(conf_level)

        # Extending bkg_power
        if type(bkg_power) in [float,int]:
            bkg_power = np.array([bkg_power for i in range(len(wavelet.freqs))])
        elif type(bkg_power) == np.ndarray:
            if len(bkg_power) != len(wavelet.freqs):
                raise ValueError('bkg_power must have same wavelet.freqs dimension')

        # Extracting normalized power (norm=2/var)
        wt_norm_power = wavelet.norm_power()
        nf, nt = wt_norm_power.shape

        # Finding point-wise contour significance
        # -------------------------------------------------------------
        # Extending power level array along the time direction
        bkg_power_2D = np.transpose(np.tile(np.flip(bkg_power),(nt,1)))

        #dof = 2
        #if not np.iscomplexobj(wavelet.wt): dof=1
        # !!! It is assumed that the wavelet power is distributed as 
        # a chi2 distribution with 2 degrees of freedom.

        if conf_level is not None:
            chi2_value = chi2.ppf(conf_level,df=2)
        else:
            chi2_value = 1.

        # With these normalizations, the expectration value of 
        # normalized power with specified confidence level is 1
        norm_power = wt_norm_power/bkg_power_2D/chi2_value

        contours = measure.find_contours(norm_power, 1)
        # -------------------------------------------------------------

        # Extracting and smoothing patches
        # -------------------------------------------------------------
        patches = []
        for contour in contours:
            
            local_y, tmp_local_x = map(list, zip(*contour))
            if len(tmp_local_x) > 51:
                local_x = savgol_filter(tmp_local_x, 51, 3)
            elif len(tmp_local_x) > 25:
                local_x = savgol_filter(tmp_local_x, 25, 3)
            elif len(tmp_local_x) > 11:
                local_x = savgol_filter(tmp_local_x, 11, 3)
            else:
                local_x = tmp_local_x
                
            xy = [(i,j) for i,j in zip(local_x,local_y)]
            
            if len(xy) >=3:
                patches += [Polygon(xy)]
        # -------------------------------------------------------------

        result = cls(patches=patches,nf=nf,nt=nt,verdicts=None,
        conf_level=conf_level,meta_data=None,notes=None)
        result.meta_data['PATCHES_CRE_MODE'] = 'Patches extracted from wavelet object'
        result.meta_data['CONF_LEVEL'] = conf_level

        return result

    def plot_patches(self,ax=None,verdict=False):
        '''
        Plot patches
        '''

        if ax is None: 
            fig, ax = plt.subplots(figsize=(12,6))

        style = '-k'
        for i,p in enumerate(self.patches):
            px = p.boundary.xy[0]
            py = p.boundary.xy[1]
            if verdict:
                if self.verdicts[i]: 
                    style = '-k'
                else:
                    style = '--r'
            ax.plot(px,py,style)

        ax.set_ylim([self.nf,0])
        ax.set_xlim([0,self.nt])

        if ax is None: 
            return fig,ax

    def extract_indices(self,patch_i):
        '''
        It draws a grid based on minimum and maximum
        '''

        patch = self.patches[patch_i]
        
        # Making grid arrays
        x_grid = np.arange(int(patch.bounds[0]),int(patch.bounds[2])+1+1)
        y_grid = np.arange(int(patch.bounds[1]),int(patch.bounds[3])+1+1)
        
        x_indices = []
        y_indices = []
        for yi in y_grid:
            row = []
            for xi in x_grid:
                if patch.contains(Point(xi,yi)):
                    row += [xi]

            if len(row) != 0:
                y_indices += [yi]
                x_indices += [row]
            
        if len(y_indices) != len(x_indices):
            raise ValueError('y_indices and x_indices have different dimension')
            
        return x_indices, y_indices

    def extract_mask(self,patch_i,opt='wrf'):
        '''
        Creates a mask of zeros with the same dimension of the wavelet 
        transform object where only the points inside the specified patch
        will be one
        '''

        patch = self.patches[patch_i]
    
        if opt is None:
            patch_x_grid = np.arange(int(patch.bounds[0]),int(patch.bounds[2])+1+1)
            patch_y_grid = np.arange(int(patch.bounds[1]),int(patch.bounds[3])+1+1)
            
            mask = np.zeros((self.nf,self.nt))
            
            for yi in patch_y_grid:
                for xi in patch_x_grid:
                    if patch.contains(Point(xi,yi)): mask[yi,xi] = 1
        elif opt in ['wrf','matlab','shapely']:
            mask = extract_mask_cython(patch,self.nf,self.nt,opt=opt)

        return mask

    def plot_patch(self,sel=0,ax=None,slices=False):
    
        if ax is None: fig,ax = plt.subplots(figsize=(6,6))  

        if slices:
            x_ind,y_ind = self.extract_indices(sel)
            for i,yi in enumerate(y_ind):
                row = x_ind[i]
                x = row
                y = [yi for i in range(len(row))]
                ax.plot(x,y)
        ax.plot(self.patches[sel].boundary.xy[0],self.patches[sel].boundary.xy[1],'k--',lw=2)

        if ax is None: return fig,ax

    def evaluate_patch_single(self,patch_i,wavelet,bkg_power,tollerance):
        '''
        Evaluates a patch significance according to a given tollerance
        
        Each patch is "squeezed" along the time axes, i.e. the power 
        INSIDE each patch is averaged along the time axes. The power of
        each row will be then distributed according a new degree of 
        freedom specified by eq. 23 in Torrence and Compo.
        Each point of this averaged power is then compared with a certain
        confidence level and then if a percent equal to tollerance points
        are above the confidence level, then the patch is considered 
        validated.

        RETURNS
        -------
        verdict, patch_flags: tuple
            verdict is a simple boolean, True or False depending on the
            patch passing the test. patch_flags is a mask of the time
            averaged patch power, True when exceeding expectation value
            according to the confidence level.
        '''
        if len(wavelet.freqs) != len(bkg_power):
            raise ValueError('The backgound power must have the same wavelet.freqs dimension')
        
        if wavelet.family == 'morlet':
            gamma = 2.32
            dof_ori = 2
        elif wavelet.family == 'mexhat':
            gamma = 1.43
            dof_ori = 1 
            
        mask = self.extract_mask(patch_i)
        # sum_mask is a 1D array with len = wavelet.freqs
        sum_mask = np.sum(mask,axis=1)
        y_mask = sum_mask != 0
        
        # In this case the masks is empty
        if np.sum(mask) == 0: 
            return False,[False]
            
        masked_power = (mask*wavelet.norm_power())[y_mask]
        # Number of wavelet power time bins inside the patch per
        # row (scale/freq) inside the patch 
        n_t_bins = (sum_mask[y_mask]).astype(int)

        # Computing corrected degree of freedom according to Eq. 23 in 
        # Torrence and Compo 1998
        dof = dof_ori*np.sqrt(1+(wavelet.tres*n_t_bins/gamma/wavelet.scales[y_mask])**2)
            
        # Averaging power over time
        average_power = np.sum(masked_power,axis=1)/n_t_bins
            
        norm = np.flip(bkg_power)[y_mask]*chi2.ppf(self.conf_level,df=dof)#/dof
        norm_power = average_power/norm
        
        patch_flags = norm_power>1
        
        verdict = False
        if sum(patch_flags) >= tollerance*len(patch_flags): 
            verdict = True
            
        return verdict,patch_flags

    def evaluate_patches(self,wavelet,bkg_power,tollerance=0.95,show_progress=False,
        multi_process=False):

        self.meta_data['TOLLERANCE'] = 0.95

        if multi_process:
            partial_evaluate = functools.partial(self.evaluate_patch_single,wavelet = wavelet,\
                bkg_power=bkg_power,tollerance=tollerance)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(partial_evaluate,[p for p in range(len(self.patches))]))
            self.verdicts = [r[0] for r in results]
            self.flags = [r[1] for r in results]

        else:
            verdicts = []
            flags = []

            for p in range(len(self.patches)):
                if show_progress: print('Processing patch {}/{}'.format(p+1,len(self.patches)))
                result = self.evaluate_patch_single(patch_i=p,wavelet=wavelet,bkg_power=bkg_power,\
                tollerance=tollerance)
                verdicts += [result[0]]
                flags += [result[1]]

            self.verdicts = verdicts
            self.flags = flags

        print('Patches Evaluated')


    def save(self,file_name='wavelet_patches.pkl',fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        if type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name
        
        try:
            with open(file_name,'wb') as output:
                pickle.dump(self,output, protocol=4)
            print('WaveletTransform Patches saved in {}'.format(file_name))
        except Exception as e:
            print(e)
            print('Could not save WaveletTransform Patches')

    @staticmethod
    def load(file_name,fold=pathlib.Path.cwd()):

        if not type(file_name) in [type(pathlib.Path.cwd()),str]:
            raise TypeError('file_name must be a string or a Path')
        elif type(file_name) == str:
            file_name = pathlib.Path(file_name)
        if file_name.suffix == '':
            file_name = file_name.with_suffix('.pkl')

        if type(fold) == str:
            fold = pathlib.Path(fold)
        if type(fold) != type(pathlib.Path.cwd()):
            raise TypeError('fold name must be either a string or a path')
        
        if not str(fold) in str(file_name):
            file_name = fold / file_name

        if not file_name.is_file():
            raise FileNotFoundError(f'{file_name} not found'.format(file_name))
        
        with open(file_name,'rb') as infile:
            wt = pickle.load(infile)
        
        return wt   

from saturnx.wavelets.functions import cwt,comp_scales