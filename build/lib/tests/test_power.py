from ..lightcurve import Lightcurve,LightcurveList
from ..power import PowerSpectrum,PowerList
from ..utilities import timmer_koenig
from time import ctime
import numpy as np
import pandas as pd
import pytest


class TestPower:

    def setup_class(self,t_res=0.01,time_bins=1701,cr=20000,std=0.5):

        self.time_bins = time_bins
        self.cr = cr
        self.std = std
        self.t_res = t_res

        self.t_dur = self.time_bins*self.t_res
        self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
      
        self.history_key = 'CREATION_DATE'
        self.history_value = ctime()
        self.history = {self.history_key:self.history_value}
        self.notes_key = ctime()
        self.notes_value = 'Created for testing purposes'
        self.notes = {self.notes_key:self.notes_value}

        self.lc = Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)

        self.nlc = 50
        lc_list = []
        for i in range(self.nlc):
            t,counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr,self.std)
            t += self.t_dur*i+self.t_res
            self.lc = Lightcurve(t,counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(t,counts,0.5,10,
        self.notes,self.history)]
        self.lcs = LightcurveList(lc_list)

        lc_list = []
        for i in range(self.nlc):
            t,counts=timmer_koenig(self.t_dur,self.t_res,0.1,(np.random.random()+1)*200,self.std)
            t += self.t_dur*i+self.t_res
            self.lc = Lightcurve(t,counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(t,counts,0.5,10,
        self.notes,self.history)]
        self.lcs2 = LightcurveList(lc_list)       

    def test_init_empty(self):
        power = PowerSpectrum()
        assert isinstance(power,PowerSpectrum)
        assert isinstance(power,pd.DataFrame)
        assert 'power' in power.columns
        assert 'freq' in power.columns
        assert len(power.power) == 0
        assert len(power.freq) == 0
        assert power.leahy_norm is None
        assert power.rms_norm is None
        assert power.poi_level is None
        assert power.low_en is None
        assert power.high_en is None
        assert power.notes == {}
        for key,value in power.history.items():
            assert value == 'Power spectrum object initialized'
        assert power.df is None
        assert power.nf is None
        assert power.a0 is None
        assert power.cr is None

    def test_make_power_single(self):
        power = PowerSpectrum.from_lc(self.lc)

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)
        assert isinstance(power,pd.DataFrame)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0',power.a0,'tot_counts',self.lc.tot_counts)     
        assert np.isclose(power.a0,self.lc.tot_counts,atol=self.lc.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df',power.df,'1/T',1./self.lc.texp)
        assert np.isclose(power.df,1./self.lc.texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2',power.freq[power.freq>0].iloc[-1],'1/2dt',1./(2*self.lc.tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lc.tres),atol=power.df,rtol=0)

        # Powers must be not NaN in all possible normalizations
        assert not power.freq.isnull().values.any()
        assert not power.power.isnull().values.any()

        leahy = power.leahy()
        rms = leahy.rms()
        assert not leahy.freq.isnull().values.any()
        assert not leahy.power.isnull().values.any()
        assert not rms.freq.isnull().values.any()
        assert not rms.power.isnull().values.any()


        # Parseval theorem (about fft computation)
        assert np.isclose(np.sum(self.lc.counts**2),np.sum(power.power)/len(self.lc))

    def test_make_power_multi(self):
        powers = PowerSpectrum.from_lc(self.lcs)

        assert isinstance(powers,PowerList)
        assert isinstance(powers,list)
        for p in powers:
            assert isinstance(p,PowerSpectrum)
            assert isinstance(p,pd.DataFrame)
            assert 'freq' in p.columns 
            assert 'power' in p.columns 
            assert len(p.freq) != 0
            assert len(p.power) != 0 

        power = powers.average_leahy()

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0',power.a0,'tot_counts',self.lcs.mean.tot_counts)     
        assert np.isclose(power.a0,self.lcs.mean.tot_counts,atol=self.lcs.mean.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df',power.df,'1/T',1./self.lcs[0].texp)
        assert np.isclose(power.df,1./self.lcs[0].texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2',power.freq[power.freq>0].iloc[-1],'1/2dt',1./(2*self.lcs[0].tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lcs[0].tres),atol=power.df,rtol=0)

        # Parseval theorem (about fft computation)
        sumt2 = np.sum(self.lcs.mean.counts**2)
        suma2 = np.sum(power.power*self.lcs.mean.tot_counts/2.)
        print('sumt2',sumt2,'suma2/N',suma2/len(self.lcs[0]))
        assert np.isclose(sumt2,suma2/len(self.lcs[0]))

    def test_make_power_multi2(self):
        powers = PowerSpectrum.from_lc(self.lcs2)

        assert isinstance(powers,PowerList)

        power = powers.average_leahy()

        # It must return an instance of the PowerSpectrum class
        assert isinstance(power,PowerSpectrum)

        # Frequency bins and time bins must be the same
        assert len(power.freq) == self.time_bins

        # Total counts must be the same   
        print('a0_diff',power.a0,'tot_counts_diff',self.lcs2.mean.tot_counts)     
        assert np.isclose(power.a0,self.lcs2.mean.tot_counts,atol=self.lcs.mean.tot_counts/1000.,rtol=0)

        # Frequency resolution must be the same
        print('df_diff',power.df,'1/T_diff',1./self.lcs2[0].texp)
        assert np.isclose(power.df,1./self.lcs2[0].texp,atol=power.df/1000,rtol=0)

        # Nyquist frequency myst be the same
        if len(self.lc)%2!=0:
            print('fn2_diff',power.freq[power.freq>0].iloc[-1],'1/2dt_diff',1./(2*self.lcs2[0].tres))
            assert np.isclose(power.freq[power.freq>0].iloc[-1],1./(2*self.lcs2[0].tres),atol=power.df,rtol=0)

        # Parseval theorem (about fft computation)
        sumt2 = np.sum(self.lcs2.mean.counts**2)
        suma2 = np.sum(power.power*self.lcs2.mean.tot_counts/2.)
        print('sumt2_diff',sumt2,'suma2/N_diff',suma2/len(self.lcs2[0]))
        assert np.isclose(sumt2,suma2/len(self.lcs2[0]),atol=sumt2/10.,rtol=0.)
    
    def test_sub_level(self):
        power = PowerSpectrum.from_lc(self.lc) 
        assert not power.a0 is None
        assert not power.sub_poi(2).a0 is None
        assert power.a0 == power.sub_poi(2).a0

    def test_leahy(self):
        power = PowerSpectrum.from_lc(self.lc)
        leahy = power.leahy()
        assert isinstance(leahy,PowerSpectrum)
        assert not leahy.leahy_norm is None
        assert np.array_equal(leahy.freq,power.freq)

        assert np.array_equal(leahy.power,leahy.leahy().power)
    
    def test_rms(self):
        power = PowerSpectrum.from_lc(self.lc)
        assert not power.power is None
        leahy = power.leahy()
        assert not leahy.power is None
        rms = leahy.rms()
        rms2 = power.rms()
        assert isinstance(rms,PowerSpectrum)
        assert not rms.rms_norm is None
        assert np.array_equal(rms.freq,leahy.freq)
        #assert np.array_equal(rms.rms(self.lc.cr).power,rms.power)

    def test_rebin(self):
        power = PowerSpectrum.from_lc(self.lc)
        rf = 2
        rebin = power.rebin(rf)
        assert isinstance(rebin,PowerSpectrum)  
        pos_freq = power.freq[power.freq>0]
        assert rebin.a0 == power.a0
        if len(pos_freq)%rf==0:
            assert len(rebin.freq) == int(len(pos_freq)/2)+1
        else:
            assert len(rebin.freq) == int(len(pos_freq)/2)+1+1

    #def test_plot(self):
    #    power = PowerSpectrum.from_lc(self.lc)
    #    rebin = power.rebin(2)
    #    rebin.plot()   
    
    def test_comp_frac_rms(self):
        power = PowerSpectrum.from_lc(self.lc).leahy().rms()
        leahy = PowerSpectrum.from_lc(self.lc).leahy()
        rms, srms = power.comp_frac_rms()
        rms2 = (rms/100.)**2
        rms2corr = rms2+1

        # Fractional rms must be equal to standard deviation devided by mean
        assert np.isclose(rms/100.,self.lc.counts.std()/self.lc.counts.mean(),atol=rms/100./100.,rtol=0.)

        # Total rms squared divided by the mean should be equal to the fractional RMS squared + 1 
        assert np.isclose(rms2+1,(self.lc.rms/100)**2/(self.lc.counts.mean()**2),atol=rms2/100.,rtol=0)

        # Total rms squared must be euqal  to a certain linear comination of Leahy power
        if len(self.lc) %2 ==0:
            lin_comb = leahy.a0/(len(self.lc)**2)*(leahy.power[leahy.freq>0].sum()+
            leahy.power[0]*0.5+leahy.power[leahy.freq>0].iloc[-1]*0.5)
        else:
            lin_comb = leahy.a0/(len(self.lc)**2)*(leahy.power[leahy.freq>0].sum()+
            leahy.power[0]*0.5)
        assert np.isclose((self.lc.rms/100)**2,lin_comb,atol=lin_comb/100.,rtol=0)

        print((self.lc.rms/100.)**2.,(self.lc.tot_counts/len(self.lc))**2.*rms2corr)
        print('-'*50)
        print((self.lc.rms/100.)**2.,self.lc.counts.var(),self.lc.counts.mean()**2)
        print((self.lc.tot_counts/len(self.lc))**2.*rms2,(self.lc.tot_counts/len(self.lc))**2)
        print('-'*50)
        print(rms/100.,self.lc.counts.std()/self.lc.counts.mean())
        #assert False

    def test_save_load(self):
        leahy = PowerSpectrum.from_lc(self.lc).leahy()
        leahy.save()

        power = PowerSpectrum.load()
        assert isinstance(power,PowerSpectrum().__class__)
        assert not power.leahy_norm is None
        assert power.rms_norm is None
        assert power.history['FILE_NAME'] == 'power_spectrum.pkl'

class TestPowerList:

    def setup_class(self):
        self.time_bins = 1000
        self.cr = 200
        self.t_res = 0.01
        self.t_dur = self.time_bins*self.t_res
        self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
      
        self.history_key = 'CREATION_DATE'
        self.history_value = ctime()
        self.history = {self.history_key:self.history_value}
        self.notes_key = ctime()
        self.notes_value = 'Created for testing purposes'
        self.notes = {self.notes_key:self.notes_value}

        self.lc = Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)

        self.nlc = 20
        lc_list = []
        for i in range(self.nlc):
            self.t,self.counts=timmer_koenig(self.t_dur,self.t_res,0.1,self.cr)
            self.lc = Lightcurve(self.t,self.counts,0.5,10,
            self.notes,self.history)
            lc_list += [Lightcurve(self.t,self.counts,0.5,10,
        self.notes,self.history)]
        self.lcs = LightcurveList(lc_list)
        self.pl = PowerSpectrum.from_lc(self.lcs)             

    def test_init(self):
        assert isinstance(self.pl,PowerList)

    def test_getitem(self):
        assert isinstance(self.pl[3],PowerSpectrum)

    def test_ave_leahy(self):
        ave_leahy = self.pl.average_leahy()
        assert isinstance(ave_leahy,PowerSpectrum)
        assert len(ave_leahy.freq)==len(self.pl[0].leahy().freq)
        assert ave_leahy.weight == len(self.pl)