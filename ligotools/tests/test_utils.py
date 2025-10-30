import pytest
import numpy as np
from ligotools.utils import whiten, reqshift
from scipy.interpolate import interp1d


def test_white_func():
   N = 1000
   dt = 1 / 2000
   t = np.arange(0, N * dt, dt)
   s = np.sin(2 * np.pi * 100 * t)  


   freq = np.fft.rfftfreq(N, dt)
   psd_vals = np.ones_like(freq)
   interp_psd = interp1d(freq, psd_vals)
  
   whitened_strain = whiten(s, interp_psd, dt)
  
   assert whitened_strain.shape == s.shape
  
 def test_reqshift_func():
   N = 1000
   sample_rate = 2000
   t = np.arange(0, N * (1/sample_rate), (1/sample_rate))
   s = np.sin(2 * np.pi * 100 * t)  


   shifted_data = reqshift(data, fshift=50, sample_rate=sample_rate)
  
   assert shifted_data.shape == data.shape
