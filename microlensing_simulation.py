# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:30:15 2016

@author: danielgodinez
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from stats_computation import compute_statistics


def simulate_microlensing(time, mag, magerr, zp = 25):
    """Simulates a microlensing event given the inserted flat lightcurve. The angular 
    impact parameter is chosen from a random distribution between 0.0 and 1.5.
    Likewise the time of maximum amplification t_0 is chosen from a normal 
    distribution with a mean of 20 days and a standard deviation of 5 days and 
    the timescale t_e from a uniform distribution between 0.0 and 30 days. 
    These parameter spaces were determined given an analysis of the OGLE III
    microlensing survey. See: The OGLE-III planet detection efficiency from six 
    years of microlensing observations (2003 to 2008), (Y. Tsapras et al (2016)).
    
    :param time: the time-varying data of the lightcurve. Must be an array.   
    :param mag: the time-varying intensity of the object. Must be an array.   
    :param magerr: photometric error for the intensity. Must be an array.
    :param zp: zeropoint of the instrument that measured the inserted lightcurve. Default is 25.
    
    :return: the function will return the simulated lightcurve (mjd, mag, magerr)
    as well as the following simulation parameters: u_0, t_0, t_e, baseline, f_b, f_s.
    """
    
    mjd, mag, magerr = remove_bad(mjd, mag, magerr)


    #Simulates microlensing event but rejects events with poor signal!
    n=0
    if n == 0:
        
        u_0 = np.random.uniform(low = 0, high = 1.5, size = 1)
        t_0 = np.random.choice(time)
        t_e = np.random.normal(loc = 30, scale = 10.0, size = 1)
   
        g = np.random.uniform(0,10)
    
        u_t = np.sqrt(u_0**2 + ((mjd - t_0) / t_e)**2)
        magnification = (u_t**2 + 2.) / (u_t * np.sqrt(u_t**2 + 4.))
    
        flux = 10**((mag - zp) / -2.5)
        baseline = np.mean(mag)
        flux_base = np.median(flux)
        flux_noise = flux-flux_base
        f_s = flux_base / (1 + g)
        f_b = g * f_s
    
        flux_obs = f_s*magnification + f_b+flux_noise
        microlensing_mag = zp - 2.5*np.log10(flux_obs)
        
        #Conditions used to ensure microlensing signal is injected into the lightcurve
        signal_measurements = np.argwhere((mjd > (t_0 - t_e)) & (mjd < (t_0 + t_e)))
        
        amp1 = np.abs(np.max(mag[signal_measurements]) - np.min(mag[signal_measurements]))
        amp2 = np.abs(np.max(magnitude[signal_measurements] - np.min(magnitude[signal_measurements])))
        m1 = np.mean(magnitude[signal_measurements])
        m2 = np.mean(mag[signal_measurements])
        to_index = np.argwhere(mjd == t_0)
        
        list1 = []
        
        #used to set magnification threshold
        for i in signal_measurements:
            value = (magnitude[i] - mag[i]) / magerr[i]
            list1.append(value)
                                                    
        list1 = np.array(list1)
                                                        
        if len(np.argwhere(list1 >= 3)) > 0 and m2 < (m1 - 0.05) and ((magnitude[to_index] - mag[to_index])/magerr[to_index]) >= 3.5 and len(np.argwhere(list1 > 3)) >= 0.33*len(signal_measurements) and (1.0/u_0) > (f_s/f_b):
                
            n = n + 1

    else:
        return mjd, microlensing_mag, magerr, u_0, t_0, t_e, baseline, f_b, f_s

def plot_microlensing(time, mag, magerr):
    """Plots a simulated microlensing event from an inserted flat lightcurve.
    
    :param time: the time-varying data of the lightcurve. Must be an array.  
    :param mag: the time-varying intensity of the object. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :return: the function will return a plot of the microlensing lightcurve.
    :rtype: plot
    """    
    
    intensity = simulate_microlensing(time, mag, magerr)[1]
    
    plt.errorbar(time, intensity, yerr = magerr, fmt='o')
    plt.gca().invert_yaxis
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Simulated Microlensing')
    plt.show()
    
def microlensing_statistics(time, mag, magerr):
    """Simulates a microlensing event given an inserted lightcurve, and calculates
    various lightcurve statistics from the compute_statistics function.

    :param time: the time-varying data of the lightcurve. Must be an array.
    :param mag: the time-varying intensity of the object. Must be an array.
    :param magerr: photometric error for the intensity. Must be an array.
    
    :return: the function will return the lightcurve statistics.
    :rtype: array 
    """
            
    microlensing_mag = simulate_microlensing(time, mag, magerr)
    stats = compute_statistics(microlensing_mag, magerr)
    
    return stats

def remove_bad(mjd, mag, magerr):
    """Function to remove bad photometric points"""
    
    bad = np.where(np.isfinite(magerr) == True)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(time, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(np.isnan(magerr) == True)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(time, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(magerr == 0)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(time, bad)
    mag = np.delete(mag, bad)
    
    bad = np.where(mag == 0)
    magerr = np.delete(magerr, bad)
    mjd = np.delete(time, bad)
    mag = np.delete(mag, bad)

    return mjd, mag, magerr
