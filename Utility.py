#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:20:38 2019

@author: TEB
"""

import numpy as np

def car(v, seq):
    
    """Function for Common Average Reference (CAR) Filtering (Unused)"""
    
    out = v - np.mean(seq)
    return out

def amp(v, a):
    
    """ Amplification of Output (Unused)"""
    
    out = v * a
    return out

def event_detection(v, threshold):
    
    """Detects Spikes"""
    
    if v > threshold:
        
        return 1
        
    else:
        
        return 0

def estimate_FR(events, t, time_window):
    
    """Estimate the output Fire rate within a time window"""
    
    if t < time_window:
        
        nb_events = sum(events[0:t])
        
        if t == 0:
            
            t = 1
        
        FR = nb_events / time_window
        return FR
    
    else:
        
        nb_events = sum(events[t - time_window:t])
        FR = nb_events / time_window
        return FR

def estimate_Amplitude(signal, t, time_window):
    
    if t < time_window:
        
        if t == 0:
            
            peak_amplitude = signal
            
        else:
            
            peak_amplitude = max(signal[0:t]) - min(signal[0:t])
    
    else:
        
        peak_amplitude = max(signal[t - time_window:t]) - min(signal[t - time_window:t])
    
    if peak_amplitude == None:
        
        peak_amplitude = 0.
    
    return peak_amplitude
    

def MA_Filter(signal, window):
    
    "Filter the frequency change to have a better idea of what neurons do"
    
    FRs = []
    
    if window > len(signal):
        
        window = len(signal)
    
    for i in range(0, len(signal)):
        
        if i + window < len(signal):
            
            FR = sum(signal[i:i + window])
            FR = FR / window
            FRs.append(FR)
            
        else:
            
            FR = sum(signal[i - window:i])
            FR = FR / window
            FRs.append(FR)
        
    
    return FRs
            

if __name__ == "__main__":
    
    a = "Hello World !"
    print(a)

