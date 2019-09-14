#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 12:15:53 2019

@author: TEB
"""
import numpy as np
import matplotlib.pyplot as plt
from nengo.processes import Piecewise

from World import Time

class Stimulus(Time):
    
    def __init__(self, temperature, time_instance = 0):
        
        self.TimeObject = Time()
        self.Timestep = self.TimeObject.Timestep
        self.Temperature = temperature
        self.TimeInstance = time_instance
        self.StimulusMagnitude = None # Stimulus normalised between 1 and -1
        self.StimulusDic = {} # Nengo requires a dictionary to work
        self.NetworkInput = None # Input of the Nengo Simulator
        self.SpikeCount = None
        self.HPReward = 0 # The Hit Point loss or gain depending on temperature.
        
        if time_instance is None:
            
            self.MomentInTime = self.TimeObject.Clock
            
        else:
            
            self.MomentInTime = time_instance * self.Timestep
        
    
    def temperature2spikecount(self):
                
        if (self.Temperature < 1) or (self.Temperature > 30):
            
            self.SpikeCount = 0
            self.HPReward = -2 # This HP reward must correspond to the starting max HP. 
            # print("The cell culture dies instantly --> Neurons no longer spike")
        
        if (self.Temperature >= 1) and (self.Temperature < 10):
            
            self. SpikeCount = round(2.762 * (10 ** -4) * ((self.Temperature - 1) ** 3) + 0.6349 * (self.Temperature - 1) + 4.999)
            self.HPReward = -1
        
        if (self.Temperature >= 10) and (self.Temperature < 15):
            
            self.SpikeCount = round(0.0243 * ((self.Temperature - 10) ** 3) +  0.0075 * ((self.Temperature - 10) ** 2) + 0.702 * (self.Temperature - 10) + 10.9143)
            self.HPReward = 0
        
        if (self.Temperature >= 15) and (self.Temperature < 20):
            
            self.SpikeCount = round(- 0.159 * ((self.Temperature - 15) ** 3) +  0.3725 * ((self.Temperature - 15)  ** 2) + 2.602 * (self.Temperature - 15) + 17.653)
            self.HPReward = 0.5
        
        if (self.Temperature >= 20) and (self.Temperature < 22):
            
            self.SpikeCount = round(0.6552 * ((self.Temperature - 20)  ** 3) - 2.0126 * ((self.Temperature - 20)  ** 2) - 5.5985 * (self.Temperature - 20) + 20.0999)
            self.HPReward = 1
        
        if (self.Temperature >= 22) and (self.Temperature < 25):
            
            self.SpikeCount = round(- 0.2135 * ((self.Temperature - 22)  ** 3) +  1.9187 * ((self.Temperature - 22) ** 2) - 5.7864 * (self.Temperature - 22) + 6.0941)
            self.HPReward = -0.5
        
        if (self.Temperature >= 25) and (self.Temperature <= 30):
            
            self.SpikeCount = round(1.8272 * (10 ** -4) * ((self.Temperature - 25) ** 3) -  0.0027 * ((self.Temperature - 25) ** 2) + 0.0385 * (self.Temperature - 25) + 0.2389)
            self.HPReward = - 1
    
    def spikecount2stimulus(self):
        
        self.temperature2spikecount()
        self.StimulusMagnitude = 2 * (self.SpikeCount / 25) - 1
        self.StimulusDic = {self.TimeInstance: self.StimulusMagnitude}
        self.NetworkInput = Piecewise(self.StimulusDic)

class StimulusTimeSeries(Stimulus):
    
    # This class was not used in the simulation.
    
    def __init__(self, temperatures, time_instances):
        
        TimeObject = Time()
        self.Timestep = TimeObject.Timestep
        self.Temperatures = temperatures
        self.TimeInstances = time_instances
        self.PiecewiseStimulations = None # Dictionary containing one temperature element.
        self.StimuliDic = {}
        self.NetworkInputs = None
        self.HPRewards = []
        
        if time_instances is None:
            
            self.NumberOfInstances = self.TimeObject.Clock / self.Timestep
            self.MomentsInTime = np.linspace(0.0, self.TimeObject.Clock, self.NumberOfInstances)
               
        else:
            
            self.NumberOfInstances = np.size(time_instances)
            self.MomentsInTime = self.Timestep * time_instances
        
        self.StimuliMagnitude = np.zeros(self.NumberOfInstances)
        self.SpikeCounts = np.zeros(self.NumberOfInstances)
        
    def temps2stimuli(self):
        
        for i in range(0, self.NumberOfInstances):
            
            stimulus = Stimulus(self.Temperatures[i], i)
            stimulus.spikecount2stimulus()
            self.SpikeCounts[i] = stimulus.SpikeCount
            self.StimuliMagnitude[i] = stimulus.StimulusMagnitude
            self.HPRewards.append(stimulus.HPReward)
            c = stimulus.StimulusDic[i]
            c = c[0]
            stimulus.StimulusDic = {self.MomentsInTime[i]: c}
            
        self.StimuliDic = {**self.StimuliDic, **stimulus.StimulusDic}
        self.NetworkInputs = Piecewise(self.StimuliDic)
        
if __name__ == "__main__":
    
    # Konstantin Nikolic's Data Confidential
    Temperatures = np.array([1, 10, 15, 20, 22, 25, 30]);
    NumberOfSpikes = np.array([5, 11, 17, 23, 3, 1, 0]);
    
    temperatureInstances = np.linspace(-10, 40, 1000) # Temperatures in degree Celsius.
    timeInstances = np.linspace(0, 1000, 1000) # Time instances.
    
    sts = StimulusTimeSeries(temperatureInstances, timeInstances)
    sts.temps2stimuli()
    ts = sts.StimuliDic
    
    plt.figure()
    plt.plot(Temperatures, NumberOfSpikes, 'X', label = 'Original Data Points')
    plt.plot(temperatureInstances, sts.SpikeCounts, label = 'Extrapolated Data Points')
    plt.title('Spike Count as a Function of Temperature', {'fontsize': 20})
    plt.xlabel('Temperature in Degrees Celcius [C]', {'fontsize': 16})
    plt.ylabel('Spike Count in Units [U]', {'fontsize': 16})
    plt.legend()
    
    plt.figure()
    plt.plot(temperatureInstances, sts.StimuliMagnitude)
    # settings = {'fontsize': rcParams['axes.titlesize'], 'fontweight' : rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': loc}
    plt.title('Voltage as a Function of Temperature', {'fontsize': 20})
    plt.xlabel('Temperature in Degrees Celcius [C]', {'fontsize': 16})
    plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 16})
    
    plt.figure()
    plt.plot(temperatureInstances, sts.HPRewards)
    # settings = {'fontsize': rcParams['axes.titlesize'], 'fontweight' : rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': loc}
    plt.title('Hit Point Reward as a Function of Temperature', {'fontsize': 20})
    plt.xlabel('Temperature in Degrees Celcius [C]', {'fontsize': 16})
    plt.ylabel('Hit Point Reward in Arbitrary Units [A.U.]', {'fontsize': 12})
    
    