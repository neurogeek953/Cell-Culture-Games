#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:17:08 2019

@author: TEB
"""

from World import Time
import numpy as np

class Hazard(Time):
    
    def __init__(self):
        
        self.Disturbance = None
        self.TI = None
    
    def setup_hazard(self, time_of_experiment, sigma = 2.):
        
        magnitude = np.random.normal(0., sigma, 1)
        th = float(np.random.random(1) * time_of_experiment)
        th = round(th, 3)
        self.Disturbance = round(magnitude[0], 2)
        self.TI = th
        print('Hazard set to occur at: ', th)
    
    def execute_hazard(self, time):
        
        if self.TI == time:
            
            print("Hazard Occured: ", self.TI)
            return self.Disturbance
        
        else:
            
            return 0.

class HazardSet(Hazard):
    
    def __init__(self):
        
        self.NbHazards = None
        self.HSet = None
        self.NbOccurences = 0
        self.CurrentDisturbance = None
        self.NbHeatHazards = 0
        self.ProbabilityHeatHazard = 0.
        self.MaxHeatHazardFrequency = 0.
    
    def plan_hazards(self, time_of_experiment, nb_hazards):
        
        ### Plans a set of random magnitude hazards occuring at random times occuring along a time axis ###
        
        self.NbHazards = int(nb_hazards)
        tmp = np.zeros((2, self.NbHazards))
        tmp2 = Time()
        tmp2 = tmp2.Timestep
        
        for i in range(0, self.NbHazards):
            
            h = Hazard()
            h.setup_hazard(time_of_experiment = time_of_experiment)
            tmp[0, i] = h.TI
            tmp[1, i] = h.Disturbance
            
            if h.Disturbance > 0.:
                
                self.NbHeatHazards = self.NbHeatHazards + 1
                tmp3 = self.NbHeatHazards * tmp2 / h.TI
                
                if self.MaxHeatHazardFrequency < tmp3:
                    
                    self.MaxHeatHazardFrequency = tmp3
        
        tmp = tmp.T
        tmp = tmp[tmp[:, 0].argsort()]
        self.HSet = tmp.T
        self.ProbabilityHeatHazard =  self.NbHeatHazards * tmp2 / time_of_experiment
        
    
    def execute_hazards(self, time):
        
        if time in self.HSet[0, :] and self.NbOccurences < self.NbHazards:
            
            print("Hazard Occured: ", time)
            self.CurrentDisturbance = self.HSet[1, self.NbOccurences]
            return self.CurrentDisturbance
        
        else:
            
            return 0.
    
    def track_next_hazard(self, time):
        
        if time in self.HSet[0, :] and self.NbOccurences < self.NbHazards:
            
            self.NbOccurences = self.NbOccurences + 1


if __name__ == "__main__":
    
    # Test Hazard
    h1 = Hazard()
    h1.setup_hazard(5)
    print(h1.execute_hazard(h1.TI))
    print(h1.execute_hazard(3))
    
    # Test HazardSet
    hset =  HazardSet()
    hset.plan_hazards(5, 5)
    print(hset.execute_hazards(hset.HSet[0,0]))
    hset.track_next_hazard(hset.HSet[0,0])
    print(hset.execute_hazards(hset.HSet[0,1]))
    hset.track_next_hazard(hset.HSet[0,1])
    print(hset.execute_hazards(hset.HSet[0,3]))
    hset.track_next_hazard(hset.HSet[0,2])
    print(hset.execute_hazards(hset.HSet[0,2]))
    hset.track_next_hazard(hset.HSet[0,3])
    print(hset.execute_hazards(hset.HSet[0,4])) 
    
    
    