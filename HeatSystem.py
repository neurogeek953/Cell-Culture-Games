#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:39:21 2019

@author: TEB
"""

import matplotlib.pyplot as plt

from World import Object


class Heater(Object):
    
    def __init__(self,
                 heater_power,
                 length,
                 width,
                 height,
                 position_x,
                 position_y,
                 position_z,
                 initial_temperature,
                 label = None):
        
        self.Object = Object(position_x, position_y, position_z, initial_temperature)
        self.XPosition = self.Object.XPosition
        self.YPosition = self.Object.YPosition
        self.ZPosition = self.Object.ZPosition
        self.Distances = self.Object.Distances
        self.Label = self.Object.Label
        self.statusW = "OFF" # status in text useful check
        self.status = 0
        self.Power = heater_power # Power of the heater is an input in the simulation.
        self.Volume = length * width * height
        self.Area = 2 * height * width + 2 * height * length + 2 * length * width
        self.Height = height
        self.Width = width
        self.Length = length
        self.EffectiveArea = length * width
        self.Kind = "Heater"
        self.Temperature = .0
        self.Memory = []
    
    def label_heater(self, label):
        
        self.Object.label_object(label)
        self.Label = self.Object.Label
        

""" Two classes one for on off style heater the other intersity of heat regulator  """

class OnOffHeater(Heater):
    
    """ Heater with a constant power, which can be turned on and off """
    
    def __init__(self,
                 heater_power,
                 length,
                 width,
                 height,
                 position_x,
                 position_y,
                 position_z,
                 initial_temperature,
                 label = None):
        
        self.Heater = Heater(heater_power,
                             length,
                             width,
                             height,
                             position_x,
                             position_y,
                             position_z,
                             initial_temperature,
                             label = None)
        
        self.Object = self.Heater.Object
        self.XPosition = self.Heater.XPosition
        self.YPosition = self.Heater.YPosition
        self.ZPosition = self.Heater.ZPosition
        self.Distances = self.Heater.Distances
        self.Label = self.Heater.Label
        self.statusW = self.Heater.statusW # status in text useful check
        self.status = self.Heater.status
        self.Power = self.Heater.Power # Power of the heater is an input in the simulation.
        self.Volume = self.Heater.Volume
        self.Area = self.Heater.Area
        self.Height = self.Heater.Height
        self.Width = self.Heater.Width
        self.Length = self.Heater.Length
        self.EffectiveArea =  self.Heater.EffectiveArea
        self.Kind = "On/Off Switch Heater"
        self.Temperature = .0
        self.Memory = []
        
    def activate(self):
        
        self.statusW = "ON"
        self.status = 1
        # print(self.statusW)
    
    def deactivate(self):
        
        self.statusW = "OFF"
        self.status = 0
        # print(self.statusW)
        
    def heat(self):
        
        if self.status == 1:
            
            self.Temperature = self.Power / self.EffectiveArea
            
        else:
            
            self.Temperature = 0
    
    def make_heater_history(self):
        
        self.Memory.append(self.status)


class IntensityHeater(Heater):
    
    """ Heater with a constant power, which can be turned on and off """
    
    def __init__(self,
                 heater_start_power,
                 heater_maxpower,
                 heater_minpower,
                 length,
                 width,
                 height,
                 position_x,
                 position_y,
                 position_z,
                 initial_temperature,
                 label = None):
        
        self.Heater = Heater(heater_power = heater_start_power,
                             length = length,
                             width = width,
                             height = height,
                             position_x = position_x,
                             position_y = position_y,
                             position_z = position_z,
                             initial_temperature = initial_temperature,
                             label = None)
        
        self.Object = self.Heater.Object
        self.XPosition = self.Heater.XPosition
        self.YPosition = self.Heater.YPosition
        self.ZPosition = self.Heater.ZPosition
        self.Distances = self.Heater.Distances
        self.Label = self.Heater.Label
        self.statusW = self.Heater.statusW # status in text useful check
        self.status = self.Heater.status
        self.MaxPower = heater_maxpower
        self.MinPower = heater_minpower
        self.Power = self.Heater.Power # Power of the heater is an input in the simulation.
        self.Volume = self.Heater.Volume
        self.Area = self.Heater.Area
        self.Height = self.Heater.Height
        self.Width = self.Heater.Width
        self.Length = self.Heater.Length
        self.EffectiveArea =  self.Heater.EffectiveArea
        self.Kind = "Up/Down Switch Heater"
        self.Temperature = .0
        self.Memory = []
        
    def activate(self):
        
        self.statusW = "ON"
        self.status = 1
        # print(self.statusW)
    
    def deactivate(self):
        
        self.statusW = "OFF"
        self.status = 0
        # print(self.statusW)
        
    def increase_power(self):
        
        if self.Power < self.MaxPower - 0.01:
            
            self.Power = self.Power + 0.01
            # print('Power Up')
    
    def decrease_power(self):
        
        if self.Power > self.MinPower + 0.05:
            
            self.Power = self.Power - 0.05
            # print('Power Down')
        
    def heat(self):
        
        if self.status == 1:
            
            self.Temperature = self.Power / self.EffectiveArea
            
        else:
            
            self.Temperature = 0
    
    def make_heater_history(self):
        
        self.Memory.append(self.Power)
        
            
    

    
if __name__ == "__main__":
    

    h1 = OnOffHeater(1000, 10, 10, 10, 0, 0, 0, 30)
    h1.activate()
    h1.heat()
    print(h1.Power)
    print(h1.Temperature)
    h1.deactivate()
    print(h1.Temperature)
    
    
    h2 = IntensityHeater(10, 20, 0, 10, 10, 10, 0, 0, 0, 30)
    h2.activate()
    h2.heat()
    print(h2.Power)
    print(h2.Temperature)
    
    for i in range(0,10):
        
        h2.increase_power()
        h2.heat()
        print(h2.Power)
        print(h2.Temperature)
        h2.make_heater_history()
    
    for i in range(0,5):
        
        h2.decrease_power()
        h2.heat()
        print(h2.Power)
        print(h2.Temperature)
        h2.make_heater_history()
    
    h2.deactivate()
    print(h2.Temperature)
    
    print(h2.Memory)
    plt.plot(h2.Memory)
    