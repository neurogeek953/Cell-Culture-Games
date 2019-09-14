#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:37:40 2019

@author: TEB
"""

# In jupyter do %matplotlib inline then %run Experiment2.py

from World import World
from CellCulture import CellCulture
from HeatSystem import IntensityHeater
from Hazard import HazardSet

if __name__ == "__main__":
    
    print('Simulation Initialisation...')
    
    initial_temperature = 30
    initial_time = 0
    cc_diameter = 0.1
    cc_height = 0.01
    cc_conduction = 0.2
    cc_position_x = 0
    cc_position_y = 0
    cc_position_z = 0.1
    cc_hp = 60000
    
    # Neural Network Parameters
    nb_afferent_neurons = 40
    nb_turnon_neurons = 30
    nb_turnoff_neurons = 30
    nb_decision_neurons = 140
    nb_efferent_neurons = 40
    
    cc = CellCulture(initial_temperature,
                     initial_time,
                     cc_diameter,
                     cc_height,
                     cc_conduction,
                     cc_position_x,
                     cc_position_y,
                     cc_position_z,
                     cc_hp)
    
    cc.create_NeuralNetwork(nb_afferent_neurons,
                             nb_turnon_neurons,
                             nb_turnoff_neurons,
                             nb_decision_neurons,
                             nb_efferent_neurons)
    
    cc.label_cell_culture('Cell Culture')
    
    # Heater
    heater_power = 0.15 # Power of 0.15 Watt [W]
    max_power = 0.2 # Max Power of 0.2 Watts [W]
    min_power = 0. # Min Power of 0 Watt [W]
    heater_length = 20
    heater_width = 10
    heater_height = 5
    heater_position_x = 0
    heater_position_y = 0
    heater_position_z = 0
    
    h = IntensityHeater(heater_power,
                        max_power,
                        min_power,
                        heater_length,
                        heater_width,
                        heater_height,
                        heater_position_x,
                        heater_position_y,
                        heater_position_z,
                        initial_temperature)
                        
    
    h.label_heater('Heater')
    
    # Parameters of the world
    initial_temperature = 20   # 20 C is Ambient temperature
    length = 20
    width = 20
    height = 20
    
    
    # Create the world
    w1 = World(initial_temperature, length, width, height)
    w1.add_object(h)
    w1.add_object(cc)
    
    # Experiment Parameters
    time_of_experiment = 80. # ten seconds
    nb_hazards = int(time_of_experiment / 8.) # One hazard for 10 seconds
    print("There will be ", nb_hazards, "hazards in this experiment")
    
    # Plan hazards
    hset = HazardSet()
    hset.plan_hazards(time_of_experiment, nb_hazards)
    
    print('Simulation Initialisation... Finished.')
    
    # Run the experiment with the parameters above
    h.activate()
    w1.run_experiment(time_of_experiment, cc, h, hset)
    h.deactivate()
    
    # Delete the objects at the end of the experiment
    w1.delete_object(h)
    w1.delete_object(cc)
    