#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Wed Jan 30 19:13:24 2019
#
#@author: TEB
#"""

# In jupyter do %matplotlib notebook (play with the image) inline (just show the plot) and %run World.py
# Jupyter Matplotlib link below
# https://ipython.readthedocs.io/en/stable/interactive/magics.html
# To run first do: conda env create -f environment.yml


# Classic libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Note this import is under warning but is actually used in strings to generate 3D plots
from mpl_toolkits.mplot3d import Axes3D 

# NENGO
import nengo

# Import Utility Functions
from Utility import event_detection
from Utility import estimate_FR
from Utility import MA_Filter
from Utility import estimate_Amplitude

# Import Information Theory Tools
from MutualInformationAndEntropy import hop

# Find Different bands in which there a gain 

#### Parameters ####

WParam = 20 # 1000 steps 20 works
FR_Threshold = 0.45

class Time:
    
    def __init__(self, timestep = 0.001):
        # the default timestep of the nengo simulator is 1[ms]
        
        self.Timestep =  timestep
        self.Clock = 0.
    
    def update_time(self):
        
        self.Clock = round(self.Clock + self.Timestep, 3)
        # print('Run Time in Seconds',self.Clock)

class Object:
    
    def __init__(self,
                 position_x,
                 position_y,
                 position_z,
                 initial_temperature):
        
        self.Temperature = initial_temperature
        self.XPosition = position_x
        self.YPosition = position_y
        self.ZPosition = position_z
        self.Distance2Origin = 0
        self.Distances = {}
        self.Label = None
    
    def label_object(self, label):
        
        self.Label = label

class World(Time):
    
    # Heat conduction in air constant is 0.024
    # https://www.physicsclassroom.com/class/thermalP/Lesson-1/Rates-of-Heat-Transfer
    
    def __init__(self,
                 initial_temperature,
                 length,
                 width,
                 height,
                 conduction = 0.024,
                 absorption = 0.85):
        
        self.TimeObject = Time()
        self.Clock = self.TimeObject.Clock
        self.Timestep = self.TimeObject.Timestep
        self.Temperature = initial_temperature
        self.HeatAbsorption = absorption # Peltier effect
        self.TimeInstance = 0
        self.Height = height
        self.Width = width
        self.Length = length
        self.Volume = self.Length * self.Width * self.Height
        self.Area = 2 * self.Height * self.Width + 2 * self.Height * self.Length + 2 * self.Length * self.Width
        self.XOrigin = self.Length / 2
        self.YOrigin = self.Width / 2
        self.ZOrigin = 0
        self.Objects = {}
        self.NumberOfObjects = 0
        self.NumberOfHeaters = 0
        self.NumberOfCellCultures = 0
        self.Heaters = {}
        self.CellCultures = {}
        self.Temperatures = []
        self.HeatConductance = conduction
        self.TimeInstances = []
        
        # Neural Activity Description
        self.ANN_Events = [] # Stores the history of spiking events in the Afferent Neural Network spikes.
        self.ENN_Events = [] # Stores the history of spiking events in the Efferent Neural Network spikes.
        self.OnNC_Events = [] # Stores the history of spiking events in the On Neural Coalition spikes.
        self.OffNC_Events = [] # Stores the history of spiking events in the Off Neural Coalition spikes.
        self.OnDNC_Events = [] # Stores the history of spiking events in the 1st channel Decision Neural Coalition spikes.
        self.OffDNC_Events = [] # Stores the history of spiking events in the 2nd channel Decision Neural Coalition spikes.
        self.DNC_Events = [] # Stores the history of spiking events in the 3rd channel Decision Neural Coalition spikes.
        
        self.ANN_FR = [] # Monitors ANN Fire Rate
        self.ENN_FR = [] # Monitors ENN Fire Rate
        self.OnNC_FR = [] # Monitors On Neural Coalition Fire Rate
        self.OffNC_FR = [] # Monitors OFF Neural Coalition Fire Rate
        self.OnDNC_FR = [] # Monitors 1st channel Decision Neural Coalition Fire Rate
        self.OffDNC_FR = [] # Monitors 2nd channel Decision Neural Coalition Fire Rate
        self.DNC_FR = [] # Monitors 3rd channel Decision Neural Coalition Fire Rate
        
        self.ANN_A = [] # Monitors ANN Amplitude
        self.ENN_A = [] # Monitors ENN Amplitude
        self.OnNC_A = [] # Monitors On Neural Coalition Amplitude
        self.OffNC_A = [] # Monitors OFF Neural Coalition Amplitude
        self.OnDNC_A = [] # Monitors 1st channel Decision Neural Coalition Amplitude
        self.OffDNC_A = [] # Monitors 2nd channel Decision Neural Coalition Amplitude
        self.DNC_A = [] # Monitors 3rd channel Decision Neural Coalition Amplitude
        
        # Information Theory variables
        
        # Events 
        self.Correct = []
        self.CorrectActivation = []
        self.CorrectDeactivation = []
        self.HPConserved = []
        
        # Probability
        self.ActivationProbability = []
        self.CorrectActivationProbability = []
        self.CorrectDeactivationProbability = []
        self.CorrectProbability = []
        self.HPConservationProbability = []
        
        # Entropies
        self.ResponseEntropy = []
        self.CorrectEntropy = []
        self.JointEntropy = []
        self.MutualInformation = []
        self.HPConservationEntropy = []
        
        # Frequency Domain
        self.ANN_FR_F = []
        self.ENN_FR_F = []
        self.OnNC_FR_F = []
        self.OffNC_FR_F = []
        self.OnDNC_FR_F = []
        self.OffDNC_FR_F = []
        self.DNC_FR_F = []
         
    def distance_2_origin(self, obj):
        
        d = np.sqrt((obj.XPosition - self.XOrigin) ** 2 + (obj.YPosition - self.YOrigin) ** 2 + (obj.ZPosition - self.ZOrigin) ** 2)
        return d
    
    def distance_obj_obj(self, obj1, obj2):
        
        d = np.sqrt((obj1.XPosition - obj2.XPosition) ** 2 + (obj1.YPosition - obj2.YPosition) ** 2 + (obj1.ZPosition - obj2.ZPosition) ** 2)
        return d
    
    def add_object(self, obj):
        
        self.NumberOfObjects = self.NumberOfObjects + 1
        self.Objects[obj.Label] = obj
        obj.Distance2Origin = self.distance_2_origin(obj)
        
        if obj.Kind == "Heater":
            
            self.Heaters[obj.Label] = obj
            self.NumberOfHeaters = self.NumberOfHeaters + 1
            print('There are', self.NumberOfHeaters, "heaters in the environment")
            print('The heaters present in the environment are:',  self.Heaters)
        
        if obj.Kind == "On/Off Switch Heater":
            
            self.Heaters[obj.Label] = obj
            self.NumberOfHeaters = self.NumberOfHeaters + 1
            print('There are', self.NumberOfHeaters, "heaters in the environment")
            print('The heaters present in the environment are:',  self.Heaters)
        
        if obj.Kind == "Up/Down Switch Heater":
            
            self.Heaters[obj.Label] = obj
            self.NumberOfHeaters = self.NumberOfHeaters + 1
            print('There are', self.NumberOfHeaters, "heaters in the environment")
            print('The heaters present in the environment are:',  self.Heaters)
        
        if obj.Kind == "Cell_Culture":
            
            self.CellCultures[obj.Label] = obj
            self.NumberOfCellCultures = self.NumberOfCellCultures + 1
            print('There are', self.NumberOfCellCultures, "cell culultures in the environment")
            print('The cell culultures present in the environment are:',  self.CellCultures)
        
        print('There are', self.NumberOfObjects, "objects in the environment")
        print('The objects present in the environment are:',  self.Objects)
        
        # Double For loop conbstructing the Distances between objects
        for key1 in self.Objects:
            
            obj1 = self.Objects[key1]
            
            for key2 in self.Objects:
                
                obj2 = self.Objects[key2]
                obj1.Distances[key1 + '-' + key2] = self.distance_obj_obj(obj1, obj2)
                obj1.Object.Distances[key1 + '-' + key2] = obj1.Distances[key1 + '-' + key2]
                obj2.Distances[key2 + '-' + key1] = obj1.Distances[key1 + '-' + key2]
                obj2.Object.Distances[key2 + '-' + key1] = obj2.Distances[key2 + '-' + key1]
        
        # Print the Distances between objects
        for key in self.Objects:
            
            print(key,'Distances', self.Objects[key].Distances)

    
    def delete_object(self, obj):
        
        # Updating the Distances between Objects
        for key1 in self.Objects:
            
            obj1 = self.Objects[key1]
            
            for key2 in self.Objects:
                
                obj2 = self.Objects[key2]
                tmp = obj1.Label + '-' + obj2.Label
                obj1.Object.Distances.pop(tmp, None)
                obj1.Distances.pop(tmp, None)
                tmp = obj2.Label + '-' + obj1.Label 
                obj2.Distances.pop(tmp, None)
                obj2.Object.Distances.pop(tmp, None)
        
        # Print the Distances between objects
        for key in self.Objects:
            
            print(key,'Distances', self.Objects[key].Distances)
        
        self.NumberOfObjects = self.NumberOfObjects - 1
        del self.Objects[obj.Label]
        
        if obj.Kind == "Heater":
            
            del self.Heaters[obj.Label]
            self.NumberOfHeaters = self.NumberOfHeaters - 1
            print('There are', self.NumberOfHeaters, "heaters in the environment")
            print('The heaters present in the environment are:',  self.Heaters)
        
        if obj.Kind == "Cell_Culture":
            
            del self.CellCultures[obj.Label]
            self.NumberOfCellCultures = self.NumberOfCellCultures - 1
            print('There are', self.NumberOfCellCultures, "cell culultures in the environment")
            print('The cell culultures present in the environment are:',  self.CellCultures)        
    
    def update(self, cell_culture, heater, t, hset):
        
        # Note this function must run with a nengo simulator (nengo.Simulator)
        # It works as well with a nengo_dl.Simulator but I opted for the previous option to maintain the biological dynamics as much as possible
        
        ## Temperature Intput
        
        # Update the world's temperature
        
        self.Temperatures.append(self.Temperature)
        heater.heat()
        d = cell_culture.Distances[cell_culture.Label + '-' + heater.Label]
        
        ### HEAT Absorption by environment
        cell_culture.Temperature = cell_culture.Temperature +  heater.Temperature * d / self.HeatConductance 
        
        # Residue heat (heat unabsorbed by the cell culture)
        residue_heat = self.HeatAbsorption * heater.Temperature * d / self.HeatConductance
        
        # Heat Transfer
        if self.Temperature <= cell_culture.Temperature:
            
            self.Temperature = self.Temperature + residue_heat / 2 - 0.00025 * d / self.HeatConductance + hset.execute_hazards(time = self.Clock)
            cell_culture.Temperature = cell_culture.Temperature - residue_heat / 2 - 0.0001875 * d / self.HeatConductance + hset.execute_hazards(time = self.Clock)
            
        if self.Temperature >= cell_culture.Temperature:
            
            self.Temperature = self.Temperature - residue_heat / 2 - 0.00025 * d / self.HeatConductance + hset.execute_hazards(time = self.Clock)
            cell_culture.Temperature = cell_culture.Temperature + residue_heat / 2 - 0.0001875 * d / self.HeatConductance + hset.execute_hazards(time = self.Clock)
        
        hset.track_next_hazard(time = self.Clock)
        
        # Stimulate the network
        cell_culture.simulate_one_NeuralNetwork_time_instance(cell_culture.Temperature, self.Clock, self.Timestep)
        
        ### Heater Neural Control
        
        ## Output Probing
        move = 0 # Boolean ensuring the Cell culture takes one action only.
        
        # ANN
        ANN_hist = cell_culture.NN_sim.data[cell_culture.InputProbe]
        ANN_output = ANN_hist[-1]
        
        # ENN
        ENN_hist = cell_culture.NN_sim.data[cell_culture.OutputProbe]
        ENN_output = ENN_hist[-1]
        
        # OnNC
        OnNC_hist = cell_culture.NN_sim.data[cell_culture.OnProbe]
        OnNC_output = OnNC_hist[-1]
        
        # OffNC
        OffNC_hist = cell_culture.NN_sim.data[cell_culture.OffProbe]
        OffNC_output = OffNC_hist[-1]
        
        # DNC
        DNC = cell_culture.NN_sim.data[cell_culture.DecisionProbe]
        OnDNC_hist = DNC[:,0]
        OnDNC_output = OnDNC_hist[-1]
        OffDNC_hist = DNC[:,1]
        OffDNC_output = OffDNC_hist[-1]
        DNC_hist = DNC[:,2]
        DNC_output = DNC_hist[-1]
        
        ## Event Detection
        self.ANN_Events.append(event_detection(ANN_output, 1.1 * np.mean(ANN_hist)))
        self.ENN_Events.append(event_detection(ENN_output, 1.1 * np.mean(ENN_hist)))
        self.OnNC_Events.append(event_detection(OnNC_output, 1.1 * np.mean(OnNC_hist)))
        self.OffNC_Events.append(event_detection(OffNC_output, 1.1 * np.mean(OffNC_hist)))
        self.OnDNC_Events.append(event_detection(OnDNC_output, 1.1 * np.mean(OnDNC_hist)))
        self.OffDNC_Events.append(event_detection(OffDNC_output, 1.1 * np.mean(OffDNC_hist)))
        self.DNC_Events.append(event_detection(DNC_output, 1.1 * np.mean(DNC_hist)))        
        
        ## Fire Rates
        ANN_fr = estimate_FR(self.ANN_Events, t, WParam) / (WParam * self.Timestep)
        ENN_fr = estimate_FR(self.ENN_Events, t, WParam) / (WParam * self.Timestep)
        OnNC_fr = estimate_FR(self.OnNC_Events, t, WParam) / (WParam * self.Timestep)
        OffNC_fr = estimate_FR(self.OffNC_Events, t, WParam) / (WParam * self.Timestep)
        OnDNC_fr = estimate_FR(self.OnDNC_Events, t, WParam) / (WParam * self.Timestep)
        OffDNC_fr = estimate_FR(self.OffDNC_Events, t, WParam) / (WParam * self.Timestep)
        DNC_fr = estimate_FR(self.DNC_Events, t, WParam) / (WParam * self.Timestep)
        
        ## Amplitude
        ANN_a = estimate_Amplitude(ENN_hist, t, WParam)
        ENN_a = estimate_Amplitude(ENN_hist, t, WParam)
        OnNC_a = estimate_Amplitude(OnNC_hist, t, WParam)
        OffNC_a = estimate_Amplitude(OffNC_hist, t, WParam)
        OnDNC_a = estimate_Amplitude(OnDNC_hist, t, WParam)
        OffDNC_a = estimate_Amplitude(OffDNC_hist, t, WParam)
        DNC_a = estimate_Amplitude(DNC_hist, t, WParam)
        
        # Monitoring Fire Rates
        self.ANN_FR.append(ANN_fr)
        self.ENN_FR.append(ENN_fr)
        self.OnNC_FR.append(OnNC_fr)
        self.OffNC_FR.append(OffNC_fr)
        self.OnDNC_FR.append(OnDNC_fr)
        self.OffDNC_FR.append(OffDNC_fr)
        self.DNC_FR.append(DNC_fr)
        
        # Monitoring Amplitudes
        self.ANN_A.append(ANN_a)
        self.ENN_A.append(ENN_a)
        self.OnNC_A.append(OnNC_a)
        self.OffNC_A.append(OffNC_a)
        self.OnDNC_A.append(OnDNC_a)
        self.OffDNC_A.append(OffDNC_a)
        self.DNC_A.append(DNC_a)
        
        # Note electrical current in the cell culture is monitored as negative but it will be postive by the heater's switch
        
        if OnNC_fr > OffNC_fr and ENN_fr > np.mean(self.ENN_FR):
            
            if heater.Kind == 'On/Off Switch Heater':
                
                heater.activate()
                
            if heater.Kind == 'Up/Down Switch Heater':
                
                heater.increase_power()
                
            move = 1
            
        if OffNC_fr > OnNC_fr and ENN_fr > np.mean(self.ENN_FR) and move == 0:
            
            if heater.Kind == 'On/Off Switch Heater':
                
                heater.deactivate()
            
            if heater.Kind == 'Up/Down Switch Heater':
                
                heater.decrease_power()
        
        # update the clocks
        self.TimeObject.update_time()
        self.Clock = self.TimeObject.Clock
        self.TimeInstances.append(self.Clock)
        
        # Save the status of the heater at that time instance.
        heater.make_heater_history()
        
        # Information Theory Analysis
        
        if heater.Kind == 'On/Off Switch Heater':
            
            check = heater.status


        if heater.Kind == 'Up/Down Switch Heater':
            
            if len(heater.Memory) == 1:
                
                check = heater.status
                
            else:
                
                check = heater.Memory[t] - heater.Memory[t - 1]
        
        if check > 0 and cell_culture.Temperature < 22:
            
            self.CorrectActivation.append(1)
            self.CorrectDeactivation.append(0)
            self.Correct.append(1)
        
        if check <= 0 and cell_culture.Temperature > 22:
            
            self.CorrectActivation.append(0)
            self.CorrectDeactivation.append(1)
            self.Correct.append(1)
            
        else:
            
            self.CorrectActivation.append(0)
            self.CorrectDeactivation.append(0)
            self.Correct.append(0)
        
        # If there is naught or positive reward HP was conserved
        if cell_culture.CurrentHPReward >= 0:
            
            self.HPConserved.append(1)
            
        else:
            
            self.HPConserved.append(0)
        
        # Probabilities           
        self.ActivationProbability.append((sum(heater.Memory) / len(heater.Memory)))
        self.CorrectActivationProbability.append(sum(self.CorrectActivation) / len(self.CorrectActivation))
        self.CorrectDeactivationProbability.append(sum(self.CorrectDeactivation) / len(self.CorrectDeactivation))
        self.CorrectProbability.append(sum(self.Correct) / len(self.Correct))
        self.HPConservationProbability.append(sum(self.HPConserved) / len(self.HPConserved))
        
        # Entropies
        self.ResponseEntropy.append(hop(self.ActivationProbability[t]) + hop(1 - self.ActivationProbability[t]))
        self.CorrectEntropy.append(hop(self.CorrectProbability[t]) + hop(1 - self.CorrectProbability[t]))
        self.JointEntropy.append(hop(self.CorrectActivationProbability[t]) + hop(self.ActivationProbability[t] * (1 - self.CorrectProbability[t])) + hop(self.CorrectDeactivationProbability[t]) + hop((1 - self.ActivationProbability[t]) * (1 - self.CorrectProbability[t])))
        self.HPConservationEntropy.append(hop(self.HPConservationProbability[t]) + hop(1 - self.HPConservationProbability[t]))
        
        # Mutual Information
        self.MutualInformation.append(self.CorrectEntropy[t] + self.ResponseEntropy[t] - self.JointEntropy[t])
        
    def display_all_experimental_variables(self, cell_culture, heater, hset): #, frequencies, time_of_experiment, nb_timesteps):
        
        
        
        cell_culture.display_network_states()
        print('Analysis of the states of the world starts')
        
        ## Plot heater status
        
        if heater.Kind == 'On/Off Switch Heater':
            
            plt.figure(figsize = (10, 6))
            plt.plot(self.TimeInstances, heater.Memory)
            plt.title('Heater Status as a Function of Time', {'fontsize': 20})
            plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
            plt.ylabel('Status [Binary]', {'fontsize': 18})
            plt.show()
        
        if heater.Kind == 'Up/Down Switch Heater':
            
            plt.figure(figsize = (10, 6))
            plt.plot(self.TimeInstances, heater.Memory)
            plt.title('Heater Status as a Function of Time', {'fontsize': 20})
            plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
            plt.ylabel('Power in Watts [W]', {'fontsize': 18})
            plt.show()
        
        
        ## Plot world Temperature
        
        plt.figure(figsize = (10, 6))
        plt.plot(self.TimeInstances, self.Temperatures)
        plt.title('Temperature of the World as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        plt.ylabel('Temperature in Degrees Celsious [C]', {'fontsize': 18})
        plt.show()
        
        ## Plot Fire Rates
        
        # Input - Output
        plt.figure(figsize = (10, 6))
        plt.plot(self.TimeInstances, self.ANN_FR, label = 'Afferent Neural Network')
        plt.plot(self.TimeInstances, self.ENN_FR, label = 'Efferent Neural Network')
        plt.plot(self.TimeInstances, self.OnNC_FR, label = 'On Neural Coalition')
        plt.plot(self.TimeInstances, self.OffNC_FR, label = 'Off Neural Coalition')
        plt.title('Fire Rate of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Fire Rate in Hertz [Hz]', {'fontsize': 18})
        plt.legend(prop={'size': 18})
        plt.show()
        
        # Decision
        plt.figure(figsize = (10, 6))
        plt.plot(self.TimeInstances, self.OnDNC_FR, label = 'D1 Neural Coalition')
        plt.plot(self.TimeInstances, self.OffDNC_FR, label = 'D2 Neural Coalition')
        plt.plot(self.TimeInstances, self.DNC_FR, label = 'D3 Neural Coalition')
        plt.title('Fire Rate of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Fire Rate in Hertz [Hz]', {'fontsize': 18})
        plt.legend(prop={'size': 18})
        plt.show()
        
        ## Plot Fire Rates filtered
        
        # Input - Output
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, MA_Filter(self.ANN_FR, 200), label = 'Afferent Neural Network')
        plt.plot(self.TimeInstances, MA_Filter(self.ENN_FR, 200), label = 'Efferent Neural Network')
        plt.plot(self.TimeInstances, MA_Filter(self.OnNC_FR, 200), label = 'On Neural Coalition')
        plt.plot(self.TimeInstances, MA_Filter(self.OffNC_FR, 200), label = 'Off Neural Coalition')
        plt.title('Fire Rate of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Fire Rate in Hertz [Hz]', {'fontsize': 18})
        plt.legend(prop={'size': 14})
        plt.show()
        
        # Decision
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, MA_Filter(self.OnDNC_FR, 200), label = 'Decision Neural Coalition 1')
        plt.plot(self.TimeInstances, MA_Filter(self.OffDNC_FR, 200), label = 'Decision Neural Coalition 2')
        plt.plot(self.TimeInstances, MA_Filter(self.DNC_FR, 200), label = 'Decision Neural Coalition 3')
        plt.title('Fire Rate of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Fire Rate in Hertz [Hz]', {'fontsize': 18})
        plt.legend(prop={'size': 14})
        plt.show()
        
        ## Probabilities
        
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, self.ActivationProbability, label = 'Probability of Activation')
        plt.plot(self.TimeInstances, self.CorrectProbability, label = 'Probability of Correct Action')
        plt.plot(self.TimeInstances, self.CorrectActivationProbability, label = 'Probability of Correct Activation')
        plt.plot(self.TimeInstances, self.CorrectDeactivationProbability, label = 'Probability of Correct Deactivation')
        plt.plot(self.TimeInstances, self.HPConservationProbability, label = 'Probability of Hit Point Conservation')
        plt.title('Activation Probability as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Probability', {'fontsize': 18})
        plt.legend(prop={'size': 14})
        plt.show()
        
        ## Plot mutual information and Entropy
        
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, self.ResponseEntropy, label = 'Activation Entropy')
        plt.plot(self.TimeInstances, self.CorrectEntropy, label = 'Correct Entropy')
        plt.plot(self.TimeInstances, self.JointEntropy, label = 'Joint Entropy')
        plt.plot(self.TimeInstances, self.MutualInformation, label = 'Mutual Information')
        plt.plot(self.TimeInstances,  self.HPConservationEntropy, label = 'HP Conservation')
        plt.title('Mutual Information and Entropy as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        plt.ylabel('Entropy in bits [b]', {'fontsize': 16})
        plt.legend(prop={'size': 14})
        plt.show()
        
        ## Plot Amplitude
        
        # ANN, OnNC, OffNC, ENN
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, MA_Filter(self.ANN_A, 200), label = 'Afferent Neural Network')
        plt.plot(self.TimeInstances, MA_Filter(self.ENN_A, 200), label = 'Efferent Neural Network')
        plt.plot(self.TimeInstances, MA_Filter(self.OnNC_A, 200), label = 'On Neural Coalition')
        plt.plot(self.TimeInstances, MA_Filter(self.OffNC_A, 200), label = 'Off Neural Coalition')
        plt.title('Amplitude of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 16})
        plt.legend(prop={'size': 12})
        plt.show()
        
        # OnDNC, OffDNC, DNC
        plt.figure(figsize = (12, 8))
        plt.plot(self.TimeInstances, MA_Filter(self.OnDNC_A, 200), label = 'D1 Neural Coalition')
        plt.plot(self.TimeInstances, MA_Filter(self.OffDNC_A, 200), label = 'D2 Neural Coalition')
        plt.plot(self.TimeInstances, MA_Filter(self.DNC_A, 200), label = 'D3 Neural Coalition')
        plt.title('Amplitude of the Neuron Coalitions as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 16})
        plt.legend(prop={'size': 12})
        plt.show()
        
        ## Frequency Analysis and Power Spectrum for Frequency
        
        # ANN
        fANN, tANN, SANN = spectrogram(np.array(self.ANN_FR), 1 / (5 * self.Timestep))
        lfANN = np.log10(fANN[1:])
        maxANN = np.amax(SANN)
        minANN = np.amin(SANN)
        maxANN = np.amax(SANN) / 250
        tANN = 2 * tANN[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tANN, lfANN, SANN[1:, 1:], vmin = minANN, vmax = maxANN, cmap='coolwarm')
        plt.yticks(lfANN[::20], np.around(fANN[1::20]))
        plt.title('ANN Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OnNC
        fOnNC, tOnNC, SOnNC = spectrogram(np.array(self.OnNC_FR), 1 / (5 * self.Timestep))
        lfOnNC = np.log10(fOnNC[1:])
        maxOnNC = np.amax(SOnNC)
        minOnNC = np.amin(SOnNC)
        maxOnNC = np.amax(SOnNC) / 31.25
        tOnNC = 2 * tOnNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOnNC, lfOnNC, SOnNC[1:, 1:], vmin = minOnNC, vmax = maxOnNC, cmap='coolwarm')
        plt.yticks(lfOnNC[::20], np.around(fOnNC[1::20]))
        plt.title('On Neural Coalition Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OffNC
        fOffNC, tOffNC, SOffNC = spectrogram(np.array(self.OffNC_FR), 1 / (5 * self.Timestep))
        lfOffNC = np.log10(fOffNC[1:])
        maxOffNC = np.amax(SOffNC)
        minOffNC = np.amin(SOffNC)
        maxOffNC = np.amax(SOffNC) / 31.25
        tOffNC = 2 * tOffNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOffNC, lfOffNC, SOffNC[1:, 1:], vmin = minOffNC, vmax = maxOffNC, cmap='coolwarm')
        plt.yticks(lfOffNC[::20], np.around(fOffNC[1::20]))
        plt.title('Off Neural Coalition Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OnDNC
        fOnDNC, tOnDNC, SOnDNC = spectrogram(np.array(self.OnDNC_FR), 1 / (5 * self.Timestep))
        lfOnDNC = np.log10(fOnDNC[1:])
        maxOnDNC = np.amax(SOnDNC)
        minOnDNC = np.amin(SOnDNC)
        maxOnDNC = np.amax(SOnDNC) / 31.25
        tOnDNC = 2 * tOnDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOnDNC, lfOnDNC, SOnDNC[1:, 1:], vmin = minOnDNC, vmax = maxOnDNC, cmap='coolwarm')
        plt.yticks(lfOnDNC[::20], np.around(fOnDNC[1::20]))
        plt.title('Decision Coalition Channel 1 Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        
        # OffDNC
        fOffDNC, tOffDNC, SOffDNC = spectrogram(np.array(self.OffDNC_FR), 1 / (5 * self.Timestep))
        lfOffDNC = np.log10(fOffDNC[1:])
        maxOffDNC = np.amax(SOffDNC)
        minOffDNC = np.amin(SOffDNC)
        maxOffDNC = np.amax(SOffDNC) / 31.25
        tOffDNC = 2 * tOffDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOffDNC, lfOffDNC, SOffDNC[1:, 1:], vmin = minOffDNC, vmax = maxOffDNC, cmap='coolwarm')
        plt.yticks(lfOffDNC[::20], np.around(fOffDNC[1::20]))
        plt.title('Decision Coalition Channel 2 Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # DNC
        fDNC, tDNC, SDNC = spectrogram(np.array(self.DNC_FR), 1 / (5 * self.Timestep))
        lfDNC = np.log10(fDNC[1:])
        maxDNC = np.amax(SDNC)
        minDNC = np.amin(SDNC)
        maxDNC = np.amax(SDNC) / 31.25
        tDNC = 2 * tDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tDNC, lfDNC, SDNC[1:, 1:], vmin = minDNC, vmax = maxDNC, cmap='coolwarm')
        plt.yticks(lfDNC[::20], np.around(fDNC[1::20]))
        plt.title('Decision Coalition Channel 3 Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Decision
        fD, tD, SD = spectrogram(np.array(self.OnDNC_FR) + np.array(self.OffDNC_FR) + np.array(self.DNC_FR), 1 / (5 * self.Timestep))
        lfD = np.log10(fD[1:])
        maxD = np.amax(SD)
        minD = np.amin(SD)
        maxD = np.amax(SD) / 250
        tD = 2 * tD[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tD, lfD, SD[1:, 1:], vmin = minD, vmax = maxD, cmap='coolwarm')
        plt.yticks(lfDNC[::20], np.around(fDNC[1::20]))
        plt.title('Decision Coalition Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # ENN
        fENN, tENN, SENN = spectrogram(np.array(self.ENN_FR), 1 / (5 * self.Timestep))
        lfENN = np.log10(fENN[1:])
        maxENN = np.amax(SENN)
        minENN = np.amin(SENN)
        maxENN = np.amax(SENN) / 31.25
        tENN = 2 * tENN[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tENN, lfENN, SENN[1:, 1:], vmin = minENN, vmax = maxENN, cmap='coolwarm')
        plt.yticks(lfENN[::20], np.around(fENN[1::20]))
        plt.title('ENN Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Output
        fO, tO, SO = spectrogram(abs(np.array(self.OnNC_FR) - np.array(self.OffNC_FR)) + np.array(self.ENN_FR), 1 / (5 * self.Timestep))
        lfO = np.log10(fO[1:])
        maxO = np.amax(SO)
        minO = np.amin(SO)
        maxO = np.amax(SO) / 250
        tO = 2 * tO[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tO, lfO, SO[1:, 1:], vmin = minO, vmax = maxO, cmap='coolwarm')
        plt.yticks(lfO[::20], np.around(fO[1::20]))
        plt.title('Output Spectrogram for Fire Rate', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        ## Frequency Analysis of Voltage
        
        # ANN
        fANN, tANN, SANN = spectrogram(np.reshape(cell_culture.NN_sim.data[cell_culture.InputProbe], np.size(cell_culture.NN_sim.data[cell_culture.InputProbe])), 1 / (5 * self.Timestep))
        lfANN = np.log10(fANN[1:])
        maxANN = np.amax(SANN)
        minANN = np.amin(SANN)
        maxANN = np.amax(SANN) / 2000
        tANN = 2 * tANN[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tANN, lfANN, SANN[1:, 1:], vmin = minANN, vmax = maxANN, cmap='coolwarm')
        plt.yticks(lfANN[::20], np.around(fANN[1::20]))
        plt.title('ANN Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OnNC
        fOnNC, tOnNC, SOnNC = spectrogram(np.reshape((cell_culture.NN_sim.data[cell_culture.OnProbe]), np.size(cell_culture.NN_sim.data[cell_culture.InputProbe])), 1 / (5 * self.Timestep))
        lfOnNC = np.log10(fOnNC[1:])
        maxOnNC = np.amax(SOnNC)
        minOnNC = np.amin(SOnNC)
        maxOnNC = np.amax(SOnNC) / 250
        tOnNC = 2 * tOnNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOnNC, lfOnNC, SOnNC[1:, 1:], vmin = minOnNC, vmax = maxOnNC, cmap='coolwarm')
        plt.yticks(lfOnNC[::20], np.around(fOnNC[1::20]))
        plt.title('On Neural Coalition Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OffNC
        fOffNC, tOffNC, SOffNC = spectrogram(np.reshape(cell_culture.NN_sim.data[cell_culture.OffProbe], np.size(cell_culture.NN_sim.data[cell_culture.OffProbe])), 1 / (5 * self.Timestep))
        lfOffNC = np.log10(fOffNC[1:])
        maxOffNC = np.amax(SOffNC)
        minOffNC = np.amin(SOffNC)
        maxOffNC = np.amax(SOffNC) / 250
        tOffNC = 2 * tOffNC / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOffNC, lfOffNC, SOffNC[1:, 1:], vmin = minOffNC, vmax = maxOffNC, cmap='coolwarm')
        plt.yticks(lfOffNC[::20], np.around(fOffNC[1::20]))
        plt.title('Off Neural Coalition Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Decision data
        Ddata = cell_culture.NN_sim.data[cell_culture.DecisionProbe]
        OnDNC_dat = Ddata[:,0]
        OffDNC_dat = Ddata[:,1]
        DNC_dat = Ddata[:,2]
        
        # OnDNC
        fOnDNC, tOnDNC, SOnDNC = spectrogram(np.reshape(OnDNC_dat, np.size(OnDNC_dat)), 1 / (5 * self.Timestep))
        lfOnDNC = np.log10(fOnDNC[1:])
        maxOnDNC = np.amax(SOnDNC)
        minOnDNC = np.amin(SOnDNC)
        maxOnDNC = np.amax(SOnDNC) / 250
        tOnDNC = 2 * tOnDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOnDNC, lfOnDNC, SOnDNC[1:, 1:], vmin = minOnDNC, vmax = maxOnDNC, cmap='coolwarm')
        plt.yticks(lfOnDNC[::20], np.around(fOnDNC[1::20]))
        plt.title('Decision Coalition Channel 1 Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # OffDNC
        fOffDNC, tOffDNC, SOffDNC = spectrogram(np.reshape(OffDNC_dat, np.size(OffDNC_dat)), 1 / (5 * self.Timestep))
        lfOffNC = np.log10(fOffNC[1:])
        maxOffDNC = np.amax(SOffDNC)
        minOffDNC = np.amin(SOffDNC)
        maxOffDNC = np.amax(SOffDNC) / 250
        tOffDNC = 2 * tOffDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tOffDNC, lfOffDNC, SOffDNC[1:, 1:], vmin = minOffDNC, vmax = maxOffDNC, cmap='coolwarm')
        plt.yticks(lfOffDNC[::20], np.around(fOffDNC[1::20]))
        plt.title('Decision Coalition Channel 2 Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # DNC
        fDNC, tDNC, SDNC = spectrogram(np.reshape(DNC_dat, np.size(DNC_dat)), 1 / (5 * self.Timestep))
        lfDNC = np.log10(fDNC[1:])
        maxDNC = np.amax(SDNC)
        minDNC = np.amin(SDNC)
        maxDNC = np.amax(SDNC) / 250
        tDNC = 2 * tDNC[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tDNC, lfDNC, SDNC[1:, 1:], vmin = minDNC, vmax = maxDNC, cmap='coolwarm')
        plt.yticks(lfDNC[::20], np.around(fDNC[1::20]))
        plt.title('Decision Coalition Channel 3 Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Decision
        fD, tD, SD = spectrogram(np.reshape(OnDNC_dat + OffDNC_dat + DNC_dat, np.size(DNC_dat)), 1 / (5 * self.Timestep))
        lfD = np.log10(fD[1:])
        maxD = np.amax(SD)
        minD = np.amin(SD)
        maxD = np.amax(SD) / 2000
        tD = 2 * tD[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tD, lfD, SD[1:, 1:], vmin = minD, vmax = maxD, cmap='coolwarm')
        plt.yticks(lfD[::20], np.around(fD[1::20]))
        plt.title('Decision Coalition Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # ENN
        fENN, tENN, SENN = spectrogram(np.reshape(cell_culture.NN_sim.data[cell_culture.OutputProbe], np.size(cell_culture.NN_sim.data[cell_culture.OutputProbe])), 1 / (5 * self.Timestep), mode = 'magnitude')
        lfENN = np.log10(fENN[1:])
        maxENN = np.amax(SENN)
        minENN = np.amin(SENN)
        maxENN = np.amax(SENN) / 25
        tENN = 2 * tENN[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tENN, lfENN, SENN[1:, 1:], vmin = minENN, vmax = maxENN, cmap='coolwarm')
        plt.yticks(lfENN[::20], np.around(fENN[1::20]))
        plt.title('ENN Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Output
        fO, tO, SO = spectrogram(np.reshape(abs(cell_culture.NN_sim.data[cell_culture.OnProbe] - cell_culture.NN_sim.data[cell_culture.OffProbe]) + cell_culture.NN_sim.data[cell_culture.OutputProbe], np.size(cell_culture.NN_sim.data[cell_culture.OutputProbe])), 1 / (5 * self.Timestep), mode = 'magnitude')
        lfO = np.log10(fO[1:])
        maxO = np.amax(SO)
        minO = np.amin(SO)
        maxO = np.amax(SO) / 12.5
        tO = 2 * tO[1:] / 10
        
        # Spectrogram
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(tO, lfO, SO[1:,1:], vmin = minO, vmax = maxO, cmap='coolwarm')
        plt.yticks(lfD[::20], np.around(fD[1::20]))
        plt.title('Output Spectrogram for Voltage', {'fontsize': 20})
        plt.ylabel('Frequency in Hertz [Hz]', {'fontsize': 16})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        c1 = plt.colorbar()
        c1.set_label('Gain in Watts [W]', fontsize = 16, labelpad = 10)
        plt.show()   
        
        print('Analysis of the states of the world ends')
        
        ### GAME ANALYSIS ###
        
        print("Start of game analysis ...")
        
        # Cell Culture's Strategy
        GameMat = np.array([np.array(self.ActivationProbability), np.array(cell_culture.HPLossMemory)])
        GameMat = GameMat.T
        indexes = GameMat[:, 0].argsort()
        GameMat = GameMat[indexes]
        GameMat = GameMat.T
        AP = GameMat[0, :]
        ratio = int(50 * np.size(AP) / 1000) # Assess the strategy every 20ms
        AP = AP[::ratio]
        HPLoss = GameMat[1, :]
        HPLoss = HPLoss [::ratio]
        EP = 1 - np.linspace(0., 1, np.size(AP)) # The set of all possible cooldown frequencies
        HHFreq = hset.ProbabilityHeatHazard
        EPinSim = 1 - np.linspace(hset.ProbabilityHeatHazard, hset.MaxHeatHazardFrequency, np.size(AP))
        CDFIndex = (np.abs(EP - HHFreq)).argmin()
        CDF = EP[CDFIndex]
        
        # Meshing and Strategy fitness
        EPMesh, APMesh = np.meshgrid(EP, AP)
        HPLMesh = np.tile(HPLoss, (np.size(AP), 1))
        ExpectedPayoff = np.transpose(HPLMesh.T * EP) * AP
        StrategyFitnessCc = np.matmul(AP, ExpectedPayoff)
        ExpectedFitnessCc = np.matmul(StrategyFitnessCc, AP.T)
        StrategyGradientCc = ExpectedFitnessCc * (1 - AP)
        StrategyFitnessE = np.matmul(EP, ExpectedPayoff)
        ExpectedFitnessE = np.matmul(StrategyFitnessE, EP.T)
        StrategyGradientE = ExpectedFitnessE * (1 - EP)
        Norm = np.hypot(StrategyGradientE, StrategyGradientCc)
        
        # Applying Von Neuman's MiniMax Theorem
        lower_limit = np.amax(np.amin(ExpectedPayoff, 1))
        upper_limit = np.amin(np.amax(ExpectedPayoff, 1))
        
        # HP Loss as a function of strategy 2D
        plt.figure(figsize = (10, 6))
        plt.plot(AP, HPLoss)
        plt.title('Hit Point Reward as a Function of the Cell Culture Strategy', {'fontsize': 20}, pad = 20)
        plt.xlabel('Probability of Activating the Heater', {'fontsize': 16}, labelpad = 20)
        plt.ylabel('Strategy Outcome in HP Reward', {'fontsize': 16}, labelpad = 20)
        plt.show()
        
        # HP Loss as a function of strategy 3D
        fig = plt.figure(figsize = (14, 10))
        ax = fig.gca(projection = '3d')
        ax.plot_surface(EPMesh, APMesh, HPLMesh.T)
        plt.title('Hit Point Reward as a Function of the Cell Culture Strategy', {'fontsize': 20}, pad = 15)
        plt.xlabel('Frequency of Environment Cool Down', {'fontsize': 16}, labelpad = 10)
        plt.ylabel('Frequency of Heater Activation', {'fontsize': 16}, labelpad = 10)
        ax.set_zlabel('Strategy Outcome in HP Reward', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Expected Payoff as a function of strategy 2D
        plt.figure(figsize = (10, 6))
        plt.plot(AP, np.flip(ExpectedPayoff[CDFIndex, :]))
        plt.title('Expected Payoff as a Function of the Cell Culture Strategy', {'fontsize': 20})
        plt.xlabel('Frequency of Heater Activation', {'fontsize': 16})
        plt.ylabel('Strategy Expected Payoff in HP Reward', {'fontsize': 16})
        plt.show()
        
        # Expected Payoff as a function of strategy 3D
        fig = plt.figure(figsize = (14, 10))
        ax = fig.gca(projection = '3d')
        ax.plot_surface(EPMesh, APMesh, ExpectedPayoff)
        plt.title('Expected Payoff as a Function of the Cell Culture Strategy', {'fontsize': 20}, pad = 15)
        plt.xlabel('Frequency of Environment Cool Down', {'fontsize': 16}, labelpad = 10)
        plt.ylabel('Frequency of Heater Activation', {'fontsize': 16}, labelpad = 10)
        ax.set_zlabel('Strategy Expected Payoff in HP Reward', fontsize = 16, labelpad = 10)
        plt.show()
        
        fig, ax = plt.subplots(figsize = (12, 8))
        ax.quiver(EPMesh, APMesh, StrategyGradientE, StrategyGradientCc, Norm)
        ax.scatter(EPMesh[::ratio, ::ratio], APMesh[::ratio, ::ratio], color='0.5', s=1)
        plt.plot(1 - EP, AP, label = 'Cell Culture Strategy')
        plt.plot(np.flip(EPinSim), AP, label = 'Environment Strategy')
        plt.title('Strategy Phase Plane', {'fontsize': 20})
        plt.xlabel('Frequency of Environment Cool Down', {'fontsize': 16})
        plt.ylabel('Frequency of Heater Activation', {'fontsize': 16})
        plt.legend(loc = 3, prop={'size': 12})
        plt.show()
        
        print("End of Game Analysis")
        
        
        
        if lower_limit == upper_limit:
            
            minFreqA = round(np.amax(AP), 3)
            minFreqH = round(1 - CDF, 3)
            print("The value of the game is a Nash Equilibrium and is: (", minFreqA, "; ", minFreqH, "; ", upper_limit, ")")
        
        else:
            
            minFreqA = round(np.amin(AP), 3)
            minFreqH = round(1 - CDF, 3)
            maxFreqA = round(np.amax(AP), 3)
            maxFreqH = round(1 - CDF, 3)
            print("The lower limit of the game is: (", maxFreqA,"; ", maxFreqH, "; ", lower_limit, ")")
            print("The upper limit of the game is: (", minFreqA,"; ", minFreqH, "; ", upper_limit, ")")
        
        print('End of game analysis')
    
    def run_experiment(self, time_of_experiment, cell_culture, heater, hset):
        
        nb_timesteps = round(time_of_experiment / self.Timestep)
        cell_culture.create_probes(self.Timestep)
        cell_culture.NN_sim = nengo.Simulator(cell_culture.NeuralNetwork)
        
        with cell_culture.NN_sim:
            
            # heater.activate()
            
            for i in range(0, nb_timesteps):
                
                # Simulation stops after simulation time was completed
                if len(cell_culture.Temperatures) > nb_timesteps:
                    
                    break
                
                # Simulation stops if the cell culture dies
                if cell_culture.HitPoints == 0:
                    
                    print("The neuron cell culture suvived ", i, " epochs.")
                    
                    break
                
                self.update(cell_culture, heater, i, hset)
                
        self.display_all_experimental_variables(cell_culture, heater, hset) #, frequencies, time_of_experiment, nb_timesteps)

        

            
if __name__ == "__main__":
    
    #  Creating neuralnet for experiment
    
    print('Simulation Initialisation...')
    
    # Import Classes
    from CellCulture import CellCulture
    from HeatSystem import OnOffHeater
    from Hazard import HazardSet
    
    # Cell Culture Parameters
    initial_temperature = 1
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
    heater_power = 0.15 # Power of 1 Watt [W]
    heater_length = 20
    heater_width = 10
    heater_height = 5
    heater_position_x = 0
    heater_position_y = 0
    heater_position_z = 0
    
    h = OnOffHeater(heater_power,
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
    time_of_experiment = 80. # eighty seconds
    nb_hazards = int(time_of_experiment / 8.) # One hazard for 10, 5, 2, 1 seconds
    print("There will be ", nb_hazards, "hazards in this experiment")
    
    # Plan hazards
    hset = HazardSet()
    hset.plan_hazards(time_of_experiment, nb_hazards)
    
    print('Simulation Initialisation... Finished.')
    
    # Runs the experiment with the parameters above
    w1.run_experiment(time_of_experiment, cc, h, hset)
    
    # Delete the objects at the end of the experiment
    w1.delete_object(h)
    w1.delete_object(cc)
    