#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:35:15 2019

@author: TEB
"""

import numpy as np
from World import Object
import nengo
from nengo.processes import WhiteNoise
from nengo.processes import Piecewise
import matplotlib.pyplot as plt
from Stimulus import Stimulus
from nengo.utils.matplotlib import rasterplot
from mpl_toolkits.mplot3d import Axes3D

class CellCulture(Object, Stimulus):
    
    def __init__(self,
                 initial_temperature,
                 initial_time,
                 diameter,
                 height,
                 conduction,
                 position_x,
                 position_y,
                 position_z,
                 HP = 1000,
                 label = None):
        
        # Innate attributes
        self.Object = Object(position_x, position_y, position_z, initial_temperature)
        self.Temperature = self.Object.Temperature
        self.XPosition = self.Object.XPosition
        self.YPosition = self.Object.YPosition
        self.ZPosition = self.Object.ZPosition
        self.Distance2Origin = self.Object.Distance2Origin
        self.Distances = self.Object.Distances
        self.Label = self.Object.Label
        self.HitPoints = HP # H.P. stands for hit points.
        self.HPMax = HP # set max HP to initial HP
        
        # Current Perception State
        self.CurrentStimulus = Stimulus(self.Temperature, initial_time)
        self.StimulusTime = initial_time
        self.Timestep = self.CurrentStimulus.TimeObject.Timestep
        self.CurrentStimulus.spikecount2stimulus()
        self.CurrentSpikeCount = self.CurrentStimulus.SpikeCount
        self.CurrentStimulusMagnitude = self.CurrentStimulus.StimulusMagnitude
        self.CurrentStimulusDic = self.CurrentStimulus.StimulusDic
        self.CurrentNetworkInput = self.CurrentStimulus.NetworkInput
        self.CurrentHPReward = 0. # At the start of the experiment it will be assumed that HP loss at the first timestep.
        self.CurrentInput = None
        self.CurrentTNode = None
        self.CurrentCNodeAaNN = None
        self.CurrentCNodeCon = None
        self.CurrentCNodeCoff = None
        self.CurrentCNodeD = None
        self.CurrentCNodeEnn = None
        
        # Probes
        self.InputProbe = []
        self.OnProbe = []
        self.OffProbe = []
        self.DecisionProbe = []
        self.OutputProbe = []
        
        # Cell Culture Memory
        self.Stimuli = []
        self.SpikeCounts = []
        self.StimuliMagnitude = [] 
        self.StimuliDic = {}
        self.NetworkInputs = []
        self.HPLossMemory = []
        self.HPMemory = []
        
        # Physical Characteristic of the Cell Culture
        self.Diameter = diameter
        self.Radius = self.Diameter / 2
        self.Height = height
        self.Volume = np.pi * self.Height * self.Radius ** 2
        self.Area = 2 * np.pi * self.Radius ** 2 + 2 * np.pi * self.Radius * self.Height
        self.EffectiveArea = np.pi * self.Radius ** 2
        self.Kind = "Cell_Culture"
        self.Temperatures = []
        self.HeatConduction = conduction
        
        # Neural Nets
        self.AfferentNeuralNetwork = None
        self.CoalitionOn = None
        self.CoalitionOff = None
        self.DecisionCoalition = None
        self.EfferentNeuralNetwork = None
        self.NeuralNetwork = None
        
        # Connections
        self.InT2Ann = None
        self.InC2Ann = None
        self.InC2Con = None
        self.InC2Coff = None
        self.InC2D = None
        self.InC2Enn = None
        self.Ann2Con = None
        self.Ann2Coff = None
        self.Con2Coff = None
        self.Coff2Con = None
        self.Con2D1 = None
        self.Con2D2 = None
        self.Con2D3 = None
        self.Coff2D1 = None
        self.Coff2D2 = None
        self.Coff2D3 = None
        self.D2D = None
        self.D2Con1 = None
        self.D2Con2 = None
        self.D2Con3 = None
        self.D2Coff1 = None
        self.D2Coff2 = None
        self.D2Coff3 = None
        
        # Simulators
        self.ANN_sim = None
        self.OnC_sim = None
        self.OffC_sim = None
        self.DC_sim = None
        self.ENN_sim = None
        self.NN_sim = None
    
    # Hit Point to Voltage (FR) (hp2fr)
    def hpcost2v(self):
        
        if self.HitPoints == 0:
            
            # If the neurons are dead the fire rate is naught and the network stops learning.
            print("The Neurons are DEAD --> Learning Stops")
            return 0.
        
        # Voltage
        v = 2. * (1. - float(self.CurrentHPReward)) / 3. - 1.        
        return v
    
    def label_cell_culture(self, label):
        
        self.Object.label_object(label)
        self.Label = self.Object.Label
    
    def update_state(self, temperature, time):
        
        # Update States
        self.Temperature = temperature
        self.CurrentStimulus = Stimulus(self.Temperature, time)
        self.StimulusTime = time
        self.CurrentStimulus.spikecount2stimulus()
        self.CurrentSpikeCount = self.CurrentStimulus.SpikeCount
        self.CurrentStimulusMagnitude = self.CurrentStimulus.StimulusMagnitude
        self.CurrentStimuliDic = self.CurrentStimulus.StimulusDic
        self.CurrentNetworkInput = self.CurrentStimulus.NetworkInput
        self.CurrentHPReward = self.CurrentStimulus.HPReward
        self.HitPoints = self.HitPoints + self.CurrentHPReward
        
        # Update Memory
        self.Temperatures.append(self.Temperature)
        self.Stimuli.append(self.CurrentStimulus)
        self.SpikeCounts.append(self.CurrentSpikeCount)
        self.StimuliMagnitude.append(self.CurrentStimulusMagnitude)
        self.HPLossMemory.append(self.CurrentHPReward)
        self.StimuliDic = {**self.StimuliDic, **self.CurrentStimulusDic}
        self.NetworkInputs = Piecewise(self.StimuliDic)
        
        ## HP Memory ##
        # Make sure Hit Points are never negative
        if self.HitPoints < 0:
            
            self.HitPoints = 0
        
        # Make sure HP does not exceed the initial HP count
        if self.HitPoints > self.HPMax:
            
            self.HitPoints = self.HPMax
        
        # Make sure Hit Points are never negative
        if self.HitPoints == 0:
            
            print('The Neurons are DEAD')
        
        self.HPMemory.append(self.HitPoints)
    
    def create_NeuralNetwork(self, nb_afferent_neurons = 100,
                             nb_turnon_neurons = 100,
                             nb_turnoff_neurons = 100,
                             nb_decision_neurons = 2000,
                             nb_efferent_neurons = 100,
                             tau = 1000):
        
        white_noise = WhiteNoise(dist=nengo.dists.Gaussian(0, 0.03), seed = 1)
        
        def feedback(x):
            
            # Set Attractor Constants Note tau = 0.1 by default
            sigma = 40
            beta = 8.0/3
            rho = 28.
            
            # Set Attractor recursive equations.
            dx0 = -sigma * x[0] + sigma * x[1]
            dx1 = -x[0] * x[2] - x[1]
            dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
            return [dx0 * tau + x[0], dx1 * tau + x[1], dx2 * tau + x[2]]
        
        self.NeuralNetwork = nengo.Network(label = 'Neural Network of the Cell Culture')
        
        with self.NeuralNetwork:
            
            # Find the inhibitory vs excitaroy in one population
            # Different neuron populations
            self.AfferentNeuralNetwork = nengo.Network(label = 'Afferent Artificial Neural Network')
            self.AfferentNeuralNetwork = nengo.Ensemble(nb_afferent_neurons, dimensions = 1, neuron_type = nengo.AdaptiveLIF(), noise = white_noise)
            self.CoalitionOn = nengo.Network(label = 'Neural Coalition Attempting to Turn On the Switch')
            self.CoalitionOn = nengo.Ensemble(nb_turnon_neurons, dimensions = 1, neuron_type = nengo.AdaptiveLIF(), noise = white_noise)
            self.CoalitionOff = nengo.Network(label = 'Neural Coalition Attempting to Turn Off the Switch')
            self.CoalitionOff = nengo.Ensemble(nb_turnoff_neurons, dimensions = 1, neuron_type = nengo.AdaptiveLIF(), noise = white_noise)
            self.DecisionCoalition = nengo.Network(label = 'Neural Decision')
            self.DecisionCoalition = nengo.Ensemble(nb_decision_neurons, dimensions = 3, neuron_type = nengo.AdaptiveLIF(), noise = white_noise)
            self.EfferentNeuralNetwork = nengo.Network(label = 'Neural Output')
            self.EfferentNeuralNetwork = nengo.Ensemble(nb_efferent_neurons, dimensions = 1, neuron_type = nengo.AdaptiveLIF(), noise = white_noise)
            
            # A vector Oja (Voja) rule for learning associations aka inputs discriminating between the importance of inputs
            voja = nengo.Voja(learning_rate = 5e-9) # A vector Oja rule
            
            # Node Input to Input
            self.CurrentTNode = nengo.Node([self.CurrentStimulusMagnitude])
            self.CurrentTNode.label = str(self.StimulusTime)
            self.CurrentCNodeAnn = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeAnnlabel = str(self.StimulusTime)
            self.CurrentCNodeCon = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeConlabel = str(self.StimulusTime)
            self.CurrentCNodeCoff = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeCofflabel = str(self.StimulusTime)
            self.CurrentCNodeD = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeDlabel = str(self.StimulusTime)
            self.CurrentCNodeEnn = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeEnnlabel = str(self.StimulusTime)
            
            # Connect Temperature Input
            self.InT2Ann = nengo.Connection(self.CurrentTNode, self.AfferentNeuralNetwork, synapse = None, learning_rule_type = voja)
            
            # Connect Hit Point Reward Input
            
            # Ann
            self.InC2Ann = nengo.Connection(self.CurrentCNodeAnn, self.AfferentNeuralNetwork, synapse = None, learning_rule_type = voja)
            
            # Con
            self.InC2Con = nengo.Connection(self.CurrentCNodeCon, self.CoalitionOn, synapse = None, learning_rule_type = voja)
            
            # Coff
            self.InC2Coff = nengo.Connection(self.CurrentCNodeCoff, self.CoalitionOff, synapse = None, learning_rule_type = voja)
            
            # D
            self.InC2D1 = nengo.Connection(self.CurrentCNodeD, self.DecisionCoalition[0], synapse = None, learning_rule_type = voja)
            self.InC2D2 = nengo.Connection(self.CurrentCNodeD, self.DecisionCoalition[1], synapse = None, learning_rule_type = voja)
            self.InC2D3 = nengo.Connection(self.CurrentCNodeD, self.DecisionCoalition[2], synapse = None, learning_rule_type = voja)
            
            # Enn
            self.InC2Enn = nengo.Connection(self.CurrentCNodeEnn, self.EfferentNeuralNetwork, synapse = None, learning_rule_type = voja)
            
            # Connect Input to On Coalition
            self.Ann2Con = nengo.Connection(self.AfferentNeuralNetwork, self.CoalitionOn, solver = nengo.solvers.LstsqL2(weights = True))
            self.Ann2Con.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Input to Off Coalition
            self.Ann2Coff = nengo.Connection(self.AfferentNeuralNetwork, self.CoalitionOff, solver = nengo.solvers.LstsqL2(weights = True))
            self.Ann2Coff.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect On Coalition On to Off Coalition
            self.Con2Coff = nengo.Connection(self.CoalitionOn, self.CoalitionOff, solver = nengo.solvers.LstsqL2(weights = True))
            self.Con2Coff.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect On Coalition Off to On Coalition
            self.Coff2Con = nengo.Connection(self.CoalitionOff, self.CoalitionOn, solver = nengo.solvers.LstsqL2(weights = True))
            self.Coff2Con.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect On Coalition to Decision
            self.Con2D1 = nengo.Connection(self.CoalitionOn, self.DecisionCoalition[0], solver = nengo.solvers.LstsqL2(weights = True))
            self.Con2D1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.Con2D2 = nengo.Connection(self.CoalitionOn, self.DecisionCoalition[1], solver = nengo.solvers.LstsqL2(weights = True))
            self.Con2D2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.Con2D3 = nengo.Connection(self.CoalitionOn, self.DecisionCoalition[2], solver = nengo.solvers.LstsqL2(weights = True))
            self.Con2D3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Off Coalition to Decision
            self.Coff2D1 = nengo.Connection(self.CoalitionOff, self.DecisionCoalition[0], solver = nengo.solvers.LstsqL2(weights = True))
            self.Coff2D1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.Coff2D2 = nengo.Connection(self.CoalitionOff, self.DecisionCoalition[1], solver = nengo.solvers.LstsqL2(weights = True))
            self.Coff2D2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.Coff2D3 = nengo.Connection(self.CoalitionOff, self.DecisionCoalition[2], solver = nengo.solvers.LstsqL2(weights = True))
            self.Coff2D3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Reccurrent Connection
            self.D2D = nengo.Connection(self.DecisionCoalition, self.DecisionCoalition, function = feedback, synapse = tau, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2D.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Decision to On Coalition Feedback 1
            self.D2Con1 = nengo.Connection(self.DecisionCoalition[0], self.CoalitionOn, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Con1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Con2 = nengo.Connection(self.DecisionCoalition[1], self.CoalitionOn, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Con2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Con3 = nengo.Connection(self.DecisionCoalition[2], self.CoalitionOn, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Con3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Decision to Off Coalition Feedback 3
            self.D2Coff1 = nengo.Connection(self.DecisionCoalition[0], self.CoalitionOff, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Coff1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Coff2 = nengo.Connection(self.DecisionCoalition[1], self.CoalitionOff, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Coff2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Coff3 = nengo.Connection(self.DecisionCoalition[2], self.CoalitionOff, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Coff3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Decision Coalition with Input Feedback 3
            self.D2Ann1 = nengo.Connection(self.DecisionCoalition[0], self.AfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Ann1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Ann2 = nengo.Connection(self.DecisionCoalition[1], self.AfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Ann2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Ann3 = nengo.Connection(self.DecisionCoalition[2], self.AfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Ann3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            
            # Connect Decision to Output
            self.D2Enn1 = nengo.Connection(self.DecisionCoalition[0], self.EfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Enn1.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Enn2 = nengo.Connection(self.DecisionCoalition[1], self.EfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Enn2.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
            self.D2Enn3 = nengo.Connection(self.DecisionCoalition[2], self.EfferentNeuralNetwork, solver = nengo.solvers.LstsqL2(weights = True))
            self.D2Enn3.learning_rule_type = [nengo.BCM(learning_rate=5e-10), nengo.Oja(learning_rate=2e-9, beta = 15.9375)]
        
    def stimulate_cell_culture(self, temperature, time, sampling_time = 0.001):
        
        self.update_state(temperature, time)
        
        with self.NeuralNetwork:
            
            self.CurrentTNode = nengo.Node([self.CurrentStimulusMagnitude])
            self.CurrentTNode.label = str(self.StimulusTime)
            self.CurrentCNodeAnn = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeAnnlabel = str(self.StimulusTime)
            self.CurrentCNodeCon = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeConlabel = str(self.StimulusTime)
            self.CurrentCNodeCoff = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeCofflabel = str(self.StimulusTime)
            self.CurrentCNodeD = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeDlabel = str(self.StimulusTime)
            self.CurrentCNodeEnn = nengo.Node([self.hpcost2v()])
            self.CurrentCNodeEnnlabel = str(self.StimulusTime)
            
    def create_probes(self, sampling_time = 0.001):
        
        with self.NeuralNetwork:
            
            # Probe electrical activity
            self.InputProbe = nengo.Probe(self.AfferentNeuralNetwork, 'decoded_output', synapse = sampling_time, sample_every = sampling_time)
            self.OnProbe = nengo.Probe(self.CoalitionOn, 'decoded_output', synapse = sampling_time, sample_every = sampling_time)
            self.OffProbe = nengo.Probe(self.CoalitionOff, 'decoded_output', synapse = sampling_time, sample_every = sampling_time)
            self.DecisionProbe = nengo.Probe(self.DecisionCoalition, 'decoded_output', synapse = sampling_time, sample_every = sampling_time)
            self.OutputProbe = nengo.Probe(self.EfferentNeuralNetwork, 'decoded_output', synapse = sampling_time, sample_every = sampling_time)
            
            # Probe weight strength
            self.weightsProbeInT2Ann = nengo.Probe(self.InT2Ann, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2Ann = nengo.Probe(self.InC2Ann, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2Con = nengo.Probe(self.InC2Con, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2Coff = nengo.Probe(self.InC2Coff, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2D1 = nengo.Probe(self.InC2D1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2D2 = nengo.Probe(self.InC2D2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2D3 = nengo.Probe(self.InC2D3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeInC2Enn = nengo.Probe(self.InC2Enn, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeAnn2Con = nengo.Probe(self.Ann2Con, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeAnn2Coff = nengo.Probe(self.Ann2Coff, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCon2Coff = nengo.Probe(self.Con2Coff, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCoff2Con = nengo.Probe(self.Coff2Con, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2D = nengo.Probe(self.D2D, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCon2D1 = nengo.Probe(self.Con2D1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCon2D2 = nengo.Probe(self.Con2D2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCon2D3 = nengo.Probe(self.Con2D3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCoff2D1 = nengo.Probe(self.Coff2D1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCoff2D2 = nengo.Probe(self.Coff2D2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeCoff2D3 = nengo.Probe(self.Coff2D3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Con1 = nengo.Probe(self.D2Con1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Con2 = nengo.Probe(self.D2Con2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Con3 = nengo.Probe(self.D2Con3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Coff1 = nengo.Probe(self.D2Coff1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Coff2 = nengo.Probe(self.D2Coff2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Coff3 = nengo.Probe(self.D2Coff3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Ann1 = nengo.Probe(self.D2Ann1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Ann2 = nengo.Probe(self.D2Ann2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Ann3 = nengo.Probe(self.D2Ann3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Enn1 = nengo.Probe(self.D2Enn1, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Enn2 = nengo.Probe(self.D2Enn2, 'weights', synapse = sampling_time, sample_every = sampling_time)
            self.weightsProbeD2Enn3 = nengo.Probe(self.D2Enn3, 'weights', synapse = sampling_time, sample_every = sampling_time)
            
            # Probe spike activity in each population
            self.InputSpikesProbe = nengo.Probe(self.AfferentNeuralNetwork.neurons, 'spikes', synapse = sampling_time, sample_every = sampling_time)
            self.OnSpikesProbe = nengo.Probe(self.CoalitionOn.neurons, 'spikes', synapse = sampling_time, sample_every = sampling_time)
            self.OffSpikesProbe = nengo.Probe(self.CoalitionOff.neurons, 'spikes', synapse = sampling_time, sample_every = sampling_time)
            self.DecisionSpikesProbe = nengo.Probe(self.DecisionCoalition.neurons, 'spikes', synapse = sampling_time, sample_every = sampling_time)
            self.OutputSpikesProbe = nengo.Probe(self.EfferentNeuralNetwork.neurons, 'spikes', synapse = sampling_time, sample_every = sampling_time)
            
    def simulate_one_NeuralNetwork_time_instance(self, temperature, time, sampling_time = 0.001):
        
        # Run Only in a simulator
        self.stimulate_cell_culture(temperature, time, sampling_time)
        self.NN_sim.step()
        #nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')
        
    def display_network_states(self):
        
        ### Get simulator Data for Decoded outputs, weights, simulation times and data preprocessing
        
        print('Analysis of the states of the cell culture starts')
        
        # time
        simulation_times = self.NN_sim.trange()
        
        # Ensemble Data
        EIn = self.NN_sim.data[self.AfferentNeuralNetwork]
        EOn = self.NN_sim.data[self.CoalitionOn]
        EOff = self.NN_sim.data[self.CoalitionOff]
        ED = self.NN_sim.data[self.DecisionCoalition]
        EOut = self.NN_sim.data[self.EfferentNeuralNetwork]
        
        # Deccoded output
        DoutIn = self.NN_sim.data[self.InputProbe]
        DoutOn = self.NN_sim.data[self.OnProbe]
        DoutOff = self.NN_sim.data[self.OffProbe]
        DoutD = self.NN_sim.data[self.DecisionProbe]
        DoutD1 = DoutD[:, 0]
        DoutD2 = DoutD[:, 1]
        DoutD3 = DoutD[:, 2]
        DoutOut = self.NN_sim.data[self.OutputProbe]
        
        # Spikes
        SIn = self.NN_sim.data[self.InputSpikesProbe]
        SOn = self.NN_sim.data[self.OnSpikesProbe]
        SOff = self.NN_sim.data[self.OffSpikesProbe]
        SD = self.NN_sim.data[self.DecisionSpikesProbe]
        SOut = self.NN_sim.data[self.OutputSpikesProbe]
        
        ## Weights
        
        # Take final weights for connection between ensembles
        
        # T2Ann
        wFInT2Ann = self.NN_sim.data[self.weightsProbeInT2Ann]
        wFInT2Ann = wFInT2Ann[-1, :]
        wFInT2Ann = wFInT2Ann * np.ones(self.AfferentNeuralNetwork.n_neurons)
        
        # C2Ann
        wFInC2Ann = self.NN_sim.data[self.weightsProbeInC2Ann]
        wFInC2Ann = wFInC2Ann[-1, :]
        wFInC2Ann = wFInC2Ann * np.ones(self.AfferentNeuralNetwork.n_neurons)
        
        # C2Con
        wFInC2Con = self.NN_sim.data[self.weightsProbeInC2Con]
        wFInC2Con = wFInC2Con[-1, :]
        wFInC2Con = wFInC2Con * np.ones(self.CoalitionOn.n_neurons)
        
        # C2Coff
        wFInC2Coff = self.NN_sim.data[self.weightsProbeInC2Coff]
        wFInC2Coff = wFInC2Coff[-1, :]
        wFInC2Coff = wFInC2Coff * np.ones(self.CoalitionOff.n_neurons)
        
        # C2D1
        wFInC2D1 = self.NN_sim.data[self.weightsProbeInC2D1]
        wFInC2D1 =  wFInC2D1[-1, :]
        wFInC2D1 = wFInC2D1 * np.ones(self.DecisionCoalition.n_neurons)
        
        # C2D2
        wFInC2D2 = self.NN_sim.data[self.weightsProbeInC2D2]
        wFInC2D2 =  wFInC2D2[-1, :]
        wFInC2D2 = wFInC2D2 * np.ones(self.DecisionCoalition.n_neurons)
        
        # C2D3
        wFInC2D3 = self.NN_sim.data[self.weightsProbeInC2D3]
        wFInC2D3 =  wFInC2D3[-1, :]
        wFInC2D3 = wFInC2D3 * np.ones(self.DecisionCoalition.n_neurons)
        
        # C2D
        wFInC2D = (wFInC2D1 + wFInC2D2 + wFInC2D3) / 3
        
        # C2Enn
        wFInC2Enn = self.NN_sim.data[self.weightsProbeInC2Enn]
        wFInC2Enn = wFInC2Enn[-1, :]
        wFInC2Enn = wFInC2Enn * np.ones(self.EfferentNeuralNetwork.n_neurons)
        
        # Ann2Con
        wFAnn2Con = self.NN_sim.data[self.weightsProbeAnn2Con]
        wFAnn2Con = wFAnn2Con[-1, :]
        
        # Ann2Coff
        wFAnn2Coff = self.NN_sim.data[self.weightsProbeAnn2Coff]
        wFAnn2Coff = wFAnn2Coff[-1, :]
        
        # Con2Coff
        wFCon2Coff = self.NN_sim.data[self.weightsProbeCon2Coff]
        wFCon2Coff = wFCon2Coff[-1, :]
        
        # Coff2Con
        wFCoff2Con = self.NN_sim.data[self.weightsProbeCoff2Con]
        wFCoff2Con = wFCoff2Con[-1, :]
        
        # D2D
        wFD2D = self.NN_sim.data[self.weightsProbeD2D]
        wFD2D = wFD2D[-1, :]
        
        # Con2D1
        wFCon2D1 = self.NN_sim.data[self.weightsProbeCon2D1]
        wFCon2D1 = wFCon2D1[-1, :]
        
        # Con2D2
        wFCon2D2 = self.NN_sim.data[self.weightsProbeCon2D2]
        wFCon2D2 = wFCon2D2[-1, :]
        
        # Con2D3
        wFCon2D3 = self.NN_sim.data[self.weightsProbeCon2D3]
        wFCon2D3 = wFCon2D3[-1, :]
        
        # Con2D
        wFCon2D = (wFCon2D1 + wFCon2D2 + wFCon2D3) / 3
        
        # Coff2D1
        wFCoff2D1 = self.NN_sim.data[self.weightsProbeCoff2D1]
        wFCoff2D1 = wFCoff2D1[-1, :]
        
        # Coff2D2
        wFCoff2D2 = self.NN_sim.data[self.weightsProbeCoff2D2]
        wFCoff2D2 = wFCoff2D2[-1, :]
        
        # Coff2D3
        wFCoff2D3 = self.NN_sim.data[self.weightsProbeCoff2D3]
        wFCoff2D3 = wFCoff2D3[-1, :]
        
        # Coff2D
        wFCoff2D = (wFCoff2D1 + wFCoff2D2 + wFCoff2D3) / 3
        
        # D2Con1
        wFD2Con1 = self.NN_sim.data[self.weightsProbeD2Con1]
        wFD2Con1 = wFD2Con1[-1, :]
        
        # D2Con2
        wFD2Con2 = self.NN_sim.data[self.weightsProbeD2Con2]
        wFD2Con2 = wFD2Con2[-1, :]
        
        # D2Con3
        wFD2Con3 = self.NN_sim.data[self.weightsProbeD2Con3]
        wFD2Con3 = wFD2Con3[-1, :]
        
        # D2Con
        wFD2Con = (wFD2Con1 + wFD2Con2 + wFD2Con3) / 3
        
        # D2Coff1
        wFD2Coff1 = self.NN_sim.data[self.weightsProbeD2Coff1]
        wFD2Coff1 = wFD2Coff1[-1, :]
        
        # D2Coff2
        wFD2Coff2 = self.NN_sim.data[self.weightsProbeD2Coff2]
        wFD2Coff2 = wFD2Coff2[-1, :]
        
        # D2Coff3
        wFD2Coff3 = self.NN_sim.data[self.weightsProbeD2Coff3]
        wFD2Coff3 = wFD2Coff3[-1, :]
        
        # D2Coff
        wFD2Coff = (wFD2Coff1 + wFD2Coff2 + wFD2Coff3) / 3
        
        # D2Ann1
        wFD2Ann1 = self.NN_sim.data[self.weightsProbeD2Ann1]
        wFD2Ann1 = wFD2Ann1[-1, :]
        
        # D2Ann2
        wFD2Ann2 = self.NN_sim.data[self.weightsProbeD2Ann2]
        wFD2Ann2 = wFD2Ann2[-1]
        
        # D2Ann3
        wFD2Ann3 = self.NN_sim.data[self.weightsProbeD2Ann3]
        wFD2Ann3 = wFD2Ann3[-1, :]
        
        # D2Ann
        wFD2Ann = (wFD2Ann1 + wFD2Ann2 + wFD2Ann3) / 3
        
        # D2Enn1
        wFD2Enn1 = self.NN_sim.data[self.weightsProbeD2Enn1]
        wFD2Enn1 = wFD2Enn1[-1, :]
        
        # D2Enn2
        wFD2Enn2 = self.NN_sim.data[self.weightsProbeD2Enn2]
        wFD2Enn2 = wFD2Enn2[-1, :]
        
        # D2Enn3
        wFD2Enn3 = self.NN_sim.data[self.weightsProbeD2Enn3]
        wFD2Enn3 = wFD2Enn3[-1, :]
        
        # D2Enn
        wFD2Enn = (wFD2Enn1 + wFD2Enn2 + wFD2Enn3) / 3
        
        # weights in ensembles
        ensembleWeightsSolver = nengo.solvers.LstsqL2(weights=True)
        
        # ANN weights
        xANN = np.dot(EIn.eval_points, EIn.encoders.T / self.AfferentNeuralNetwork.radius)
        A_ANN = self.AfferentNeuralNetwork.neuron_type.rates(xANN, EIn.gain, EIn.bias)
        ANNWeights, _ = ensembleWeightsSolver(A_ANN, EIn.eval_points, E = EIn.scaled_encoders.T)
        
        # OnNC weights
        xOnNC = np.dot(EOn.eval_points, EOn.encoders.T / self.CoalitionOn.radius)
        A_OnNC = self.CoalitionOn.neuron_type.rates(xOnNC, EOn.gain, EOn.bias)
        OnNCWeights, _ = ensembleWeightsSolver(A_OnNC, EOn.eval_points, E = EOn.scaled_encoders.T)
        
        # OffNC weights
        xOffNC = np.dot(EOff.eval_points, EOff.encoders.T / self.CoalitionOff.radius)
        A_OffNC = self.CoalitionOff.neuron_type.rates(xOffNC, EOff.gain, EOff.bias)
        OffNCWeights, _ = ensembleWeightsSolver(A_OffNC, EOff.eval_points, E = EOff.scaled_encoders.T)
        
        # D weights
        xDNC = np.dot(ED.eval_points, ED.encoders.T / self.DecisionCoalition.radius)
        A_DNC = self.DecisionCoalition.neuron_type.rates(xDNC, ED.gain, ED.bias)
        DNCWeights, _ = ensembleWeightsSolver(A_DNC, ED.eval_points, E = ED.scaled_encoders.T)
        DNCWeights = DNCWeights
        DNCWeights = (DNCWeights + wFD2D) / 2
        
        # ENN weights
        xENN = np.dot(EOut.eval_points, EOut.encoders.T / self.EfferentNeuralNetwork.radius)
        A_ENN = self.EfferentNeuralNetwork.neuron_type.rates(xENN, EOut.gain, EOut.bias)
        ENNWeights, _ = ensembleWeightsSolver(A_ENN, EOut.eval_points, E = EOut.scaled_encoders.T)
        
        # Weight Map aka Connectome of the cell culture
        tot_neurons = self.AfferentNeuralNetwork.n_neurons + self.CoalitionOn.n_neurons + self.CoalitionOff.n_neurons + self.DecisionCoalition.n_neurons + self.EfferentNeuralNetwork.n_neurons
        
        row1 = np.concatenate((wFInT2Ann,
                               np.transpose(np.zeros(tot_neurons - self.AfferentNeuralNetwork.n_neurons))))
        row1 = np.reshape(row1, (1, tot_neurons))
        
        row2 = np.concatenate((wFInC2Ann,
                               wFInC2Con,
                               wFInC2Coff,
                               wFInC2D,
                               wFInC2Enn))
        row2 = np.reshape(row2, (1, tot_neurons))
        
        row3 = np.concatenate((ANNWeights,
                               wFAnn2Con,
                               wFAnn2Coff,
                               np.zeros((self.DecisionCoalition.n_neurons, self.AfferentNeuralNetwork.n_neurons)),
                               np.zeros((self.EfferentNeuralNetwork.n_neurons, self.AfferentNeuralNetwork.n_neurons))))
        row3 = np.reshape(row3, (self.AfferentNeuralNetwork.n_neurons, tot_neurons))
        
        
        row4 = np.concatenate((np.zeros((self.AfferentNeuralNetwork.n_neurons, self.CoalitionOn.n_neurons)),
                               OnNCWeights,
                               wFCon2Coff.T,
                               wFCon2D,
                               np.transpose(np.zeros((self.CoalitionOn.n_neurons, self.EfferentNeuralNetwork.n_neurons)))))
        row4 = np.reshape(row4, (self.CoalitionOn.n_neurons, tot_neurons))
        
        row5 = np.concatenate((np.zeros((self.AfferentNeuralNetwork.n_neurons, self.CoalitionOff.n_neurons)),
                               wFCoff2Con.T,
                               OffNCWeights,
                               wFCoff2D,
                               np.transpose(np.zeros((self.CoalitionOff.n_neurons, self.EfferentNeuralNetwork.n_neurons)))))
        row5 = np.reshape(row5, (self.CoalitionOn.n_neurons, tot_neurons))
        
        row6 = np.concatenate((wFD2Ann, wFD2Con, wFD2Coff, DNCWeights, wFD2Enn))
        row6 = np.reshape(row6, (self.DecisionCoalition.n_neurons, tot_neurons))
        
        row7 = np.concatenate((np.transpose(np.zeros((self.EfferentNeuralNetwork.n_neurons, tot_neurons - self.EfferentNeuralNetwork.n_neurons))),
                               ENNWeights))
        row7 = np.reshape(row7, (self.EfferentNeuralNetwork.n_neurons, tot_neurons))
        
        Connectome = np.concatenate((row1, row2, row3, row4, row5, row6, row7))
        Cs = Connectome.shape
        
        
        ## Weight Map aka. Connectome
        for i in range(0, Cs[0]):
            
            for j in range(0, Cs[1]):
                
                if Connectome[i, j] > 0:
                    
                    Connectome[i, j] = 1
                    
                if Connectome[i, j] == 0:
                    
                    Connectome[i, j] = 0
                
                if Connectome[i, j] < 0:
                    
                    Connectome[i, j] = -1
                
                if ((j > 2 + self.AfferentNeuralNetwork.n_neurons + self.CoalitionOn.n_neurons + self.CoalitionOff.n_neurons) or (i > self.AfferentNeuralNetwork.n_neurons + self.CoalitionOn.n_neurons + self.CoalitionOff.n_neurons)) and ((j < 2 + self.AfferentNeuralNetwork.n_neurons + self.CoalitionOn.n_neurons + self.CoalitionOff.n_neurons + self.DecisionCoalition.n_neurons) or (i < self.AfferentNeuralNetwork.n_neurons + self.CoalitionOn.n_neurons + self.CoalitionOff.n_neurons + self.DecisionCoalition.n_neurons + 2)):
                    
                    Connectome[i, j] = 3 * Connectome[i, j]
        
        print('preprocessing done ...')
        
        print('There are', Cs[0] * Cs[1], 'connections in the neuron cell culture.')
        
        ### Display Data
        
        ## Plot Decoded outputs
        
        # ANN Decoded Output
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, DoutIn)
        plt.title('Voltage of the Afferent Neural Network as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 18})
        plt.show()
        
        # OnNC 
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, DoutOn)
        plt.title('Voltage of the On Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 18})
        plt.show()
        
        # Off Coalition
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, DoutOff)
        plt.title('Voltage of the Off Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 18})
        plt.show()
        
        # Decision Attractor
        fig = plt.figure(figsize = (14, 10))
        ax = fig.gca(projection='3d')
        ax.plot(DoutD1, DoutD2, DoutD3)
        plt.title('Attractor Dynamics in Decision Coalition', {'fontsize': 20}, pad = 20)
        ax.set_xlabel('Voltage in Millivolts [mV] in C1', fontsize = 16, labelpad = 10)
        ax.set_ylabel('Voltage in Millivolts [mV] in C2', fontsize = 16, labelpad = 10)
        ax.set_zlabel('Voltage in Millivolts [mV] in C3', fontsize = 16, labelpad = 10)
        plt.show()
        
        # Decision Coalition.
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, DoutD1, label = 'Decision Coalition C1')
        plt.plot(simulation_times, DoutD2, label = 'Decision Coalition C2')
        plt.plot(simulation_times, DoutD3, label = 'Decision Coalition C3')
        plt.title('Voltage of the Decision Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 18})
        plt.legend(prop={'size': 12})
        plt.show()
        
        # ENN
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, DoutOut)
        plt.title('Voltage of the Efferent Neural Network as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Voltage in Millivolts [mV]', {'fontsize': 18})
        plt.show()
        
        ## Temperature "felt" by the network
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, self.Temperatures)
        plt.title('Temperature of the Cell Culture as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 16})
        plt.ylabel('Temperature in Degrees Celsious [C]', {'fontsize': 18})
        plt.show()
        
        ## Raster plots
        
        # ANN
        plt.figure(figsize = (10, 6))
        rasterplot(simulation_times, SIn)
        plt.title('Spikes of the Afferent Neural Network as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Neuron Index', {'fontsize': 18})
        plt.show()
        
        # On Coalition
        plt.figure(figsize = (10, 6))
        rasterplot(simulation_times, SOn)
        plt.title('Spikes of the On Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Neuron Index', {'fontsize': 18})
        plt.show()
        
        # Off Coalition
        plt.figure(figsize = (10, 6))
        rasterplot(simulation_times, SOff)
        plt.title('Spikes of the Off Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Neuron Index', {'fontsize': 18})
        plt.show()
        
        # Decision Coalition
        plt.figure(figsize = (10, 6))
        rasterplot(simulation_times, SD)
        plt.title('Spikes of the Decision Coalition as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Neuron Index', {'fontsize': 18})
        plt.show()
        
        # ENN
        plt.figure(figsize = (10, 6))
        rasterplot(simulation_times, SOut)
        plt.title('Spikes of the Efferent Neurons as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Neuron Index', {'fontsize': 18})
        plt.show()
        
        ## HP Monitoring
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, self.HPMemory)
        plt.title('Hit Points of the Cell Culture as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Hit Points Arbitrary Units [A.U.]', {'fontsize': 18})
        plt.show()
        
        # HP Loss Monitoring
        plt.figure(figsize = (10, 6))
        plt.plot(simulation_times, self.HPLossMemory)
        plt.title('Hit Point Reward as a Function of Time', {'fontsize': 20})
        plt.xlabel('Time in Seconds [s]', {'fontsize': 18})
        plt.ylabel('Hit Points in Arbitrary Units [A.U.]', {'fontsize': 18})
        plt.show()
        
        print('COLORMESH of WEIGHTMAP ...')
        
        # Plot the Connectome as a weightmap
        plt.figure(figsize = (8, 6))
        plt.pcolormesh(Connectome, cmap = 'coolwarm')
        plt.xlabel('Neuron Index in Units [U]', {'fontsize': 16})
        plt.ylabel('Neuron Index in Units [U]', {'fontsize': 16})
        plt.title('Cell Culture Connectome or Weight Map', {'fontsize': 20})
        c1 = plt.colorbar()
        c1.set_label('Weight Strength in Arbitrary Units [A.U.]', fontsize = 16, labelpad = 10)
        plt.show()
        
        print('Analysis of the states of the cell culture ends')
        
    def simulate_NN_in_experiment(self, temperatures, times, sampling_time = 0.001):
        
        experiment_length = np.size(temperatures)
        
        # number of time instances in the experiment is equal to the number of temperatures as
        # real_time version
        self.create_probes()
        self.NN_sim = nengo.Simulator(self.NeuralNetwork)
        
        with self.NN_sim:
            
            for i in range(0, experiment_length):
                
                self.simulate_one_NeuralNetwork_time_instance(temperatures[i], times[i], sampling_time)
                nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')
            
            self.display_network_states()


if __name__ == "__main__":
    
    # Simple Temperature Ramp Time Serie
    temperatureInstances = np.linspace(-10, 40, 1000) # Temperatures in degree Celsius.
    timeInstances = np.linspace(0, 1000, 1000) # Time instances.
    
    #  Creating neuralnet for temp test
    
    # Cell Culture Parameters
    initial_temperature = temperatureInstances[0]
    initial_time = timeInstances[0]
    diameter = 10
    height = 1
    conduction = 0.2
    position_x = 0
    position_y = 0
    position_z = 10
    
    # Neural Network Parameters: Use less neurons ! 
    nb_afferent_neurons = 50
    nb_turnon_neurons = 30
    nb_turnoff_neurons = 30
    nb_decision_neurons = 300
    nb_efferent_neurons = 50
    
    # initialising a cell culture.
    cc1 = CellCulture(initial_temperature, initial_time, diameter, height, conduction, position_x, position_y, position_z)
    
    cc1.create_NeuralNetwork(nb_afferent_neurons,
                             nb_turnon_neurons,
                             nb_turnoff_neurons,
                             nb_decision_neurons,
                             nb_efferent_neurons)
    
    cc1.simulate_NN_in_experiment(temperatureInstances, timeInstances)




    