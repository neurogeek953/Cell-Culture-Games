#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:40:45 2019

@author: TEB
"""

import numpy as np
from numpy.random import normal

# Renyi Entropy 
# Compute mutual information directly
# Dependency 
# Bishop book.

def hop(probability):
    
    if probability == 0:
        
        entropy = 0
        
    else:
        
        entropy = - probability * np.log2(probability)
        
    return entropy

def compute_entropy(probabilities):
    
    """Compute information entropy."""
    
    entropy = 0.
    
    for i in range(0, len(probabilities)):
        
        entropy = entropy - probabilities[i] * np.log2(probabilities[i])
        
    return entropy


def compute_direct_mutual_information(probability_stimulus, probability_response):
    
    if len(probability_response) != len(probability_stimulus):
        
        print("The number of events is not the same.  Mutual information analysis is not possible. --> RETURN None")
        return None
    
    entropy = 0.
    probability_response_and_stimulus = []
    # Prands = np.array([(1 - Ps) * (1 - Pr), Ps * (1 - Pr), (1 - Ps) * Pr, Ps * Pr])
    
    for i in range(0, len(probability_response)):
    
        probability_response_and_stimulus.append(probability_response[i] * probability_stimulus[i])
        entropy = entropy + probability_response_and_stimulus[i] * np.log2(probability_response_and_stimulus[i] / (probability_stimulus[i] * probability_response[i]))
            
    return entropy
            
            
    


def compute_mutual_information(probabilities_response, probabilities_response_if_stimulus):
    
    if len(probabilities_response_if_stimulus) != len(probabilities_response):
        
        print("The number of events is not the same.  Mutual information analysis is not possible. --> RETURN None")
        return None
    
    mutual_information = compute_entropy(probabilities_response) - compute_entropy(probabilities_response_if_stimulus)
    return mutual_information

if __name__ == "__main__":
    
    mu = 10
    sigma = 0.25
    Ibar = 0
    I2bar = 0
    
    for i in range(0, 10):
    
        Ps = np.array(normal(mu, sigma, 10000)) # Normal distribution with standard deviation 5 and mean 6
        Ps = Ps / sum(Ps) # normalise Prs
        Pr = np.array(normal(mu, sigma, 10000)) # Normal distribution with standard deviation 5 and mean 6
        Pr = Pr / sum(Pr) # normalise Pr
        Prands = np.array([(1 - Ps) * (1 - Pr), Ps * (1 - Pr), (1 - Ps) * Pr, Ps * Pr])
        Prs = Prands[3,:] / Pr
        I = compute_mutual_information(Pr, Prs)
        I2 = compute_direct_mutual_information(Ps, Pr)
        Ibar = Ibar + I
        I2bar = I2bar + I2
    
    Ibar = Ibar / (i + 1)
    I2bar = I2bar / (i + 1)   
        
        
