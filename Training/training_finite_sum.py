#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:00:05 2024

@author: carlosesteveyague
"""

import torch
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, points_int, points_bound, f, g):
        'Initialization'
        self.f = f
        self.g = g
        
        self.points = torch.concat( (points_int, points_bound), dim = 0)
        
        self.is_int = torch.concat((torch.ones(points_int.shape[0]), 
                                   torch.zeros(points_bound.shape[0])))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.points)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.points[index]
        
        is_int = self.is_int[index]
        
        y = self.f(self.points[index])*(is_int == 1) + self.g(self.points[index])*(is_int == 0)

        return X, y, is_int

def train_finite_sum(NN, int_points, bound_points, training_params):
    
    f = training_params['f'] ## Right hand side of the equation
    g = training_params['g'] ## Initial condition
    training_data = Dataset(int_points, bound_points, f, g)
        
    num_scheme = training_params['numerical_scheme']
    c = training_params['c'] ## Term multiplying |\nabla u(x)|    
    
    delta = training_params['delta']
    alpha = training_params['alpha']
    beta = training_params['beta']
    optimizer = training_params['optimizer']
    
    epochs = training_params['epochs']
    
    lam = training_params['lambda']    
    
    # Generator
    params = {
            'batch_size': training_params['batch_size'],
            'shuffle': True
            }
    
    training_loss = torch.zeros(epochs)
    training_PDE_loss = torch.zeros(epochs)
    training_boundary_loss = torch.zeros(epochs)
                
    training_generator = torch.utils.data.DataLoader(training_data, **params)
    
    for epoch in tqdm(range(epochs)):
        
        # Training
        trainLoss = 0.
        trainBoundLoss = 0.
        trainPDELoss = 0.
        samples = 0
        samples_int = 0
        samples_bound = 0
        
        for x, y, is_int in training_generator:
            
            optimizer.zero_grad()
            
            interior_points = x[is_int == 1]
            boundary_points = x[is_int == 0]
            
            if interior_points.size(0) >0:
                H_points = num_scheme(NN, interior_points, delta, alpha, beta, c)
                PDE_loss = ((H_points - y[is_int == 1])**2).mean()
            else:
                PDE_loss = torch.zeros(1)
            
            if boundary_points.size(0) >0:
                boundary_loss = ((NN(boundary_points).squeeze() - y[is_int == 0])**2).mean()
            else:
                boundary_loss = torch.zeros(1)
          
        
            loss =  PDE_loss + lam*boundary_loss
            loss.backward()
            optimizer.step()
            
            trainLoss += loss.item()*x.size(0)
            trainPDELoss += PDE_loss.item()*interior_points.size(0)
            trainBoundLoss += boundary_loss.item()*boundary_points.size(0)
            samples += x.size(0)
            samples_int += interior_points.size(0)
            samples_bound += boundary_points.size(0)
        
        
        training_loss[epoch] = trainLoss/samples
        training_PDE_loss[epoch] = trainPDELoss/samples_int
        training_boundary_loss[epoch] = trainBoundLoss/samples_bound
    
    return training_loss, training_PDE_loss, training_boundary_loss