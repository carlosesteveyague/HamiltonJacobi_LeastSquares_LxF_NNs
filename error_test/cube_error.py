#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:40:07 2024

@author: carlosesteveyague
"""

import torch

def error_cube(NN, side_length, n_points):
    
    dim = NN.L1.in_features
    
        
    with torch.no_grad():
        
        X = -1 + 2*torch.rand([n_points, dim])
        X = .5*side_length*X
        X[0] = 0.*X[0]
        
        Y = side_length/2 - X.abs().max(-1)[0]
        Y_hat = NN(X).squeeze()
        
        MSE = ((Y - Y_hat)**2).mean()
        
        L_inf_error = (Y - Y_hat).abs().max()
    
    return MSE, L_inf_error


def FD_loss(NN, test_data, training_params):
    
    H = training_params['numerical_scheme']
    alpha = training_params['alpha']
    beta = training_params['beta']
    d = training_params['delta']
    f = training_params['f']
    c = training_params['c']
    
    W = H(NN, test_data, d, alpha, beta, c) - f(test_data)
    
    return W**2