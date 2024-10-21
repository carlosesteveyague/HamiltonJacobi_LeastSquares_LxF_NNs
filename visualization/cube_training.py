#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:47:33 2024

@author: carlosesteveyague
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np


def plot_2d_proj_w(axis1, axis2, NN, n_grid, side_length, training_params, 
                   colloc_points = None,  dim = None):
    
    
    H = training_params['numerical_scheme']
    alpha = training_params['alpha']
    beta = training_params['beta']
    d = training_params['delta']
    f = training_params['f']
    c = training_params['c']
    
    if dim == None:
        dim = NN.L1.in_features
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)

    Grid = torch.zeros([n_grid, n_grid, dim])
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY
    
    W = H(NN, Grid, d, alpha, beta, c) - f(Grid)
    
    W = W**2

    plt.pcolormesh(GridX.detach(), GridY.detach(), W.detach())
    plt.clim(0, 1)
    plt.title('F.D. LxF residual')
    
    if colloc_points is not None:
        plt.scatter(colloc_points[:, 0], colloc_points[:, 1], c='red')
    
    plt.colorbar()
    plt.show()
    
    return W