#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:45:08 2024

@author: carlosesteveyague
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np


def plot_2d_proj_w(axis1, axis2, NN, n_grid, radii, training_params, 
                   colloc_points = None,  dim = None):
    
    
    H = training_params['numerical_scheme']
    alpha = training_params['alpha']
    beta = training_params['beta']
    d = training_params['delta']
    f = training_params['f']
    c = training_params['c']
    
    if dim == None:
        dim = NN.L1.in_features
    
    Theta_grid = torch.linspace(0., 2*torch.pi, n_grid)
    r_grid = torch.linspace(min(radii), max(radii), n_grid)
    
    Grid_r, Grid_Theta = torch.meshgrid(r_grid, Theta_grid)

    GridX = Grid_r*torch.cos(Grid_Theta)
    GridY = Grid_r*torch.sin(Grid_Theta)

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