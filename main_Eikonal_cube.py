#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:00:16 2024

@author: carlosesteveyague
"""

import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from time import time as t

from NeuralNetworks.NNs import FCFF_3L, FCFF_2L

from PointSampling.Cube import data_gen_cube
from visualization.plots_cube import plot_2d_proj, plot_2d_proj_w

from Hamiltonians.Eikonal_LxF import Eikonal_sq_LF_multiD

from Training.training import train
from error_test.cube_error import error_cube


dim = 5

side_length = 6.

domain = data_gen_cube(side_length, dim)

#Right-hand-side of the PDE
def f(X):    
    return 1

#Boundary condition
def g(X):    
    return 0

delta_list = [.75, .5, .3, .1, .05]
alpha_list = [2.5, 2., 1.5, 1., .5]
N_col_list = [50, 50, 50, 50, 50]
N_b_list = [10, 10, 10, 10, 10]
rounds = len(delta_list)

NN = FCFF_3L([dim,50,50])
#NN = FCFF_2L([dim,20])

training_params = {
    'numerical_scheme': Eikonal_sq_LF_multiD,

    'f': f,
    'g': g,
    'c': None,
    
    'beta': 0.,  ## parameter for the +u_i term
    
    'optimizer': optim.SGD(NN.parameters(), lr = .02, momentum = .2),
    'num_iterations': 2000,
    'lambda': 1. #weight parameter for the boundary loss
    }



n_grid = 100 # Number of grid points for comparison with the ground truth

MSE_history = torch.zeros(rounds)
L_inf_error_history = torch.zeros(rounds)
run_time_history = torch.zeros(rounds)

for i in range(rounds):
    
    training_params['alpha'] = alpha_list[i]
    training_params['delta'] = delta_list[i]
    training_params['n_coloc_points'] = N_col_list[i]
    training_params['n_boundary_points'] = N_b_list[i]
    
    t0 = t()
    total_loss, PDE_loss, boundary_loss = train(NN, domain, training_params)
    t1 = t() - t0 
    
    MSE, L_inf = error_cube(NN, side_length, n_grid)
    
    MSE_history[i] = MSE
    L_inf_error_history[i] = L_inf
    run_time_history[i] = t1
    
    X_axis = 0
    Y_axis = 1

    plot_2d_proj(X_axis, Y_axis, NN, n_grid, side_length)

    
print('Mean square error:', MSE)
print('L-infinity error:', L_inf)
print('Run time:', run_time_history.sum())

plot_2d_proj_w(X_axis, Y_axis, NN, n_grid, side_length, training_params)

#%%
import numpy as np

from mpl_toolkits.axes_grid1 import host_subplot


x = np.arange(rounds) + 1
MSE_log10 =torch.log10(MSE_history)

ax = host_subplot(111)
ax.plot(x, MSE_log10)
ax.set_xticks(x)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('$\log_{10} (MSE)$', fontsize = 'xx-large')
plt.show()


L_inf_log10 = torch.log10(L_inf_error_history)
ax = host_subplot(111)
ax.plot(x, L_inf_log10)
ax.set_xticks(x)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('$\log_{10} (E_{\infty})$', fontsize = 'xx-large')
plt.show()



