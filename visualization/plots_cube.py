#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:15:43 2023

@author: carlosesteveyague
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np


def plot_2d_proj(axis1, axis2, NN, n_grid, side_length, dim = None):
    
    if dim == None:
        dim = NN.L1.in_features
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)

    Grid = torch.zeros([n_grid, n_grid, dim])
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    #plt.pcolormesh(GridX.detach(), GridY.detach(), V.detach())
    #plt.title('Solution')
    #plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(np.array(GridX), np.array(GridY), np.array(V.detach()), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.axes.set_zlim3d(bottom=0, top=side_length/2)
    plt.show()


def plot_level_set_cube(axis1, axis2, NN, n_grid, side_length, levels = [0], dim = None):
    
    if dim == None:
        dim = NN.L1.in_features
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)

    Grid = torch.zeros([n_grid, n_grid, dim])
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    fig, ax = plt.subplots()   
    CS = ax.contour(np.array(GridX), np.array(GridY), np.array(V.detach()), levels)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.show()


def plot_2d_proj_w(axis1, axis2, NN, n_grid, side_length, training_params, dim = None):
    
    
    H = training_params['numerical_scheme']
    alpha = training_params['alpha']
    beta = training_params['beta']
    d = training_params['delta']
    f = training_params['f']
    c = training_params['c']
    
    if dim == None:
        dim = NN.L1.in_features
    
    Xgrid = torch.linspace(-side_length/2+d, side_length/2-d, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)

    Grid = torch.zeros([n_grid, n_grid, dim])
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY
    
    W = H(NN, Grid, d, alpha, beta, c) - f(Grid)

    plt.pcolormesh(GridX.detach(), GridY.detach(), W.detach())
    plt.title('PDE residual')
    plt.show()

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.plot_surface(np.array(GridX), np.array(GridY), np.array(W.detach()), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #ax.axes.set_zlim3d(bottom=0, top=side_length/2)
    #plt.show()
    

def plot_slice(NN, X_axis, Y_axis, side_length, n_frames, n_grid, dim = None):
    
    if dim == None:
        dim = NN.L1.in_features
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)
    
    for l in torch.linspace(-side_length/2, side_length/2, n_frames):
        
        Grid = torch.zeros([n_grid, n_grid, dim]) + l
        
        Grid[:,:,X_axis] = GridX
        Grid[:,:,Y_axis] = GridY
        
        
        V = NN(Grid).squeeze()
        
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(GridX.detach(), GridY.detach(), V.detach())
        ax.axes.set_zlim3d(bottom=0, top=side_length/2)
        plt.show()


def plot_traj_cube(traj, axis1, axis2, NN, n_grid, side_length):
    
    dim = traj[0].shape[-1]
    
    Xgrid = torch.linspace(-side_length/2, side_length/2, n_grid)

    GridX, GridY = torch.meshgrid(Xgrid,Xgrid)

    Grid = torch.zeros([n_grid, n_grid, dim])
    
    Grid[:,:,axis1] = GridX
    Grid[:,:,axis2] = GridY

    V = NN(Grid).squeeze()

    plt.pcolormesh(GridX.detach(), GridY.detach(), V.detach())
    
    
    for traj_i in traj:
        plt.plot(traj_i[:,0], traj_i[:,1], c = 'red')
        plt.scatter(traj_i[0,0], traj_i[0,1], c = 'red', marker = '*', s = 100)
    plt.show()


def plot_traj_game(traj, side_length):
    
    for traj_i in traj:
        plt.plot(traj_i[:,0], traj_i[:,1])
        plt.scatter(traj_i[0,0], traj_i[0,1], marker = '*', s = 100)
        plt.xlim((-side_length/2, side_length/2))
        plt.ylim((-side_length/2, side_length/2))
    plt.show()


def plot_traj_game_movie(traj, side_length):
    
    traj_p1 = traj[0][0:-1:5]
    traj_p2 = traj[1][0:-1:5]
    
    for i in range(traj_p1.shape[0]):
        i_low = max(0, i-20)
        plt.plot(traj_p1[i_low:i+1, 0], traj_p1[i_low:i+1, 1])
        plt.scatter(traj_p1[i,0], traj_p1[i,1], marker = '*', s = 100)
        
        plt.plot(traj_p2[i_low:i+1, 0], traj_p2[i_low:i+1, 1])
        plt.scatter(traj_p2[i,0], traj_p2[i,1], marker = '*', s = 100)
        
        plt.xlim((-side_length/2, side_length/2))
        plt.ylim((-side_length/2, side_length/2))
        
        plt.show()
        
    

def plot_traj_car_circ(traj, axis1, axis2, NN, n_grid, side_length, rho):
    
    theta0 = traj[0][0, -1]
    X0 = traj[0][0, :2]
    center = X0 + rho*torch.tensor([-torch.sin(theta0), torch.cos(theta0)])
    if center.norm() > X0.norm():
        center = X0 - rho*torch.tensor([-torch.sin(theta0), torch.cos(theta0)])
    
    theta_grid = torch.linspace(0, 2*torch.pi, n_grid)
    
    x_circ = center[0] + rho*torch.cos(theta_grid) 
    y_circ = center[1] + rho*torch.sin(theta_grid) 
    
    plt.plot(x_circ, y_circ)
    plt.scatter(torch.zeros([1,1]), torch.zeros([1,1]), c = 'blue', marker = 'o', s = 100)
    
    for traj_i in traj:
        
        """
        ##
        
        for j in range(2, traj_i.shape[0]):
            
            v1 = traj_i[j, :2] - traj_i[j-1, :2]
            v2 = traj_i[j-1, :2] - traj_i[j-2, :2]
            
            if torch.dot(v1, v2) < 0:
                
                theta0 = traj[0][j-1, -1]
                X0 = traj[0][j-1, :2]
                center = X0 + rho*torch.tensor([-torch.sin(theta0), torch.cos(theta0)])
                if center.norm() > X0.norm():
                    center = X0 - rho*torch.tensor([-torch.sin(theta0), torch.cos(theta0)])
                
                theta_grid = torch.linspace(0, 2*torch.pi, n_grid)
    
                x_circ = center[0] + rho*torch.cos(theta_grid) 
                y_circ = center[1] + rho*torch.sin(theta_grid) 
                
                plt.plot(x_circ, y_circ, c = 'green')
        
        ##
        """
        
        plt.plot(traj_i[:,0], traj_i[:,1], c = 'red')
        plt.scatter(traj_i[0,0], traj_i[0,1], c = 'red', marker = '*', s = 100)
    
    plt.axis('equal')
    plt.xlim((-side_length/2, side_length/2))
    plt.ylim((-side_length/2, side_length/2))
    
    
    plt.show()