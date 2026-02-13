#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Variable
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lossIC(net, x):
    mse_cost_function = torch.nn.MSELoss()
    
    #Compute estimated initial condition
    zero = torch.zeros_like(x).to(device)
    
    u = net(x, zero) 
    
    #Compute actual initial condition
    u_0 = InitialCondition_u(net, x)
    
    u_IC_loss = mse_cost_function(u, u_0)
    u_IC_loss_scaler = mse_cost_function(u_0, zero)
     
    return u_IC_loss/u_IC_loss_scaler 

def InitialCondition_u(net, x): 
    u_0 = x**2 * torch.sin(2 *np.pi * x)
    return u_0

def lossBdry(net, t):
    mse_cost_function = torch.nn.MSELoss()
    zero = torch.zeros_like(t).to(device)
    
    x_l_Bdry = Variable(net.x1_l * torch.ones_like(t), requires_grad=True).to(device)
    x_u_Bdry = Variable(net.x1_u * torch.ones_like(t), requires_grad=True).to(device)
    
    
    # Define the variables on 4 boundaries at outer squre
    
    u_left = net(x_l_Bdry, t)
    u_right = net(x_u_Bdry, t)
    
    u_x_left = torch.autograd.grad(u_left.sum(), x_l_Bdry, create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right.sum(), x_u_Bdry, create_graph=True)[0]
    
    
    loss_u = mse_cost_function(u_left, u_right)
    loss_u_x = mse_cost_function(u_x_left, u_x_right)
    
    return loss_u, loss_u_x

def lossNSpde(net, x, t):
    mse_cost_function = torch.nn.MSELoss()
    
    u = net(x, t)
    zero = torch.zeros_like(x).to(device)
    
    #Compute Derivatives
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    #AC
    AC_Residual = u_t - 0.0001 * u_xx + 4*u**3 - 4*u
    loss = mse_cost_function(AC_Residual, zero)
    
    return loss

def lossNSpde_rank(net, x, t,batch_size):
    mse_cost_function = torch.nn.MSELoss()
    
    u = net(x, t)
    zero = torch.zeros_like(x).to(device)
    
    #Compute Derivatives
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    #Loss functions w.r.t. governing Navier-Stokes equation on inner space
    #AC
    AC_Residual = u_t - 0.0001 * u_xx + 4*u**3 - 4*u
    #loss = mse_cost_function(AC_Residual, zero)
    loss = torch.abs(AC_Residual)
    sorted_tensor, indices = torch.sort(torch.reshape(loss, (-1,)), descending=True) #PDEloss_tensor.view(-1)
    sorted_tensor = torch.reshape(sorted_tensor, (-1,1) )#sorted_tensor.view(-1, 1)
            
    max_stad = sorted_tensor[int(batch_size/10)-1,0]
    PDEloss_picked = torch.where(loss>=max_stad, loss, 0)
    loss = mse_cost_function(PDEloss_picked, zero)*10 #adjust denominator from len(x[:,0]) to len(x[:,0])/10
    return loss 

def total_free_e_derivative(net, x, t, min_raw_loss):
    mse_cost_function = torch.nn.MSELoss()
    zero = torch.zeros_like(x)
    if t == None:
        u = InitialCondition_u(net, x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        #u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    else:
        u = net(x, t)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        #u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    e_derivative = 4*(u**2-1)**2 + 0.0001*mse_cost_function(u_x,zero) #(u**2-1)**2 +.5* 0.0001*u_x**2
    return e_derivative

