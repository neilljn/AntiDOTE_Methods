# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# jax version
def U_bounds_from_X_jax (X:np.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,theta:np.array,gamma:float,T:int,N:int,h:np.array):

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # T - length of time to run the simulation for
    # N - number of individuals
    # h - household matrix
    
    # probability of recovery at all points in time
    prob_rec = jnp.tile(1-jnp.exp(-1*gamma),(T,N))

    # probability of infection at all points in time
    #seasonal = 1 - jnp.cos(2*jnp.pi*t/52)
    globe = (jnp.tile(np.sum(X,axis=1),(N,1)) / N).T
    household = theta[1] * jnp.matmul(X,h)
    covariate = theta[2]*age + theta[3]*sex
    transmission_matrix = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G+household*seasonal_matrix_H))
    prob_col = 1-jnp.exp(-1*transmission_matrix[0:T])

    # creating a matrix comparing the value of the latent variable at time t vs time t-1
    latent_diff = X[1:(T+1)] - X[0:T]

    # using this to find binary matrix for each of the four possible events
    not_infect = (latent_diff==0)*(1-X[1:(T+1)])
    infect = (latent_diff==1)
    not_recover = (latent_diff==0)*X[1:(T+1)]
    recover = (latent_diff==-1)

    # upper and lower bounds at all points in time
    lower = not_infect*prob_col + not_recover*prob_rec
    upper = not_infect + infect*prob_col + not_recover + recover*prob_rec

    # returning the upper and lower bounds
    return {'lower':lower,'upper':upper}

# creating the just in time version of the function
U_bounds_from_X_jit = jit(U_bounds_from_X_jax, static_argnames=["T","N"])