# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# creating a function for the log conditional distribution of the beta parameters given the data, the latent variables, and other parameters
def log_dis_theta_maker(X:np.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,T:int,N:int,h:np.array,mu:np.array) :

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # T - length of time to run the simulation for
    # N - number of individuals
    # h - household matrix
    # mu - vector of hyperparameters for the priors of beta_G, beta_H, delta_A, delta_S
    
    # creating the function
    def log_dis_theta_inside(theta:np.array):

        # list of inputs:
        # theta - beta_G, beta_H, delta_A, delta_S

        # setting the output to -inf if either beta value is less than or equal to zero, 0 otherwise
        output = jnp.nan_to_num((theta[0]<=0)*(-1)*jnp.inf,nan=0,neginf=jnp.inf*(-1)) + jnp.nan_to_num((theta[1]<=0)*(-1)*jnp.inf,nan=0,neginf=jnp.inf*(-1))

        # creating the transmission matrix
        globe = (jnp.tile(jnp.sum(X,axis=1),(N,1)) / N).T
        household = theta[1] * jnp.matmul(X,h)
        covariate = theta[2]*age + theta[3]*sex
        transmission_matrix = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G+household*seasonal_matrix_H))

        # creating a matrix comparing the value of the latent variable at time t vs time t-1
        latent_diff = X[1:(T+1)] - X[0:T]

        # using this to find binary matrix for each of the four possible events
        not_infect = (latent_diff==0)*(1-X[1:(T+1)])
        infect = (latent_diff==1)

        # multiplying in the log likelihood of each event occuring (including returning -inf if there are any infections when the transmission rate is 0)
        not_infect = not_infect * (-1*transmission_matrix[0:T])
        infect = jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_matrix[0:T])), nan=0)

        # adding the colonisation part to the output
        output += jnp.sum(not_infect) + jnp.sum(infect)

        # adding the prior part to the output
        output += ((-1)*mu[0]*theta[0]) + ((-1)*mu[1]*theta[1]) + ((-1)*mu[2]*jnp.abs(theta[2])) + ((-1)*mu[3]*jnp.abs(theta[3]))

        # returning overall log likelihood, including from the priors (and returning -inf if either beta is less than or equal to 0)
        return output
    
    # returning the function
    return log_dis_theta_inside