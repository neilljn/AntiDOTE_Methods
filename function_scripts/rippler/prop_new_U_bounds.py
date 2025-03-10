# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# jax version
def prop_new_U_bounds_jax (X:np.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,theta:np.array,gamma:float,T:int,N:int,h:np.array,t:int,j:int):

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
    # t - timepoint of interest
    # j - individual of interest
    
    # probability of recovery at any point in time
    prob_rec = 1-jnp.exp(-1*gamma)

    # probability of colonisation at time t+1
    #seasonal = 1 - jnp.cos(2*jnp.pi*(t+17)/52)
    globe = sum(X[t]) / N
    household = theta[1] * jnp.matmul(h,X[t])
    covariate = theta[2]*age + theta[3]*sex
    transmission = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[t]+household*seasonal_matrix_H[t]))
    prob_col = 1-jnp.exp(-1*transmission)

    # when we change the state X[t+1,j], what is the difference compared to the previous state
    latent_diff = (1-X[t+1,j]) - X[t,j]

    # using this to determine which of the four possible events occured
    not_infect = (latent_diff==0)*(1-X[t,j])
    infect = (latent_diff==1)
    not_recover = (latent_diff==0)*X[t,j]
    recover = (latent_diff==-1)

    # upper and lower bounds to produce ta new u
    lower = not_infect*prob_col[j] + not_recover*prob_rec
    upper = not_infect + infect*prob_col[j] + not_recover + recover*prob_rec

    # calculating the proposal ratio
    #prob_jump = 1/(upper-lower)
    #prob_jump_reverse = 1/(1-(upper-lower))
    #q_move = prob_jump_reverse/prob_jump
    log_q_move = jnp.log(upper-lower) - jnp.log(1+lower-upper)

    # returning the bounds of the new u matrix
    return {'lower':lower, 'upper':upper, 'log_q_move':log_q_move}

# making the just in time version
prop_new_U_bounds_jit = jit(prop_new_U_bounds_jax, static_argnames=["T","N"])