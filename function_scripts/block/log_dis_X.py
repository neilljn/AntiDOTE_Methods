# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# function to work out the number of TP/FP/TN/FNs
def test_acc(X:np.array,test_results:np.array) :
    
    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # test_results - matrix of all individuals colonisation status over time

    # creating a matrix comparing the results with the current latent variable values (0 for TPs or TNs, 1 for FPs, -1 for FNs, and nan for no tests)
    test_diff_matrix = test_results - X

    # calculating the number of TP/FP/TN/FNs from this matrix
    TP = np.sum((test_diff_matrix==0)*X)
    FN = np.sum(test_diff_matrix==-1)
    FP = np.sum(test_diff_matrix==1)
    TN = np.sum((test_diff_matrix==0)*(1-X))

    # returning outputs
    output = {'TP':TP, 'FN':FN, 'FP':FP, 'TN':TN}
    return output

# creating a function for the log conditional distribution of X given the data and parameters
def log_dis_X_jax(X:np.array,test_results:np.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,theta:np.array,gamma:float,sens:float,spec:float,T:int,N:int,h:np.array) :

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # test_results - matrix of test results
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # sens - test true positive rate
    # spec - test true negative rate
    # T - length of time to run the simulation for
    # N - number of individuals
    # h - household matrix

    # finding the number of TP/FP/TN/FNs
    acc = test_acc(X,test_results)

    # total log likelihood from the data given the latent variables and parameters (including returning -inf if there are any false positives)
    output = (acc['TP']*jnp.log(sens)) + (acc['FN']*jnp.log(1-sens)) + (acc['TN']*jnp.log(spec)) + (acc['FP']*jnp.log(1-spec))

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
    not_recover = (latent_diff==0)*X[1:(T+1)]
    recover = (latent_diff==-1)

    # multiplying in the log likelihood of each event occuring (including returning -inf if there are any infections when the transmission rate is 0)
    not_infect = not_infect * (-1*transmission_matrix[0:T])
    infect = jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_matrix[0:T])), nan=0)
    not_recover = not_recover * (-1*gamma)
    recover = recover * jnp.log(1-jnp.exp(-1*gamma))

    # adding the colonisation part to the output
    output += jnp.sum(not_infect) + jnp.sum(infect) + jnp.sum(not_recover) + jnp.sum(recover)

    # returning overall log likelihood
    return output

# just in time version of the log conditional of the betas
log_dis_X_jit = jit(log_dis_X_jax, static_argnames=["T","N"])