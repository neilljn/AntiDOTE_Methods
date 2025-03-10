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

# jax version of the log of the conditional distribution of the latent variables (u)
def log_dis_U_jax (X:np.array, test_results:np.array, sens:float, spec:float) :

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # test_results - matrix of all individuals colonisation status over time
    # sens - true positive rate
    # spec - true negative rate

    # finding the number of TP/FP/TN/FNs
    acc = test_acc(X,test_results)

    # total log likelihood from the data given the latent variables and parameters (including returning -inf if there are any false positives)
    #output = (acc['TP']*jnp.log(sens)) + (acc['FN']*jnp.log(1-sens)) + jnp.nan_to_num((acc['FP']>0)*-1*jnp.inf,nan=0,neginf=jnp.inf*(-1))
    output = (acc['TP']*jnp.log(sens)) + (acc['FN']*jnp.log(1-sens)) + (acc['TN']*jnp.log(spec)) + (acc['FP']*jnp.log(1-spec))

    # returning the output
    return output

# creating the just in time version
log_dis_U_jit = jit(log_dis_U_jax)