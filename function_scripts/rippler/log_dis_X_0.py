# importing numpy
import numpy as np
from numpy import random

# log conditional distribution of beta_0 (initial transmission rate)
def log_dis_X_0 (X:np.array,prior_X_0:float,N:int):

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # prior_X_0 - prior probability of each individual being colonised at time 0
    # c_0 - hyperparameter from the prior for beta_0 (rate of exponential distribution)
    # N - total number of individuals

    # the number of individuals who are initially colonised
    no_col_0 = np.sum(X[0])

    # log condtional distribution
    output = no_col_0*np.log(prior_X_0) + (N-no_col_0)*np.log(1-prior_X_0)

    # returning output
    return output