# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# function to create inside
def inside_maker (N:int, T:int, theta:np.array, gamma:float, seasonal_matrix_G:np.array, seasonal_matrix_H:np.array, age:jnp.array, sex:jnp.array, h:jnp.array):

    # list of inputs:
    # N - number of individuals
    # T - total time
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # h - household matrix

    # calculating the probability of recovery for each individual
    prob_rec = jnp.repeat(1-jnp.exp(-1*gamma),N)
    
    # function for inside the for loop in the jax version
    def inside (t:int, X:jnp.array) :

        # list of inputs:
        # t - the current time to create a new C row for
        # X - t rows of the C matrix, and the rest is the u matrix

        # calculating the transmission rate for each individual
        #seasonal = 1 - jnp.cos(2*jnp.pi*(t+16)/52)
        globe = sum(X[t-1]) / N
        household = theta[1] * jnp.matmul(h,X[t-1])
        covariate = theta[2]*age + theta[3]*sex
        transmission = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[t-1]+household*seasonal_matrix_H[t-1]))

        # calculating the probilitiy of colonisation for each individual
        prob_col = 1-jnp.exp(-1*transmission)
            
        # using the u values to determine if colonisation or recovery events occur 
        #C[t] = (1-C[t-1])*(u[t-1]<prob_col) + (C[t-1])*(u[t-1]>=prob_rec)
        X_new_row = (1-X[t-1])*(X[t]<prob_col) + (X[t-1])*(X[t]>=prob_rec)
        X = X.at[t].set(X_new_row)

        # returning the current value of C
        return X
    
    # returnin the interior function
    return inside

# jax version of the function to create the C matrix (T+1 x n) from u and C_0
def X_from_U_jax(U:jnp.array,X_0:jnp.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:jnp.array,sex:jnp.array,theta:np.array,gamma:float,N:int,T:int,h:jnp.array) :

    # list of inputs:
    # u - matrix of probabilities
    # X_0 - matrix of intital conditions
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # N - number of individuals
    # T - total time
    # h - household matrix

    # creating the inside function
    inside_loop = inside_maker(N,T,theta,gamma,seasonal_matrix_G,seasonal_matrix_H,age,sex,h)

    # creating the C matrix to be the right size, with an added bonus/dummy row
    X = jnp.vstack((X_0,U))

    # size of the intial conditions
    t_0 = X_0.shape[0]
    t = t_0

    # considering each point in time
    X = jax.lax.fori_loop(t_0,T+1,inside_loop,X)

    # returning the final value of C
    return X

# creating the just in time version
X_from_U_jit = jit(X_from_U_jax, static_argnames=["T","N"])