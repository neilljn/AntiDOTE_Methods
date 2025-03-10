# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# creating a function for the gradiant of the potential at a given theta
def grad_log_dis_theta_maker(X:np.array,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,T:int,N:int,h:np.array,mu:np.array) :

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
    def grad_log_dis_theta_inside(theta:np.array):

        # list of inputs:
        # theta - beta_G, beta_H, delta_A, delta_S

        # creating the transmission matrix
        globe = (jnp.tile(jnp.sum(X,axis=1),(N,1)) / N).T
        household = theta[1] * jnp.matmul(X,h)
        covariate = theta[2]*age + theta[3]*sex
        transmission_matrix = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G+household*seasonal_matrix_H))
        transmission_matrix_2 = jnp.exp(covariate)*theta[0]*jnp.matmul(X,h)

        # differentiating lambda
        # d_lambda_d_theta = jnp.zeros((4,T,N))
        # d_lambda_d_theta[0] = transmission_matrix[0:T]/theta[0]
        # d_lambda_d_theta[1] = transmission_matrix_2[0:T]
        # d_lambda_d_theta[2] = jnp.tile(age,(T,1))*transmission_matrix[0:T]
        # d_lambda_d_theta[3] = jnp.tile(sex,(T,1))*transmission_matrix[0:T]
        d_lambda_d_theta = jnp.array([transmission_matrix[0:T]/theta[0],transmission_matrix_2[0:T],jnp.tile(age,(T,1))*transmission_matrix[0:T],jnp.tile(sex,(T,1))*transmission_matrix[0:T]])

        # creating a matrix comparing the value of the latent variable at time t vs time t-1
        latent_diff = X[1:(T+1)] - X[0:T]

        # using this to find binary matrix for each of the four possible events
        not_infect = (latent_diff==0)*(1-X[1:(T+1)])
        infect = (latent_diff==1)

        # stacking the binary matrices and transmission matrix (one per parameter)
        not_infect_stack = jnp.tile(not_infect,(4,1,1))
        infect_stack = jnp.tile(infect,(4,1,1))
        transmission_stack = jnp.tile(transmission_matrix[0:T],(4,1,1))

        # multiplying in the gradient of each event
        not_infect_stack = not_infect_stack * d_lambda_d_theta
        infect_stack = jnp.nan_to_num(infect_stack * d_lambda_d_theta / (jnp.exp(transmission_stack) - 1), nan=0)

        # summing over all events
        grad = jnp.sum(infect_stack,axis=(1,2)) - jnp.sum(not_infect_stack,axis=(1,2))

        # adding the prior part to the output
        grad = grad - jnp.array([mu[0],mu[1],mu[2]*(theta[2]>0)-mu[2]*(theta[2]<0),mu[3]*(theta[3]>0)-mu[3]*(theta[3]<0)])

        # adding -inf to the output if either beta is less than 0
        grad = grad + jnp.array([jnp.nan_to_num((theta[0]<=0)*(-1)*jnp.inf,nan=0,neginf=jnp.inf*(-1)),jnp.nan_to_num((theta[1]<=0)*(-1)*jnp.inf,nan=0,neginf=jnp.inf*(-1)),0,0])

        # returning overall log likelihood, including from the priors (and returning -inf if either beta is less than or equal to 0)
        return grad

    # returning the function
    return grad_log_dis_theta_inside