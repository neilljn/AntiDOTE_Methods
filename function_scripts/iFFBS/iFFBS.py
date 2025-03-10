# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# a function to run the iFFBS
def iFFBS(X:np.array, test_results:np.array, j:int, seasonal_matrix_G:np.array, seasonal_matrix_H:np.array, age:np.array, sex:np.array, theta:np.array, gamma:float, T:int, N:int, h:np.array, prior_X_0:float, sens:float, spec:float) :

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # test_results - matrix of test results
    # j - individual to sample for
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # T - length of time to run the simulation for
    # N - number of individuals
    # h - household matrix
    # prior_X_0 - prior probability of colonisation at time 0
    # sens - test sensitivity
    # spec - test specificity

    # creating a matrix for forward probabilities
    forward_prob = np.zeros((T+1,2))

    # calculating the infectious pressure when X[0,j]=1 and when X[0,j]=0
    Xt_0 = X[0].copy()
    Xt_0[j] = 0
    Xt_1 = X[0].copy()
    Xt_1[j] = 1
    globe_0 = np.sum(Xt_0) / N
    globe_1 = np.sum(Xt_1) / N
    household_0 = theta[1] * np.matmul(h,Xt_0)
    household_1 = theta[1] * np.matmul(h,Xt_1)
    covariate = theta[2]*age + theta[3]*sex
    transmission_0 = np.array(np.exp(covariate)*theta[0]*(globe_0*seasonal_matrix_G[0]+household_0*seasonal_matrix_H[0]))
    transmission_1 = np.array(np.exp(covariate)*theta[0]*(globe_1*seasonal_matrix_G[0]+household_1*seasonal_matrix_H[0]))

    # events for individuals other than j at time 1
    latent_diff = X[1] - X[0]
    latent_diff[j] = 2
    not_infect = (latent_diff==0)*(1-X[1])
    infect = (latent_diff==1)
    not_recover = (latent_diff==0)*X[1]
    recover = (latent_diff==-1)

    # log probabilities of colonisation events when X[0,j]=1
    log_prob_not_infect_1 = np.sum(not_infect * (-1*transmission_1))
    log_prob_infect_1 = np.sum(np.nan_to_num(infect * np.log(1-np.exp(-1*transmission_1)), nan=0))

    # log probabilities of colonisation events when X[0,j]=0
    log_prob_not_infect_0 = np.sum(not_infect * (-1*transmission_0))
    log_prob_infect_0 = np.sum(np.nan_to_num(infect * np.log(1-np.exp(-1*transmission_0)), nan=0))

    # log probabilities of recovery events
    log_prob_not_recover = np.sum(not_recover * (-1*gamma))
    log_prob_recover = np.sum(recover * np.log(1-np.exp(-1*gamma)))

    # unnormalsied log probabilites
    log_prob_1 = np.log(prior_X_0) + log_prob_not_infect_1 + log_prob_infect_1 + log_prob_not_recover + log_prob_recover
    log_prob_0 = np.log(1-prior_X_0) + log_prob_not_infect_0 + log_prob_infect_0 + log_prob_not_recover + log_prob_recover

    # normalised probabilites
    p_0 = np.exp(log_prob_0-log_prob_1)/(1+np.exp(log_prob_0-log_prob_1))
    forward_prob[0,0] = p_0
    forward_prob[0,1] = 1-p_0

    # probabilites for the forward steps that won't change
    prob_CU = 1 - np.exp(-1*gamma)
    prob_CC = np.exp(-1*gamma)
    prob_UC = np.zeros(T)
    prob_UU = np.zeros(T)

    # calculating the forward probabilities for t=1 to T-1
    for t in range(1,T):

        # calculating the infectious pressure when X[t-1,j]=0
        Xt_minus_0 = X[t-1].copy()
        Xt_minus_0[j] = 0
        globe = np.sum(Xt_minus_0) / N
        household = theta[1] * np.matmul(h,Xt_minus_0)
        #covariate = theta[2]*age + theta[3]*sex
        transmission = np.array(np.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[t-1]+household*seasonal_matrix_H[t-1]))

        # calculating probabilites based on the previous step
        prob_UC[t-1] = 1 - np.exp(-1*transmission[j])
        prob_UU[t-1] = np.exp(-1*transmission[j])
        log_prob_prev_0 = np.log(prob_UU[t-1]*forward_prob[t-1,0] + prob_CU*forward_prob[t-1,1])
        log_prob_prev_1 = np.log(prob_UC[t-1]*forward_prob[t-1,0] + prob_CC*forward_prob[t-1,1])

        # calculating the test result bit
        log_prob_TN_0 = np.log(spec) * (test_results[t,j]==0)
        log_prob_FP_0 = np.log(1-spec) * (test_results[t,j]==1)
        log_prob_TP_1 = np.log(sens) * (test_results[t,j]==1)
        log_prob_FN_1 = np.log(1-sens) * (test_results[t,j]==0)
        log_prob_test_0 = log_prob_TN_0 + log_prob_FP_0
        log_prob_test_1 = log_prob_TP_1 + log_prob_FN_1

        # calculating the infectious pressure when X[t,j]=1 and when X[t,j]=0
        Xt_0 = X[t].copy()
        Xt_0[j] = 0
        Xt_1 = X[t].copy()
        Xt_1[j] = 1
        globe_0 = np.sum(Xt_0) / N
        globe_1 = np.sum(Xt_1) / N
        household_0 = theta[1] * np.matmul(h,Xt_0)
        household_1 = theta[1] * np.matmul(h,Xt_1)
        #covariate = theta[2]*age + theta[3]*sex
        transmission_0 = np.array(np.exp(covariate)*theta[0]*(globe_0*seasonal_matrix_G[t]+household_0*seasonal_matrix_H[t]))
        transmission_1 = np.array(np.exp(covariate)*theta[0]*(globe_1*seasonal_matrix_G[t]+household_1*seasonal_matrix_H[t]))

        # events for individuals other than j at time 1
        latent_diff = X[t+1] - X[t]
        latent_diff[j] = 2
        not_infect = (latent_diff==0)*(1-X[t+1])
        infect = (latent_diff==1)
        not_recover = (latent_diff==0)*X[t+1]
        recover = (latent_diff==-1)

        # log probabilities of colonisation events when X[0,j]=1
        log_prob_not_infect_1 = np.sum(not_infect * (-1*transmission_1))
        log_prob_infect_1 = np.sum(np.nan_to_num(infect * np.log(1-np.exp(-1*transmission_1)), nan=0))

        # log probabilities of colonisation events when X[0,j]=0
        log_prob_not_infect_0 = np.sum(not_infect * (-1*transmission_0))
        log_prob_infect_0 = np.sum(np.nan_to_num(infect * np.log(1-np.exp(-1*transmission_0)), nan=0))

        # log probabilities of recovery events
        log_prob_not_recover = np.sum(not_recover * (-1*gamma))
        log_prob_recover = np.sum(recover * np.log(1-np.exp(-1*gamma)))

        # unnormalised log probabilities
        log_prob_1 = log_prob_prev_1 + log_prob_test_1 + log_prob_not_infect_1 + log_prob_infect_1 + log_prob_not_recover + log_prob_recover
        log_prob_0 = log_prob_prev_0 + log_prob_test_0 + log_prob_not_infect_0 + log_prob_infect_0 + log_prob_not_recover + log_prob_recover

        # normalised probabilities
        prob_0 = np.exp(log_prob_0-log_prob_1)/(1+np.exp(log_prob_0-log_prob_1))
        forward_prob[t,0] = prob_0
        forward_prob[t,1] = 1-prob_0

    # calculating the infectious pressure when X[T-1,j]=0
    Xt_minus_0 = X[T-1].copy()
    Xt_minus_0[j] = 0
    globe = np.sum(Xt_minus_0) / N
    household = theta[1] * np.matmul(h,Xt_minus_0)
    #covariate = theta[2]*age + theta[3]*sex
    transmission = np.array(np.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[T-1]+household*seasonal_matrix_H[T-1]))

    # calculating probabilites based on the previous step
    prob_UC[T-1] = 1 - np.exp(-1*transmission[j])
    prob_UU[T-1] = np.exp(-1*transmission[j])
    log_prob_prev_0 = np.log(prob_UU[T-1]*forward_prob[T-1,0] + prob_CU*forward_prob[T-1,1])
    log_prob_prev_1 = np.log(prob_UC[T-1]*forward_prob[T-1,0] + prob_CC*forward_prob[T-1,1])

    # calculating the test result bit
    log_prob_TN_0 = np.log(spec) * (test_results[T,j]==0)
    log_prob_FP_0 = np.log(1-spec) * (test_results[T,j]==1)
    log_prob_TP_1 = np.log(sens) * (test_results[T,j]==1)
    log_prob_FN_1 = np.log(1-sens) * (test_results[T,j]==0)
    log_prob_test_0 = log_prob_TN_0 + log_prob_FP_0
    log_prob_test_1 = log_prob_TP_1 + log_prob_FN_1

    # unnormalised log probabilities
    log_prob_1 = log_prob_prev_1 + log_prob_test_1
    log_prob_0 = log_prob_prev_0 + log_prob_test_0

    # normalised probabilities
    prob_0 = np.exp(log_prob_0-log_prob_1)/(1+np.exp(log_prob_0-log_prob_1))
    forward_prob[T,0] = prob_0
    forward_prob[T,1] = 1-prob_0

    # sampling for X[T,j]
    X_j_new = np.zeros(T+1)
    X_j_new[T] = random.binomial(1,forward_prob[T,1])

    # sampling backwards for t=T-1 to 0
    for t in reversed(range(T)):
        
        # calculating the unnormalised probabilities
        prob_0_ast = (X_j_new[t+1]*prob_UC[t] + (1-X_j_new[t+1])*prob_UU[t]) * forward_prob[t-1,0]
        prob_1_ast = (X_j_new[t+1]*prob_CC + (1-X_j_new[t+1])*prob_CU) * forward_prob[t-1,1]

        # normalising the probabilities
        prob_1 = prob_1_ast/(prob_0_ast+prob_1_ast)

        # sampling for X[t,j]
        X_j_new[t] = random.binomial(1,prob_1)

    # creating a new X matrix with column j changed
    X_new = X.copy()
    X_new[:,j] = X_j_new

    # returning the new X matrix
    return X_new

# jax version
def iFFBS_jax(X:np.array, test_results:np.array, j:int, seasonal_matrix_G:np.array, seasonal_matrix_H:np.array, age:np.array, sex:np.array, theta:np.array, gamma:float, T:int, N:int, h:np.array, prior_X_0:float, sens:float, spec:float, u:np.array) :

    # list of inputs:
    # X - matrix of all individuals colonisation status over time
    # test_results - matrix of test results
    # j - individual to sample for
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # T - length of time to run the simulation for
    # N - number of individuals
    # h - household matrix
    # prior_X_0 - prior probability of colonisation at time 0
    # sens - test sensitivity
    # spec - test specificity
    # u - vector of random numbers between 0 and 1, length T+1

    # creating a matrix for forward probabilities
    forward_prob = jnp.zeros((T+1,2))

    # calculating the infectious pressure when X[0,j]=1 and when X[0,j]=0
    Xt_0 = X[0].copy()
    #Xt_0[j] = 0
    Xt_0 = Xt_0.at[j].set(0)
    Xt_1 = X[0].copy()
    #Xt_1[j] = 1
    Xt_1 = Xt_1.at[j].set(1)
    globe_0 = jnp.sum(Xt_0) / N
    globe_1 = jnp.sum(Xt_1) / N
    household_0 = theta[1] * jnp.matmul(h,Xt_0)
    household_1 = theta[1] * jnp.matmul(h,Xt_1)
    covariate = theta[2]*age + theta[3]*sex
    transmission_0 = jnp.array(jnp.exp(covariate)*theta[0]*(globe_0*seasonal_matrix_G[0]+household_0*seasonal_matrix_H[0]))
    transmission_1 = jnp.array(jnp.exp(covariate)*theta[0]*(globe_1*seasonal_matrix_G[0]+household_1*seasonal_matrix_H[0]))

    # events for individuals other than j at time 1
    latent_diff = X[1] - X[0]
    #latent_diff[j] = 2
    latent_diff = latent_diff.at[j].set(2)
    not_infect = (latent_diff==0)*(1-X[1])
    infect = (latent_diff==1)
    not_recover = (latent_diff==0)*X[1]
    recover = (latent_diff==-1)

    # log probabilities of colonisation events when X[0,j]=1
    log_prob_not_infect_1 = jnp.sum(not_infect * (-1*transmission_1))
    log_prob_infect_1 = jnp.sum(jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_1)), nan=0))

    # log probabilities of colonisation events when X[0,j]=0
    log_prob_not_infect_0 = jnp.sum(not_infect * (-1*transmission_0))
    log_prob_infect_0 = jnp.sum(jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_0)), nan=0))

    # log probabilities of recovery events
    log_prob_not_recover = jnp.sum(not_recover * (-1*gamma))
    log_prob_recover = jnp.sum(recover * jnp.log(1-jnp.exp(-1*gamma)))

    # unnormalsied log probabilites
    log_prob_1 = jnp.log(prior_X_0) + log_prob_not_infect_1 + log_prob_infect_1 + log_prob_not_recover + log_prob_recover
    log_prob_0 = jnp.log(1-prior_X_0) + log_prob_not_infect_0 + log_prob_infect_0 + log_prob_not_recover + log_prob_recover

    # normalised probabilites
    p_0 = jnp.exp(log_prob_0-log_prob_1)/(1+jnp.exp(log_prob_0-log_prob_1))
    #forward_prob[0,0] = p_0
    #forward_prob[0,1] = 1-p_0
    forward_prob = forward_prob.at[0,0].set(p_0)
    forward_prob = forward_prob.at[0,1].set(1-p_0)

    # probabilites for the forward steps that won't change
    prob_CU = 1 - jnp.exp(-1*gamma)
    prob_CC = jnp.exp(-1*gamma)
    prob_UC = jnp.zeros(T)
    prob_UU = jnp.zeros(T)

    # calculating the forward probabilities for t=1 to T-1
    for t in range(1,T):

        # calculating the infectious pressure when X[t-1,j]=0
        Xt_minus_0 = X[t-1].copy()
        #Xt_minus_0[j] = 0
        Xt_minus_0 = Xt_minus_0.at[j].set(0)
        globe = jnp.sum(Xt_minus_0) / N
        household = theta[1] * jnp.matmul(h,Xt_minus_0)
        #covariate = theta[2]*age + theta[3]*sex
        transmission = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[t-1]+household*seasonal_matrix_H[t-1]))

        # calculating probabilites based on the previous step
        #prob_UC[t-1] = 1 - jnp.exp(-1*transmission[j])
        #prob_UU[t-1] = jnp.exp(-1*transmission[j])
        prob_UC = prob_UC.at[t-1].set(1 - jnp.exp(-1*transmission[j]))
        prob_UU = prob_UU.at[t-1].set(jnp.exp(-1*transmission[j]))
        log_prob_prev_0 = jnp.log(prob_UU[t-1]*forward_prob[t-1,0] + prob_CU*forward_prob[t-1,1])
        log_prob_prev_1 = jnp.log(prob_UC[t-1]*forward_prob[t-1,0] + prob_CC*forward_prob[t-1,1])

        # calculating the test result bit
        log_prob_TN_0 = jnp.log(spec) * (test_results[t,j]==0)
        log_prob_FP_0 = jnp.log(1-spec) * (test_results[t,j]==1)
        log_prob_TP_1 = jnp.log(sens) * (test_results[t,j]==1)
        log_prob_FN_1 = jnp.log(1-sens) * (test_results[t,j]==0)
        log_prob_test_0 = log_prob_TN_0 + log_prob_FP_0
        log_prob_test_1 = log_prob_TP_1 + log_prob_FN_1

        # calculating the infectious pressure when X[t,j]=1 and when X[t,j]=0
        Xt_0 = X[t].copy()
        #Xt_0[j] = 0
        Xt_0 = Xt_0.at[j].set(0)
        Xt_1 = X[t].copy()
        #Xt_1[j] = 1
        Xt_1 = Xt_1.at[j].set(1)
        globe_0 = jnp.sum(Xt_0) / N
        globe_1 = jnp.sum(Xt_1) / N
        household_0 = theta[1] * jnp.matmul(h,Xt_0)
        household_1 = theta[1] * jnp.matmul(h,Xt_1)
        #covariate = theta[2]*age + theta[3]*sex
        transmission_0 = jnp.array(jnp.exp(covariate)*theta[0]*(globe_0*seasonal_matrix_G[t]+household_0*seasonal_matrix_H[t]))
        transmission_1 = jnp.array(jnp.exp(covariate)*theta[0]*(globe_1*seasonal_matrix_G[t]+household_1*seasonal_matrix_H[t]))

        # events for individuals other than j at time 1
        latent_diff = X[t+1] - X[t]
        #latent_diff[j] = 2
        latent_diff = latent_diff.at[j].set(2)
        not_infect = (latent_diff==0)*(1-X[t+1])
        infect = (latent_diff==1)
        not_recover = (latent_diff==0)*X[t+1]
        recover = (latent_diff==-1)

        # log probabilities of colonisation events when X[0,j]=1
        log_prob_not_infect_1 = jnp.sum(not_infect * (-1*transmission_1))
        log_prob_infect_1 = jnp.sum(jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_1)), nan=0))

        # log probabilities of colonisation events when X[0,j]=0
        log_prob_not_infect_0 = jnp.sum(not_infect * (-1*transmission_0))
        log_prob_infect_0 = jnp.sum(jnp.nan_to_num(infect * jnp.log(1-jnp.exp(-1*transmission_0)), nan=0))

        # log probabilities of recovery events
        log_prob_not_recover = jnp.sum(not_recover * (-1*gamma))
        log_prob_recover = jnp.sum(recover * jnp.log(1-jnp.exp(-1*gamma)))

        # unnormalised log probabilities
        log_prob_1 = log_prob_prev_1 + log_prob_test_1 + log_prob_not_infect_1 + log_prob_infect_1 + log_prob_not_recover + log_prob_recover
        log_prob_0 = log_prob_prev_0 + log_prob_test_0 + log_prob_not_infect_0 + log_prob_infect_0 + log_prob_not_recover + log_prob_recover

        # normalised probabilities
        prob_0 = jnp.exp(log_prob_0-log_prob_1)/(1+jnp.exp(log_prob_0-log_prob_1))
        #forward_prob[t,0] = prob_0
        #forward_prob[t,1] = 1-prob_0
        forward_prob = forward_prob.at[t,0].set(prob_0)
        forward_prob = forward_prob.at[t,1].set(1-prob_0)

    # calculating the infectious pressure when X[T-1,j]=0
    Xt_minus_0 = X[T-1].copy()
    #Xt_minus_0[j] = 0
    Xt_minus_0 = Xt_minus_0.at[j].set(0)
    globe = jnp.sum(Xt_minus_0) / N
    household = theta[1] * jnp.matmul(h,Xt_minus_0)
    #covariate = theta[2]*age + theta[3]*sex
    transmission = jnp.array(jnp.exp(covariate)*theta[0]*(globe*seasonal_matrix_G[T-1]+household*seasonal_matrix_H[T-1]))

    # calculating probabilites based on the previous step
    #prob_UC[T-1] = 1 - jnp.exp(-1*transmission[j])
    #prob_UU[T-1] = jnp.exp(-1*transmission[j])
    prob_UC = prob_UC.at[T-1].set(1 - jnp.exp(-1*transmission[j]))
    prob_UU = prob_UU.at[T-1].set(jnp.exp(-1*transmission[j]))
    log_prob_prev_0 = jnp.log(prob_UU[T-1]*forward_prob[T-1,0] + prob_CU*forward_prob[T-1,1])
    log_prob_prev_1 = jnp.log(prob_UC[T-1]*forward_prob[T-1,0] + prob_CC*forward_prob[T-1,1])

    # calculating the test result bit
    log_prob_TN_0 = jnp.log(spec) * (test_results[T,j]==0)
    log_prob_FP_0 = jnp.log(1-spec) * (test_results[T,j]==1)
    log_prob_TP_1 = jnp.log(sens) * (test_results[T,j]==1)
    log_prob_FN_1 = jnp.log(1-sens) * (test_results[T,j]==0)
    log_prob_test_0 = log_prob_TN_0 + log_prob_FP_0
    log_prob_test_1 = log_prob_TP_1 + log_prob_FN_1

    # unnormalised log probabilities
    log_prob_1 = log_prob_prev_1 + log_prob_test_1
    log_prob_0 = log_prob_prev_0 + log_prob_test_0

    # normalised probabilities
    prob_0 = jnp.exp(log_prob_0-log_prob_1)/(1+jnp.exp(log_prob_0-log_prob_1))
    #forward_prob[T,0] = prob_0
    #forward_prob[T,1] = 1-prob_0
    forward_prob = forward_prob.at[T,0].set(prob_0)
    forward_prob = forward_prob.at[T,1].set(1-prob_0)

    # sampling for X[T,j]
    X_j_new = jnp.zeros(T+1)
    #X_j_new[T] = jnp.random.binomial(1,forward_prob[T,1])
    #X_j_new[T] = (u[T]<forward_prob[T,1])
    X_j_new = X_j_new.at[T].set(u[T]<forward_prob[T,1])

    # sampling backwards for t=T-1 to 0
    for t in reversed(range(T)):
        
        # calculating the unnormalised probabilities
        prob_0_ast = (X_j_new[t+1]*prob_UC[t] + (1-X_j_new[t+1])*prob_UU[t]) * forward_prob[t,0]
        prob_1_ast = (X_j_new[t+1]*prob_CC + (1-X_j_new[t+1])*prob_CU) * forward_prob[t,1]

        # normalising the probabilities
        prob_1 = prob_1_ast/(prob_0_ast+prob_1_ast)

        # sampling for X[t,j]
        #X_j_new[t] = jnp.random.binomial(1,prob_1)
        #X_j_new[t] = (u[t]<prob_1)
        X_j_new = X_j_new.at[t].set(u[t]<prob_1)

    # creating a new X matrix with column j changed
    X_new = X.copy()
    #X_new[:,j] = X_j_new
    X_new = X_new.at[:,j].set(X_j_new)

    # returning the new X matrix
    return X_new

# jit version
iFFBS_jit = jit(iFFBS_jax, static_argnames=["T","N"])