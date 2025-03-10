# importing numpy
import numpy as np
from numpy import random

# function to determine lambda at time t
def transmission_rate(N:int,h:np.array,seasonality_mode:int,seasonal_period:float,t_ast:float,age:np.array,sex:np.array) :

    # list of inputs:
    # N - number of individuals in the population
    # h - household mixing matrix (n x n)
    # seasonality_mode - which model of seasonality we are using (0, 1, or 2)
    # seasonal_period - number of time units in one year
    # t_ast - offset to match seasonality with the wet-dry cycle
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual

    # defining a function inside this function
    def transmission_rate_interior(t:int,C:np.array,theta:np.array) :

        # list of inputs:
        # t - current time
        # C - vector of length n, indicating if each individual is currently colonised
        # theta - beta_G, beta_H, delta_A, delta_S

        # calculating the seasonal component
        seasonal = 1 - np.cos(2*np.pi*(t+t_ast)/seasonal_period)
        #seasonal = 1

        # calculating the global component
        #globe = beta_G * sum(C) / n
        globe = sum(C) / N

        # calculating the within-household component
        household = theta[1] * np.matmul(h,C)

        # calculating the covariate component
        covariate = theta[2]*age + theta[3]*sex
        
        # combining the three components, returning the overall transmission rate
        if seasonality_mode==0:
            return(np.exp(covariate)*theta[0]*(globe+household))
        elif seasonality_mode==1:
            return(np.exp(covariate)*theta[0]*seasonal*(globe+household))
        elif seasonality_mode==2:
            return(np.exp(covariate)*theta[0]*(globe*seasonal+household))

    # returning the interior function
    return transmission_rate_interior

# function to simulate from the model
def UC_sim(N:int,h:np.array,age:np.array,sex:np.array,prop_0:float,theta:np.array,gamma:float,test_prob:float,sens:float,spec:float,T:int,seasonality_mode:int,seasonal_period:float,t_ast:float,seed:int) :

    # list of inputs:
    # N - number of individuals in the population
    # h - household mixing matrix (n x n)
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # prop_0 - proportion of individuals initially colonised
    # theta - beta_G, beta_H, delta_A, delta_S
    # gamma - recovery rate
    # test_prob - probability each individual is tested each day
    # sens - true positive rate
    # spec - true negative rate
    # T - length of time to run the simulation for
    # seasonality_mode - which model of seasonality we are using (0, 1, or 2)
    # seasonal_period - number of time units in one year
    # t_ast - offset to match seasonality with the wet-dry cycle
    # seed - value of the random seed

    # setting the seed
    random.seed(seed)

    # creating initial arrays to repesent the current infection status of each individual
    C = random.binomial(1,prop_0,N)
    U = 1-C
    
    # creating arrays to store the infection status of each individual over time
    U_store = U
    C_store = C

    # creating an array to store the test results
    test_results = np.repeat(np.nan, N*(T+1))
    test_results.shape = (T+1,N)

    # # simulating potetial tests for each individual
    N_tests = 0
    # test_occurance = random.binomial(1,test_prob,n)
    # n_tests = sum(test_occurance)
    # test_occurance = np.where(test_occurance==0, np.nan, test_occurance)
    # pos_prob = (sens*C)+((1-spec)*U)
    # test_outcome = random.binomial(1,pos_prob,n)
    # test_results[0] = test_occurance * test_outcome

    # setting the transmission rate function
    transmission = transmission_rate(N,h,seasonality_mode,seasonal_period,t_ast,age,sex)

    # simualting for T time units
    for t in range(T):

        # creating arrays for transistions
        UC = np.repeat(0,N)
        CU = np.repeat(0,N)

        # simulating transitions for each individual
        UC = random.binomial(U,1-np.exp(-1*transmission(t,C,theta)))
        CU = random.binomial(C,1-np.exp(-1*gamma))

        # applying the transitions
        U = U - UC + CU
        C = C + UC - CU

        # simulating potetial tests for each individual
        test_occurance = random.binomial(1,test_prob,N)
        N_tests += sum(test_occurance)
        test_occurance = np.where(test_occurance==0, np.nan, test_occurance)
        pos_prob = (sens*C)+((1-spec)*U)
        test_outcome = random.binomial(1,pos_prob,N)
        test_results[t+1] = test_occurance * test_outcome

        # storing new values of U and C
        U_store = np.vstack((U_store, U))
        C_store = np.vstack((C_store, C))

    # returning outputs
    output = {'U':U_store, 'C':C_store, 'X':C_store, 'sum_U':np.sum(U_store,axis=1), 'sum_C':np.sum(C_store,axis=1), 'test_results': test_results, 'N_tests': N_tests}
    return output

    # list of outputs:
    # U - array of uncolonised status for each individual over time
    # C - array of colonised status for each individual over time
    # X - the same as C
    # sum_U - total number of uncolonised individuals over time
    # sum_C - total number of uncolonised individuals over time
    # test_results - matrix of test_results
    # n_tests - number of tests performed