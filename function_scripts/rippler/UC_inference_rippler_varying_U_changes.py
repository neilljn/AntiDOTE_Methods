# importing numpy
import numpy as np
from numpy import random

# importing jax
import jax
import jax.numpy as jnp
from jax import jit

# importing storage for results
import zarr

# importing time
import time

# importing other functions
from function_scripts.rippler.X_from_U import X_from_U_jit
from function_scripts.rippler.U_bounds_from_X import U_bounds_from_X_jit
from function_scripts.rippler.log_dis_U import log_dis_U_jit
from function_scripts.rippler.log_dis_theta import log_dis_theta_jit
from function_scripts.rippler.log_dis_X_0 import log_dis_X_0
from function_scripts.rippler.prop_new_U_bounds import prop_new_U_bounds_jit

# MCMC algorithm for inference using u instead of C
def inference_rippler (test_results:np.array,N:int,h:np.array,gamma:float,T:int,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,sens:float,spec:float,theta_start:np.array,X_start:np.array,covariance_start:np.array,nu_0:float,f,delta:float,mu:np.array,prior_X_0:float,K:int,K_latent:int,K_chunk:int,K_U:int,zarr_names_start:str,seed:int) :

    # list of inputs:
    # test_results - matrix of all individuals colonisation status over time
    # N - number of individuals in the population
    # h - household mixing matrix (n x n)
    # gamma - recovery rate
    # T - length of time to run the simulation for
    # seasonal_matrix_G - matrix of the seasonality term for all individuals over time, as applied to the global term
    # seasonal_matrix_H - matrix of the seasonality term for all individuals over time, as applied to the household term
    # age - vector of the age of each individual
    # sex - vector of the sex of each individual
    # sens - test sensitivity
    # spec - test specificity
    # theta_start - starting values of the parameters
    # X_start - starting matrix of the latent variables
    # covariance_start - starting covariance matrix for theta
    # nu_0
    # f
    # delta
    # mu - vector of hyperparameters for the priors of beta_G, beta_H, delta_A, delta_S
    # prior_X_0 - prior proportion of individuals initially colonised
    # K - number of macroreplications
    # K_latent - number of latent variable changes within each macroreplication
    # K_chunk - number of chunks
    # zarr_names_start - names of the zarr files that the results will be stored in
    # seed - value of the random seed

    # start time
    start_time = time.time()

    # setting the seed
    random.seed(seed)

    # making the current parameter values equal to the initial values
    theta = theta_start
    X = X_start

    # starting values for the adaptive scaling
    covariance = covariance_start
    d = 4
    lambda_current = 2.38/np.sqrt(d)
    change = lambda_current/10
    theta_mean = theta

    # initial number of acceptances in M-H steps
    acc_theta = 0
    acc_initial = 0
    acc_latent = 0
    acc_latent_times = np.repeat(0,T)
    acc_latent_times_total = np.repeat(0,T)
    acc_latent_indiv_total = np.repeat(0,N)

    # the size of each chunk
    chunk_size = int(K/K_chunk)

    # creating the storage arrays in zarr
    theta_store = zarr.open(zarr_names_start+'/theta.zarr', mode='w', shape=(K, 4), chunks=(chunk_size, 4))
    X_store = zarr.open(zarr_names_start+'/X.zarr', mode='w', shape=(K, T+1, N), chunks=(chunk_size, T+1, N))
    like_theta_store = zarr.open(zarr_names_start+'/like_theta.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    like_initial_store = zarr.open(zarr_names_start+'/like_initial.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    like_latent_store = zarr.open(zarr_names_start+'/like_latent.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    covariance_store = zarr.open(zarr_names_start+'/covariance.zarr', mode='w', shape=(K_chunk, 4, 4), chunks=(chunk_size, 4, 4))
    acc_store = zarr.open(zarr_names_start+'/acc.zarr', mode='w', shape=(3, ), chunks=(1, ))
    acc_latent_times_store = zarr.open(zarr_names_start+'/acc_latent_times.zarr', mode='w', shape=(T, ), chunks=(1, ))
    times_chosen_store = zarr.open(zarr_names_start+'/times_chosen.zarr', mode='w', shape=(T, ), chunks=(1, ))
    indiv_chosen_store = zarr.open(zarr_names_start+'/indiv_chosen.zarr', mode='w', shape=(N, ), chunks=(1, ))
    lambda_current_store = zarr.open(zarr_names_start+'/lambda_current.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))

    # creating a temporary storage array for theta (in the RAM)
    theta_store_temp = np.zeros((K,4))
    lambda_current_store_temp = np.zeros((K))

    # running each chunk
    for k_chunk in range(K_chunk):

        # creating temporary storage arrays (in the RAM)
        X_store_temp = np.zeros((chunk_size,T+1,N))
        like_theta_store_temp = np.zeros(chunk_size)
        like_initial_store_temp = np.zeros(chunk_size)
        like_latent_store_temp = np.zeros(chunk_size)

        # start and end values of the macroreplications for this chunk
        k_start = k_chunk*chunk_size
        k_end = (k_chunk+1)*chunk_size

        # running the macroreplications of the MCMC
        for k in range(k_start,k_end):

            # n from k (for adaptive steps)
            n = k + 1

            # proposing a new value of theta
            u_adapt = random.uniform()
            if u_adapt < delta:
                adapt_step = 0
                theta_prop = random.multivariate_normal(theta,covariance_start)
            else:
                adapt_step = 1
                theta_prop = random.multivariate_normal(theta,(lambda_current**2)*covariance)

            # calculating the likelihood of the current and proposed thetas
            ll_curr = log_dis_theta_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,T,N,h,mu)
            ll_prop = log_dis_theta_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta_prop,T,N,h,mu)

            # M-H step to potentially update theta
            log_alpha = ll_prop - ll_curr
            log_u = np.log(random.uniform())
            if log_u < log_alpha :
                theta = theta_prop
                ll_curr = ll_prop
                acc_theta += 1
                lambda_current = lambda_current + adapt_step*2.38*change*(n**(-0.5))
            else:
                lambda_current = lambda_current - adapt_step*change*(n**(-0.5))
            
            # updating the covariance matrix
            theta_mean_previous = theta_mean
            if n==1:
                theta_mean = (theta+theta_start) / 2
                covariance = (np.outer(theta_start,theta_start) + np.outer(theta,theta) - 2*np.outer(theta_mean,theta_mean) + (nu_0+d+1)*covariance) / (nu_0+d+3)
            elif f(n)==f(n-1):
                theta_mean = theta_mean_previous*(n-f(n))/(n-f(n)+1) + theta/(n-f(n)+1)
                covariance = ((n-f(n)+nu_0+d+1)*covariance + np.outer(theta,theta) + (n-f(n))*np.outer(theta_mean_previous,theta_mean_previous) - (n-f(n)+1)*np.outer(theta_mean,theta_mean)) / (n-f(n)+nu_0+d+2)
            else:
                theta_replaced = theta_store_temp[f(n)-2]
                theta_mean = theta_mean_previous + (theta - theta_replaced) / (n-f(n)+1)
                covariance = covariance + (np.outer(theta,theta) - np.outer(theta_replaced,theta_replaced) + (n-f(n)+1)*(np.outer(theta_mean_previous,theta_mean_previous)-np.outer(theta_mean,theta_mean))) / (n-f(n)+nu_0+d+2)

            # the log likelihood for this X matrix
            like = log_dis_U_jit(X,test_results,sens,spec)

            # generating a u matrix for our current X matrix 
            U_bounds = U_bounds_from_X_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
            U = random.uniform(U_bounds['lower'],U_bounds['upper'])

            # proposing new initial conditions
            X_0_prop = np.array(X.copy())
            j_change = random.randint(0,N)
            X_0_prop[0,j_change] = 1-X_0_prop[0,j_change]

            # calculating the log likelihood based on the proposed C
            X_prop = X_from_U_jit(U,np.array([X_0_prop[0]]),seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,N,T,h)
            #C_prop = C_create(u,C_0_prop[0],transmission_rate(394,h,52,3),beta_G,beta_HH,gamma,n,T)
            like_prop = log_dis_U_jit(X_prop,test_results,sens,spec)

            # M-H step to potentially update the initial conditions
            log_alpha = like_prop - like + log_dis_X_0(X_prop,prior_X_0,N) - log_dis_X_0(X,prior_X_0,N)
            log_v = np.log(random.uniform())
            if log_v < log_alpha :
                X = X_prop
                like = like_prop
                like_initial = like_prop + log_dis_X_0(X_prop,prior_X_0,N)
                acc_initial += 1
            else:
                like_initial = like + log_dis_X_0(X,prior_X_0,N)

            # generating a u matrix for our current X matrix 
            U_bounds = U_bounds_from_X_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)

            # update the latent variables K_latent times
            for k_latent in range(K_latent):

                # generating a u matrix for our current X matrix 
                U = random.uniform(U_bounds['lower'],U_bounds['upper'])

                # randomly choosing which U to move
                U_prop = U.copy()
                U_prop_widths_matrix = 1-U_bounds['upper']+U_bounds['lower']
                #total_prop_width = np.sum(U_prop_widths_matrix)
                #u_prop_all = random.uniform(0,total_prop_width)
                U_prop_widths_vector = np.reshape(U_prop_widths_matrix,N*T)
                U_prop_widths_vector_cumsum = np.cumsum(U_prop_widths_vector)
                total_prop_width = U_prop_widths_vector_cumsum[-1]

                # proposing a new u
                for k_u in range(K_U):
                    u_prop_all = random.uniform(0,total_prop_width)
                    index = np.sum(U_prop_widths_vector_cumsum < u_prop_all)
                    j_change = index % N
                    t_change = int((index-j_change)/N)
                    U_prop_bounds = prop_new_U_bounds_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h,t_change,j_change)
                    U_prop[t_change,j_change] = random.uniform(U_prop_bounds['lower'],U_prop_bounds['upper'])

                # calculating the log likelihood based on the proposed u
                X_prop = X_from_U_jit(U_prop,np.array([X[0]]),seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,N,T,h)
                like_prop = log_dis_U_jit(X_prop,test_results,sens,spec)

                # finding log_q_move
                U_reverse_bounds = U_bounds_from_X_jit(X_prop,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
                U_reverse_widths_matrix = 1-U_reverse_bounds['upper']+U_reverse_bounds['lower']
                total_reverse_width = np.sum(U_reverse_widths_matrix)
                log_q_move = K_U * (np.log(total_prop_width) - np.log(total_reverse_width))

                # M-H step to potentially update the latent variables
                log_alpha = like_prop - like + log_q_move
                log_v = np.log(random.uniform())
                if log_v < log_alpha :
                    X = X_prop
                    U_bounds = U_reverse_bounds
                    like = like_prop
                    acc_latent += 1
                    #acc_latent_times[t_change] += 1

            # saving current parameter values to the storage (in the RAM)
            k_temp = k % chunk_size
            theta_store_temp[k] = theta
            lambda_current_store_temp[k] = lambda_current
            X_store_temp[k_temp] = X
            like_theta_store_temp[k_temp] = ll_curr
            like_initial_store_temp[k_temp] = like_initial
            like_latent_store_temp[k_temp] = like

            # printing the current iteration number
            print("Completed iterations:", k+1, end='\r')

        # saving the covariance matrix to the zarr storage
        covariance_store[k_chunk] = covariance

        # saving current parameter and likelihood values to the zarr storage
        X_store[k_start:k_end] = X_store_temp
        like_theta_store[k_start:k_end] = like_theta_store_temp
        like_initial_store[k_start:k_end] = like_initial_store_temp
        like_latent_store[k_start:k_end] = like_latent_store_temp

    # saving the acceptance rates to the zarr storage
    theta_store[:] = theta_store_temp
    lambda_current_store[:] = lambda_current_store_temp
    acc_store[0] = acc_theta/K
    acc_store[1] = acc_initial/K
    acc_store[2] = acc_latent/(K*K_latent)
    #acc_latent_times_store[:] = acc_latent_times/acc_latent_times_total
    times_chosen_store[:] = acc_latent_times_total
    indiv_chosen_store[:] = acc_latent_indiv_total
    
    # returning outputs
    #output = {'acc_theta': acc_theta/K, 'acc_initial': acc_initial/K, 'acc_latent': acc_latent/(K*K_latent), 'acc_latent_times':acc_latent_times/acc_latent_times_total}
    #return output

    # list of outputs:
    # acc_theta - acceptance rate of the theta update (RWM)
    # acc_initial - acceptance rate of the initial conditions update
    # acc_latent - acceptance rate of the latent variable update
    # acc_latent_times - acceptance rate of the latent variable update at each point in time

    # returning time taken
    return(time.time()-start_time)