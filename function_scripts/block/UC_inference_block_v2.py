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

# importing teams messaging
from knockknock import teams_sender
#@teams_sender("https://livelancsac.webhook.office.com/webhookb2/3d6a96cd-dfcc-4879-a4b4-fbf819b2b0aa@9c9bcd11-977a-4e9c-a9a0-bc734090164a/IncomingWebhook/fc46931452294793b7c5f4fa55c75b07/e07e4eb0-8af1-4f92-8974-01147fb1408a")

# importing other functions
from function_scripts.block.log_dis_X import log_dis_X_jit
from function_scripts.block.log_dis_theta import log_dis_theta_jit
from function_scripts.block.log_dis_X_0 import log_dis_X_0_jit
from function_scripts.block.move import move_event
from function_scripts.block.add import add_event
from function_scripts.block.remove import remove_event

# MCMC algorithm for inference using block updates
@teams_sender("https://livelancsac.webhook.office.com/webhookb2/3d6a96cd-dfcc-4879-a4b4-fbf819b2b0aa@9c9bcd11-977a-4e9c-a9a0-bc734090164a/IncomingWebhook/fc46931452294793b7c5f4fa55c75b07/e07e4eb0-8af1-4f92-8974-01147fb1408a")
def inference_block (test_results:np.array,N:int,h:np.array,gamma:float,T:int,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,sens:float,spec:float,theta_start:np.array,X_start:np.array,covariance_start:np.array,nu_0:float,f,delta:float,mu:np.array,prior_X_0:float,K:int,K_latent:int,K_chunk:int,m:int,zarr_names_start:str,seed:int) :

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
    # scaling - how much to scale the theta covariance matrix by
    # mu - vector of hyperparameters for the priors of beta_G, beta_H, delta_A, delta_S
    # prior_X_0 - prior proportion of individuals initially colonised
    # K - number of macroreplications
    # K_latent - number of latent variable changes within each macroreplication
    # K_chunk - number of chunks
    # m - maximum block update size
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

            # proposing new initial conditions
            X_0_prop = np.array(X.copy())
            j_change = random.randint(0,N)
            X_0_prop[0,j_change] = 1-X_0_prop[0,j_change]

            # calculating the log likelihoods
            ll_prop = log_dis_X_0_jit(X_0_prop,prior_X_0,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
            ll_curr = log_dis_X_0_jit(X,prior_X_0,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)

            # M-H step to potentially update the initial conditions
            log_alpha = ll_prop - ll_curr
            log_v = np.log(random.uniform())
            if log_v < log_alpha :
                X = X_0_prop
                like_initial = ll_prop
                acc_initial += 1
            else:
                like_initial = ll_curr
            #like_theta = 0
            #like_initial = 0

            # calculating the log likelihood of the latent variables
            like = log_dis_X_jit(X,test_results,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,sens,spec,T,N,h)

            # update the latent variables K_latent times
            for k_latent in range(K_latent):

                # proposing new latent variables: either move, add, or remove
                choice = random.choice(np.array([1,2,3]))
                # choice = random.choice(np.array([2,3]))
                # choice = 1
                if choice == 1:
                    latent_new = move_event(X,m,N,T)
                elif choice == 2:
                    latent_new = add_event(X,m,N,T)
                else:
                    latent_new = remove_event(X,m,N,T)

                # extracting the new colonisation matrix and q_move
                X_prop = latent_new['C_prop']
                log_q_move = latent_new['log_q_move']
                
                # calculating the log likelihood based on the proposed X
                like_prop = log_dis_X_jit(X_prop,test_results,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,sens,spec,T,N,h)

                # M-H step to potentially update the latent variables
                log_alpha = like_prop - like + log_q_move
                log_v = np.log(random.uniform())
                if log_v < log_alpha :
                    X = X_prop
                    like = like_prop
                    acc_latent += 1

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
    
    # returning outputs
    #output = {'acc_theta': acc_theta/K, 'acc_initial': acc_initial/K, 'acc_latent': acc_latent/(K*K_latent), 'acc_latent_times':acc_latent_times/acc_latent_times_total}
    #return outputs

    # list of outputs:
    # acc_theta - acceptance rate of the theta update (RWM)
    # acc_initial - acceptance rate of the initial conditions update
    # acc_latent - acceptance rate of the latent variable update
    # acc_latent_times - acceptance rate of the latent variable update at each point in time

    # returning time taken
    return(time.time()-start_time)