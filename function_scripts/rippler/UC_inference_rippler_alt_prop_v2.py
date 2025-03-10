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
from function_scripts.rippler.X_from_U import X_from_U_jit
from function_scripts.rippler.U_bounds_from_X import U_bounds_from_X_jit
from function_scripts.rippler.log_dis_U import log_dis_U_jit
from function_scripts.rippler.log_dis_theta import log_dis_theta_jit
from function_scripts.rippler.log_dis_X_0 import log_dis_X_0
from function_scripts.rippler.prop_new_U_bounds import prop_new_U_bounds_jit

# MCMC algorithm for inference using u instead of C
@teams_sender("https://livelancsac.webhook.office.com/webhookb2/3d6a96cd-dfcc-4879-a4b4-fbf819b2b0aa@9c9bcd11-977a-4e9c-a9a0-bc734090164a/IncomingWebhook/fc46931452294793b7c5f4fa55c75b07/e07e4eb0-8af1-4f92-8974-01147fb1408a")
def inference_rippler (test_results:np.array,N:int,h:np.array,gamma:float,T:int,seasonal_matrix_G:np.array,seasonal_matrix_H:np.array,age:np.array,sex:np.array,sens:float,spec:float,theta_start:np.array,X_start:np.array,covariance_start:np.array,scaling:float,mu:np.array,prior_X_0:float,K:int,K_latent:int,K_chunk:int,K_U:int,zarr_names_start:str,seed:int) :

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
    # zarr_names_start - names of the zarr files that the results will be stored in
    # seed - value of the random seed

    # start time
    start_time = time.time()

    # setting the seed
    random.seed(seed)

    # making the current parameter values equal to the initial values
    theta = theta_start
    X = X_start

    # creating the inital covariance matrix
    #scaling = ((2.38**2)/4)
    #covariance = scaling*scaling_start*np.identity(4)
    covariance = covariance_start

    # # calculating the intital likelihood
    # like = log_dis_u_jit(C,test_results,sens,spec)

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
    U_store = zarr.open(zarr_names_start+'/U.zarr', mode='w', shape=(K, T, N), chunks=(chunk_size, T, N))
    like_theta_store = zarr.open(zarr_names_start+'/like_theta.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    like_initial_store = zarr.open(zarr_names_start+'/like_initial.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    like_latent_store = zarr.open(zarr_names_start+'/like_latent.zarr', mode='w', shape=(K, ), chunks=(chunk_size, ))
    covariance_store = zarr.open(zarr_names_start+'/covariance.zarr', mode='w', shape=(K_chunk, 4, 4), chunks=(chunk_size, 4, 4))
    acc_store = zarr.open(zarr_names_start+'/acc.zarr', mode='w', shape=(3, ), chunks=(1, ))
    acc_latent_times_store = zarr.open(zarr_names_start+'/acc_latent_times.zarr', mode='w', shape=(T, ), chunks=(1, ))
    times_chosen_store = zarr.open(zarr_names_start+'/times_chosen.zarr', mode='w', shape=(T, ), chunks=(1, ))
    indiv_chosen_store = zarr.open(zarr_names_start+'/indiv_chosen.zarr', mode='w', shape=(N, ), chunks=(1, ))

    # running each chunk
    for k_chunk in range(K_chunk):

        # creating temporary storage arrays (in the RAM)
        theta_store_temp = np.zeros((chunk_size,4))
        X_store_temp = np.zeros((chunk_size,T+1,N))
        U_store_temp = np.zeros((chunk_size,T,N))
        like_theta_store_temp = np.zeros(chunk_size)
        like_initial_store_temp = np.zeros(chunk_size)
        like_latent_store_temp = np.zeros(chunk_size)

        # start and end values of the macroreplications for this chunk
        k_start = k_chunk*chunk_size
        k_end = (k_chunk+1)*chunk_size

        # running the macroreplications of the MCMC
        for k in range(k_start,k_end):

            # proposing a new value of theta
            theta_prop = random.multivariate_normal(theta,covariance)

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

            # update the latent variables K_latent times
            for k_latent in range(K_latent):

                # # generating a u matrix for our current X matrix 
                # U_bounds = U_bounds_from_X_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
                # U = random.uniform(U_bounds['lower'],U_bounds['upper'])

                # # randomly choosing which u to move
                # U_prop = U.copy()
                # U_prop_widths_matrix = 1-U_bounds['upper']+U_bounds['lower']
                # U_prop_widths_vector = np.reshape(U_prop_widths_matrix,N*T)
                # probs = U_prop_widths_vector/np.sum(U_prop_widths_vector)
                # print(np.sum(probs))
                # index = random.choice(N*T,p=probs)
                # j_change = index % N
                # t_change = int((index-j_change)/N)
                # acc_latent_times_total[t_change] += 1

                # # propsing a new u
                # U_prop_bounds = prop_new_U_bounds_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h,t_change,j_change)
                # U_prop[t_change,j_change] = random.uniform(U_prop_bounds['lower'],U_prop_bounds['upper'])
                
                # # calculating the log likelihood based on the proposed u
                # X_prop = X_from_U_jit(U_prop,np.array([X[0]]),seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,N,T,h)
                # like_prop = log_dis_U_jit(X_prop,test_results,sens,spec)

                # # finding log_q_move
                # U_reverse_bounds = U_bounds_from_X_jit(X_prop,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
                # U_reverse_widths_matrix = 1-U_reverse_bounds['upper']+U_reverse_bounds['lower']
                # log_q_move = np.log(U_reverse_widths_matrix[t_change,j_change]) - np.log(np.sum(U_reverse_widths_matrix)) - np.log(U_prop_widths_matrix[t_change,j_change]) + np.log(np.sum(U_prop_widths_matrix))

                # generating a u matrix for our current X matrix 
                U_bounds = U_bounds_from_X_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
                U = random.uniform(U_bounds['lower'],U_bounds['upper'])

                # randomly choosing which u to move
                U_prop = U.copy()
                U_prop_widths_matrix = 1-U_bounds['upper']+U_bounds['lower']
                total_prop_width = np.sum(U_prop_widths_matrix)
                U_prop_widths_vector = np.reshape(U_prop_widths_matrix,N*T)
                U_prop_widths_vector_cumsum = np.cumsum(U_prop_widths_vector)
                for k_u in range(K_U):
                    u_prop_all = random.uniform(0,total_prop_width)
                    index = np.sum(U_prop_widths_vector_cumsum < u_prop_all)
                    j_change = index % N
                    t_change = int((index-j_change)/N)
                    acc_latent_indiv_total[j_change] += 1
                    acc_latent_times_total[t_change] += 1
                    U_prop_bounds = prop_new_U_bounds_jit(X,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h,t_change,j_change)
                    U_prop[t_change,j_change] = random.uniform(U_prop_bounds['lower'],U_prop_bounds['upper'])

                # calculating the log likelihood based on the proposed u
                X_prop = X_from_U_jit(U_prop,np.array([X[0]]),seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,N,T,h)
                like_prop = log_dis_U_jit(X_prop,test_results,sens,spec)

                # finding log_q_move
                U_reverse_bounds = U_bounds_from_X_jit(X_prop,seasonal_matrix_G,seasonal_matrix_H,age,sex,theta,gamma,T,N,h)
                U_reverse_widths_matrix = 1-U_reverse_bounds['upper']+U_reverse_bounds['lower']
                total_reverse_width = np.sum(U_reverse_widths_matrix)
                log_q_move = np.log(total_prop_width) - np.log(total_reverse_width)

                # M-H step to potentially update the latent variables
                log_alpha = like_prop - like + log_q_move
                log_v = np.log(random.uniform())
                if log_v < log_alpha :
                    X = X_prop
                    U = U_prop
                    like = like_prop
                    acc_latent += 1
                    acc_latent_times[t_change] += 1

            # saving current parameter values to the storage (in the RAM)
            k_temp = k % chunk_size
            theta_store_temp[k_temp] = theta
            X_store_temp[k_temp] = X 
            U_store_temp[k_temp] = U
            like_theta_store_temp[k_temp] = ll_curr
            like_initial_store_temp[k_temp] = like_initial
            like_latent_store_temp[k_temp] = like

            # printing the current iteration number
            print("Completed iterations:", k+1, end='\r')
        
        # updating the covariance matrix of theta
        beta_G_mean = np.mean(theta_store_temp[:,0])
        beta_H_mean = np.mean(theta_store_temp[:,1])
        delta_A_mean = np.mean(theta_store_temp[:,2])
        delta_S_mean = np.mean(theta_store_temp[:,3])
        K_expected = np.tile([beta_G_mean,beta_H_mean,delta_A_mean,delta_S_mean],(chunk_size,1))
        K_tilda = theta_store_temp-K_expected
        covariance = scaling*(1/(chunk_size-1))*np.matmul(np.transpose(K_tilda),K_tilda)

        # saving the covariance matrix to the zarr storage
        covariance_store[k_chunk] = covariance

        # saving current parameter and likelihood values to the zarr storage
        theta_store[k_start:k_end] = theta_store_temp
        X_store[k_start:k_end] = X_store_temp
        U_store[k_start:k_end] = U_store_temp
        like_theta_store[k_start:k_end] = like_theta_store_temp
        like_initial_store[k_start:k_end] = like_initial_store_temp
        like_latent_store[k_start:k_end] = like_latent_store_temp

    # saving the acceptance rates to the zarr storage
    acc_store[0] = acc_theta/K
    acc_store[1] = acc_initial/K
    acc_store[2] = acc_latent/(K*K_latent)
    acc_latent_times_store[:] = acc_latent_times/acc_latent_times_total
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