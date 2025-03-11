##### real_data_rippler

# ---------------------------------------------------------------------------------------------------------------------------------------------------

### preamble 

# importing numpy
import numpy as np
from numpy import random

# importing system
import sys

# importing pandas
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------------------------------------

### simulating data

# moving upwards
sys.path.append('./.')

# importing functions for model simulation
import function_scripts.model.UC_simulation as UC_simulation_file

# naming the functions
UC_sim = UC_simulation_file.UC_sim

# importing the csv - only the important columns
data_real = pd.read_csv("Antidote_household.csv", usecols=["hid","pid","date","age","sex","hiv","hivChild","cpt","result","week"])

# converting the pid to integers (and saving lists of sex and age)
pid_unique = []
no_individuals = 0
test_individuals = []
sex = []
age = []
for i in range(1659):
    pid = data_real['pid'][i]
    if pid in pid_unique:
        test_individuals.append(pid_unique.index(pid))
    else:
        pid_unique.append(pid)
        test_individuals.append(no_individuals)
        sex.append(data_real['sex'][i])
        age.append(data_real['age'][i])
        no_individuals += 1
sex = np.array(sex)
age = np.array(age)

# centring age and sex
age = age - np.mean(age)
sex = sex - np.mean(sex)

# calculating N and T
N = no_individuals
T = int(max(data_real['week']))

# making a test result matrix (for the real test results)
test_results_real = np.tile(np.nan,(T+1,N))
for i in range(1659):
    j = test_individuals[i]
    t = int(data_real['week'][i])
    test_results_real[t,j] = data_real['result'][i]

# converting the hid to integers
hid_unique = []
no_houses = 0
test_houses = []
for hid in data_real['hid'] :
    if hid in hid_unique:
        test_houses.append(hid_unique.index(hid))
    else:
        hid_unique.append(hid)
        test_houses.append(no_houses)
        no_houses += 1

# list of which house each individual is in
house_list = []
for i in range(N):
    test_no = test_individuals.index(i)
    house_list.append(test_houses[test_no])

# making a household mixing matrix
h = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if house_list[i]==house_list[j]:
            h[i,j] = 1

# assumed true values
gamma = 0.5
sens = 0.8
spec = 0.95

# seasonality modifiers
seasonality_mode = 2
seasonal_period = 52
t_ast = 17

# resulting matrices
seasonal_vector = np.array([1 - np.cos(2*np.pi*(t+t_ast)/seasonal_period) for t in range(T+1)])
seasonal_matrix_G = np.tile(seasonal_vector, (N,1)).T
seasonal_matrix_H = np.tile(1, (T+1,N))

# ---------------------------------------------------------------------------------------------------------------------------------------------------

### inference

# importing function for inference
import function_scripts.rippler.UC_inference_rippler as inference_file

# naming the function
inference_rippler = inference_file.inference_rippler

# hyperparameters of priors
mu = np.array([0.001,0.001,0.001,0.001])
prior_X_0 = 0.35

# starting values
theta_start = np.array([0.5,0.5,0,0])
X_start = UC_sim(N,h,age,sex,prior_X_0,theta_start,gamma,0.1,sens,spec,T,seasonality_mode,seasonal_period,t_ast,1)['X']
# theta_start = theta
# X_start = UC_results['X']
covariance_start = 0.000001*np.identity(4)
nu_0 = 100
delta = 0.05

# defining f
def f(n):
    return int(0.3*n)

# running the MCMC
K = 100000
K_chunk = int(K/1000)
K_latent = 400
MCMC_real_data_rippler = inference_rippler(test_results_real,N,h,gamma,T,seasonal_matrix_G,seasonal_matrix_H,age,sex,sens,spec,theta_start,X_start,covariance_start,nu_0,f,delta,mu,prior_X_0,K,K_latent,K_chunk,'experiments/real_data_rippler/MCMC_output_test',1)