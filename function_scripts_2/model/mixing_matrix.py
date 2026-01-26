# importing numpy
import numpy as np
from numpy import random

# function to create a household mixing matrix
def mixing_matrix(no_houses:int,mean_size:float,seed:int) :

    # list of inputs
    # no_houses - number of houses
    # mean_size - average number of individuals per house
    # seed - value of the random seed

    # setting the seed
    random.seed(seed)

    # the intial number of individuals (+2 dummy indivduals)
    N = 2

    # creating a matrix (with 2 dummy individuals)
    h = np.array([[1,1],[1,1]])

    # iterating over every household
    for i in range(no_houses) :

        # generating how many individuals are in this house
        house_size = random.poisson(mean_size,1)

        # appending the matrix with new columns
        new_column = np.repeat(0,N)
        new_column.shape = (N,1)
        for j in range(house_size[0]) :
            h = np.hstack((h,new_column))

        # appending the matrix with new rows
        new_row = np.concatenate((np.repeat(0,N),np.repeat(1,house_size[0])))
        for j in range(house_size[0]) :
            h = np.vstack((h,new_row))

        # updating the total number of individuals
        N = N + house_size[0]

    # removing the dummy individuals
    N -= 2
    h = np.delete(h,0,0)
    h = np.delete(h,0,0)
    h = np.delete(h,0,1)
    h = np.delete(h,0,1)

    # returning the mixing matrix
    return h