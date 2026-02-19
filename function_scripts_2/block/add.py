# importing numpy
import numpy as np
from numpy import random

# function to add an event
def add_event (C:np.array, m:int, n:int, T:int) :

    # list of inputs:
    # C - current colonisation matrix
    # m - maximum block size change
    # n - number of individuals
    # T - total time

    # randomly choosing which individual to update
    j_change = random.randint(0,n)

    # converting a column of C into an event list
    event_indicator = abs(C[1:(T+1),j_change]-C[0:T,j_change])
    time_events = np.where(event_indicator==1)[0] + 1

    # creating a list of possible times to add an event
    proposal_times = np.array(range(1,T+1))

    # eliminating times where there is an event, and times immediatly before an event
    proposal_times = np.setdiff1d(proposal_times,time_events)
    proposal_times = np.setdiff1d(proposal_times,time_events-1)

    # if there are no times to add an event, return the original colonisation matrix and log_q_move = 0
    if len(proposal_times)==0 :
        return {'C_prop': C, 'log_q_move': 0.0, 'j_change': j_change, 'add_time':"F", 'add_end_time':"F"}
    
    # choosing a time to add a new event
    add_time = random.choice(proposal_times)

    # creating a list of possible times to end the block update
    end_times = np.array(range(add_time+1,T+2))

    # eliminating times outside the maximum block size range
    end_times = end_times[end_times <= add_time+m]

    # eliminating times after the next event
    future_events = time_events[time_events > add_time]
    if len(future_events)>0 :
        next_event = np.min(future_events)
        end_times = end_times[end_times < next_event]

    # choosing a time to end the new event
    add_end_time = random.choice(end_times)

    # adding the new events (and T+1) to the list of events
    time_events_2 = np.append(time_events,[add_time,add_end_time,T+1])
    time_events_2 = np.sort(time_events_2)

    # the differences between event times
    event_differences = time_events_2[1:] - time_events_2[:-1]

    # the number of possible times where we can remove a block
    total_remove = np.sum(event_differences <= m)

    # calculating log_q_move
    log_q_move = np.log(len(proposal_times)) + np.log(len(end_times)) - np.log(total_remove)

    # calculating the proposed C matrix based on the new list of events
    C_prop = C.copy()
    C_prop[add_time:add_end_time,j_change] = 1-C[add_time:add_end_time,j_change]

    # returning the proposed C matrix and log_q_move
    return {'C_prop': C_prop, 'log_q_move': log_q_move, 'j_change': j_change, 'add_time':add_time, 'add_end_time':add_end_time}