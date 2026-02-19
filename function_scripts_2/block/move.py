# importing numpy
import numpy as np
from numpy import random

# function to move an event
def move_event (C:np.array, m:int, n:int, T:int) :

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

    # if no events, return the original colonisation matrix and log_q_move = 0
    if len(time_events)==0 :
        return {'C_prop': C, 'log_q_move': 0.0, 'j_change': j_change, 'event_change_time': "F", 'event_moved_time': "F"}

    # randomly choosing an event to move
    event_change_index = random.randint(0,len(time_events))
    event_change_time = time_events[event_change_index]

    # creating a list of possible times to move the chosen event to
    proposal_times = np.array(range(1,T+1))

    # eliminating times outside the maximum block size range
    proposal_times = proposal_times[proposal_times >= event_change_time-m]
    proposal_times = proposal_times[proposal_times <= event_change_time+m]
    
    # eliminating times because of other events
    if event_change_index != 0 :
        proposal_times = proposal_times[proposal_times > time_events[event_change_index-1]]
    if event_change_index != (len(time_events)-1) :
        proposal_times = proposal_times[proposal_times < time_events[event_change_index+1]]
    
    # eliminating the time of the event
    proposal_times = proposal_times[proposal_times != event_change_time]

    # if no times to move the event to, return the original colonisation matrix and log_q_move = 0
    if len(proposal_times)==0 :
        return {'C_prop': C, 'log_q_move': 0.0, 'j_change': j_change, 'event_change_time':event_change_time, 'event_moved_time': "F"}
    
    # choosing a time to move the event to
    event_moved_time = random.choice(proposal_times)

    # making a list of event times with one event moved
    time_events_2 = time_events
    time_events_2[event_change_index] = event_moved_time

    # creating a list of possible times to move the chosen event to
    reverse_times = np.array(range(1,T+1))

    # eliminating times outside the maximum block size range
    reverse_times = reverse_times[reverse_times >= event_moved_time-m]
    reverse_times = reverse_times[reverse_times <= event_moved_time+m]
    
    # eliminating times because of other events
    if event_change_index != 0 :
        reverse_times = reverse_times[reverse_times > time_events_2[event_change_index-1]]
    if event_change_index != (len(time_events)-1) :
        reverse_times = reverse_times[reverse_times < time_events_2[event_change_index+1]]
    
    # eliminating the time of the event
    reverse_times = reverse_times[reverse_times != event_moved_time]

    # calculating log_q_move
    log_q_move = np.log(len(proposal_times)) - np.log((len(reverse_times)))

    # calculating the proposed C matrix based on the new list of events
    C_prop = C.copy()
    time_min = np.min((event_change_time,event_moved_time))
    time_max = np.max((event_change_time,event_moved_time))
    C_prop[time_min:time_max,j_change] = 1-C[time_min:time_max,j_change]

    # returning the proposed C matrix and log_q_move
    return {'C_prop': C_prop, 'log_q_move': log_q_move, 'j_change': j_change, 'event_change_time':event_change_time, 'event_moved_time':event_moved_time}