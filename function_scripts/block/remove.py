# importing numpy
import numpy as np
from numpy import random

# function to remove an event
def remove_event (C:np.array, m:int, n:int, T:int) :

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

    # adding T+1 to the list of events
    time_events_2 = np.append(time_events,[T+1])

    # the differences between event times
    event_differences = time_events_2[1:] - time_events_2[:-1]

    # a binary vector: 1 if we can remove an event, 0 if we can't
    proposal_bool = (event_differences <= m)

    # if there are no events we can remove, return the original colonisation matrix and log_q_move = 0
    if np.sum(proposal_bool)==0 :
        return {'C_prop': C, 'log_q_move': 0.0, 'j_change': j_change, 'remove_time':"F"}

    # creating a list of event times we could remove
    proposal_times = time_events*proposal_bool
    proposal_times = proposal_times[proposal_times!=0]

    # choosing an event to remove
    remove_time = random.choice(proposal_times)

    # finding the next event after the removed event
    remove_time_index = np.where(time_events == remove_time)[0]
    next_event_time = time_events_2[remove_time_index+1][0]

    # creating a list of events with the chosen event removed
    time_events_3 = np.setdiff1d(time_events,np.array([remove_time,next_event_time]))

    # creating a list of possible times to add an event
    add_times = np.array(range(1,T+1))

    # eliminating times where there is an event, and times immediatly before an event
    add_times = np.setdiff1d(add_times,time_events_3)
    add_times = np.setdiff1d(add_times,time_events_3-1)

    # creating a list of possible times to end the reverse update
    end_times = np.array(range(remove_time+1,T+2))

    # eliminating times outside the maximum block size range
    end_times = end_times[end_times <= remove_time+m]

    # eliminating times after the next event
    future_events = time_events_3[time_events_3 > remove_time]
    if len(future_events)>0 :
        next_event = np.min(future_events)
        end_times = end_times[end_times < next_event]

    # calculating log_q_move
    log_q_move = np.log(len(proposal_times)) - np.log(len(add_times)) - np.log(len(end_times))

    # calculating the proposed C matrix based on the new list of events
    C_prop = C.copy()
    C_prop[remove_time:next_event_time,j_change] = 1-C[remove_time:next_event_time,j_change]

    # returning the proposed C matrix and log_q_move
    return {'C_prop': C_prop, 'log_q_move': log_q_move, 'j_change': j_change, 'remove_time':remove_time}
    