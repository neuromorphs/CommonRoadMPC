import numpy as np
from cycler import cycler

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from car import Car
from track import Track
from constants import *


DIST_TOLLERANCE = 2


track = Track()
car = Car(track)
waypoint_index = 1

for i in range(200):
    simulated_history = car.simulate_trajectory_distribution( u_dist )

    trajectory_index = 0
    best_trajectory = 1000000
    best_trajectory_index = 0

    for trajectory in simulated_history:
        for state in trajectory:
            
            position = (state[0], state[1])
            error = track.distance_to_waypoint(position, waypoint_index)
            if(error<best_trajectory):
                best_trajectory = error
                best_trajectory_index = trajectory_index
        trajectory_index += 1

    # print(best_trajectory_index)
    # print("error", best_trajectory)
    if(best_trajectory < DIST_TOLLERANCE):
        waypoint_index +=1
        waypoint_index = waypoint_index % len(track.waypoints_x)
        print("Waypoint Index", waypoint_index)

    best_trajectory_control_sequence = u_dist[best_trajectory_index]
    next_control_input = best_trajectory_control_sequence[0]
    # print(next_control_input)
    # print(car.state[3])
    car.step(next_control_input)

car.draw_history()
car.draw_simulated_history()







 




