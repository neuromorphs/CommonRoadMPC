import numpy as np

from cycler import cycler

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from car import Car
from track import Track
from constants import *

import shapely.geometry as geom

# coords_x = [1,2,3]
# coords_y = [1,1,1]

# coords = [[coords_x[i], coords_y[i]] for i in range(len(coords_x))]
# print(coords)
# # coords = [[1,1], [2,1], [3,1]] 
# line = geom.LineString(coords)
# point = geom.Point(5,3)
# print (point.distance(line))

# exit()


COVARIANCE = [[0.2, 0], [0, 0]] 
NUMBER_OF_TRAJECTORIES = 100
DIST_TOLLERANCE = 5


def sample_control_inputs(control_input):
    x, y = np.random.multivariate_normal(control_input, COVARIANCE, NUMBER_OF_TRAJECTORIES).T
    # plt.plot(x, y, 'x')
    # plt.savefig('input_distribution.png')

    control_input_sequences = [0]*len(x)
    for i in range(len(x)):
        control_input_sequences[i] = 20*[[round(x[i],3), round(y[i], 3)]]
    return control_input_sequences


u = [0,0]

control_inputs = sample_control_inputs(u)

# print("CONTROL INPUITS")
# print(control_inputs)
# print("UDIST")
# print(u_dist)

# exit()
# x = control_inputs[:,0]
# y = control_inputs[:,1]


# print(control_inputs)



track = Track()
car = Car(track)
waypoint_index = 1
last_control_input = [0,0]


# car.simulate_trajectory_distribution(control_inputs)
# car.draw_simulated_history()
# exit()

for i in range(60):

    # input_samples = u_dist
    input_samples = sample_control_inputs(last_control_input)
    simulated_history = car.simulate_trajectory_distribution( input_samples )

    trajectory_index = 0
    best_trajectory = 1000000
    best_trajectory_index = 0
   

    for trajectory in simulated_history:
        error = 0

        error = car.cost_function(trajectory, waypoint_index)
        print(error)

        if(error<best_trajectory):
            best_trajectory = error
            best_trajectory_index = trajectory_index
            
        trajectory_index += 1

    print("Best index:", best_trajectory_index)
    print("error", best_trajectory)
    # exit(); /

    distance_car_to_waypoint = track.distance_to_waypoint(car.state[:2], waypoint_index)

    # if(best_trajectory < DIST_TOLLERANCE):
    if(distance_car_to_waypoint < DIST_TOLLERANCE):
        waypoint_index +=1
        waypoint_index = waypoint_index % len(track.waypoints_x)
        print("Waypoint Index", waypoint_index)

    best_trajectory_control_sequence = input_samples[best_trajectory_index]
    next_control_input = best_trajectory_control_sequence[0]
    print(next_control_input)
    last_control_input = next_control_input
    last_control_input[0] = min(last_control_input[0], 1)
    last_control_input[1] = min(last_control_input[1], 0.5)
    print(car.state[3])
    car.step(next_control_input)
    car.draw_simulated_history(waypoint_index)
    car.draw_history()

car.draw_history()







 




