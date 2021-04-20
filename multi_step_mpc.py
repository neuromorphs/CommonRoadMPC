import numpy as np
import math
from car import Car
from track import Track
from constants import *
import matplotlib.pyplot as plt
# a = [1,2,3,4,5,6]
# a.extend(a)
# print(a)

# exit()

def column(matrix, i):
        return [row[i] for row in matrix]
        


COVARIANCE = [[0.1, 0], [0, 0.05]] 
NUMBER_OF_TRAJECTORIES = 50
DIST_TOLLERANCE = 4


def sample_control_inputs(control_input):
    x, y = np.random.multivariate_normal(control_input, COVARIANCE, NUMBER_OF_TRAJECTORIES).T
    # plt.plot(x, y, 'x')
    # plt.savefig('input_distribution.png')

    control_input_sequences = [0]*len(x)
    for i in range(len(x)):
        control_input_sequences[i] = 10*[[round(x[i],3), round(y[i], 3)]]
    return control_input_sequences


u = [0,0]

control_inputs = sample_control_inputs(u)


track = Track()
car = Car(track)
waypoint_index = 1
last_control_input = [0,0]


# index = track.get_closest_index([70,40])
# print(index)
# exit()


# car.simulate_trajectory_distribution(control_inputs)
# car.draw_simulated_history()
# exit()

for i in range(200):

    # input_samples = u_dist
    input_samples = sample_control_inputs(last_control_input)
    simulated_history = car.simulate_trajectory_distribution( input_samples )

    trajectory_index = 0    
    lowest_cost = 100000
    best_index = 0

    weights = np.zeros(len(simulated_history))
    for trajectory in simulated_history:
        cost = car.cost_function(trajectory, waypoint_index)

        #find best
        if cost < lowest_cost:
            best_index = trajectory_index
            lowest_cost = cost

        weight =  math.exp(-1 * cost)
        weights[trajectory_index] = weight
        trajectory_index += 1

    # print("weights:", weights)  


    best_input = input_samples[best_index]
    inputs = np.array(column(input_samples,0))
   
    # print("Input samples",inputs)
    # print("Best input", best_input)
    






    #Finding weighted avg input
    u_sum = np.zeros(2)
    total_weight = 0
    for i in range (len(weights)):
        total_weight += weights[i]
        u_sum = np.add(u_sum, weights[i] * inputs[i])
    print( u_sum)
    # print( "total_weight", total_weight)

    weighted_avg_input = u_sum/total_weight


   
    # exit()
    next_control_input = weighted_avg_input
    
    # exit(); /

    distance_car_to_waypoint = track.distance_to_waypoint(car.state[:2], waypoint_index)

    # if(best_trajectory < DIST_TOLLERANCE):
    if(distance_car_to_waypoint < DIST_TOLLERANCE):
        waypoint_index +=1
        waypoint_index = waypoint_index % len(track.waypoints_x)
        print("Waypoint Index", waypoint_index)

    # best_trajectory_control_sequence = input_samples[best_trajectory_index]
    # next_control_input = best_trajectory_control_sequence[0]
    # print(next_control_input)

    last_control_input = next_control_input
    last_control_input[0] = min(last_control_input[0], 1)
    last_control_input[1] = min(last_control_input[1], 0.5)

    print(car.state[3])
    

    weighted_avg_trajectory = car.simulate_trajectory([next_control_input] *10)
    print("next ctr input",  next_control_input)
   
    print("weighted_avg_trajectory",weighted_avg_trajectory)

    plt = car.draw_simulated_history(waypoint_index)

    weighted_avg_trajectories = car.simulated_history[-1]
    t_x = []
    t_y =[]
    for trajectory in weighted_avg_trajectories:
        t_x.append(trajectory[0])
        t_y.append(trajectory[1])
    print("Trajectory", t_x)

    plt.scatter(t_x, t_y, c='#D94496', label="Weighted Average Solution")
    fig = plt.gcf()
    plt.legend(  fancybox=True, shadow=True, loc="best")
    plt.savefig("weighted_avt_and_history.png")


    car.step(next_control_input)

    car.draw_history()

car.draw_history()







 




