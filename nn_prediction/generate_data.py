import numpy as np
import math
import csv
import pandas as pd
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController
from constants import *
from globals import *
import matplotlib.pyplot as plt
from tqdm import trange
import os


track = Track()
car = Car(track, stay_on_track=False)

pi = math.pi


def generate_distribution( number_of_initial_states = 500, number_of_trajectories = 500,number_of_steps_per_trajectory = 10):
    '''
    Generate a distribution of artificial training datapoints
    and saves it to a .csv file in nn_prediction/training/data
    @param number_of_initial_states{int} The number of random initial states
    @param number_of_trajectories{int} The number of random trajectories that are simulated for every initial state
    @param number_of_steps_per_trajectory{int} the length of the simulated trajectories
    '''

    output_file = "nn_prediction/training/data/{}".format(DATA_GENERATION_FILE)
    if os.path.exists(output_file):
        os.remove(output_file)



    car.state = [0,0,0,0,0,0,0]

    #Position => invariant since we are only interrested in delta
    x_dist = np.zeros(number_of_initial_states)
    y_dist = np.zeros(number_of_initial_states)

    #Steering of front wheels
    delta_dist = np.random.uniform(-1,1, number_of_initial_states) 

    #velocity in face direction
    v_dist = np.random.uniform(10, 30, number_of_initial_states) 

    #Yaw Angle
    yaw_dist = np.random.uniform(-pi, pi, number_of_initial_states)

    #Yaw rate
    yaw_rate_dist = np.random.uniform(-0, 0, number_of_initial_states)

    #Slip angle
    slip_angle_dist = np.random.uniform(-0, 0, number_of_initial_states)


    states = np.column_stack((x_dist, y_dist, delta_dist, v_dist, yaw_dist, yaw_rate_dist, slip_angle_dist))

    print(states.shape)
    print(states[0])

    for i in trange(len(states)):

        state = states[i]
        mu, sigma = 0, 0.4 # mean and standard deviation
        u0_dist = np.random.normal(mu, sigma , number_of_trajectories)
        mu, sigma = 0, 0.5 # mean and standard deviation
        u1_dist = np.random.normal(mu, sigma , number_of_trajectories)

        # u0_dist = np.random.uniform(-1, 1, number_of_trajectories)
        # u1_dist = np.random.uniform(-1, 1, number_of_trajectories)

        controls = np.column_stack((u0_dist, u1_dist))
        results = []

        car.state_history = []
        car.control_history =[]

        for j in range(number_of_trajectories):
            car.state = state   

            for k in range (number_of_steps_per_trajectory):
                control = controls[j]


                # if car.state[4] > 7:
                #     control[0] = -control[0]

                # if car.state[4] > -7:
                #      control[0] = -control[0]

                state_and_control = np.append(car.state,control)
                car.step(control)
                state_and_control_and_future_state = np.append(state_and_control,car.state)
                results.append(state_and_control_and_future_state)



            car.draw_history("data_generation.png")

        with open(output_file, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            time = 0
            for result in results:
                
                time_state_and_control = np.append(time, result)

                #time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
                writer.writerow(time_state_and_control)
                time = round(time+0.2, 2)



if __name__ == "__main__":

    generate_distribution( 100, 100, 60)

