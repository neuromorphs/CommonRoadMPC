import numpy as np
import math
import pandas as pd
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController
from constants import *
import matplotlib.pyplot as plt


track = Track()
car = Car(track)


def generate_random_walk(steps):

    print("test")
    track = Track()
    car = Car(track)
    car.state = [0,0,0,5,0,0,0]

    mu, sigma = 0, 0.01 # mean and standard deviation
    steering = np.random.normal(mu, sigma, steps)
    mu, sigma = 0, 0.00 # mean and standard deviation
    acceleration = np.random.normal(mu, sigma, steps)

    control_input = [0,0]
    for i in range(steps): 
        control_input[0] = max(min(control_input[0] + steering[i], 1), -1)
        control_input[1] = min(control_input[1] + acceleration[i], 0.4)

        print(control_input)
        car.step(control_input)
        # car.step([0.1, 0.1])
        print(i)

    car.draw_history("test.png")
    car.save_history()


def generate_random():

    nn_inputs = []
    nn_results = []

    track = Track()
    car = Car(track)
    number_of_initial_states = 100
    number_of_steps_per_trajectory = 1000

    car.state = [0,0,0,0,0,0,0]

    #Position
    mu, sigma = 100, 50 # mean and standard deviation
    s_x_dist = np.random.normal(mu, sigma, number_of_initial_states) 
    s_y_dist = np.random.normal(mu, sigma, number_of_initial_states)

    #delta
    mu, sigma = 1, 0.2 # mean and standard deviation
    delta_dist = np.random.normal(mu, sigma, number_of_initial_states) 

    #velocity
    mu, sigma = 10, 5 # mean and standard deviation
    v_dist = np.random.normal(mu, sigma, number_of_initial_states) 

    #epsylon
    mu, sigma = 1, 0.1 # mean and standard deviation
    epsylon_dist = np.random.normal(mu, sigma, number_of_initial_states)
    epsylon_dot_dist = np.random.normal(mu, sigma, number_of_initial_states)

    #beta
    mu, sigma = 1, 0.1 # mean and standard deviation
    beta_dist = np.random.normal(mu, sigma, number_of_initial_states)
    for i in range(number_of_initial_states): 
        #Set random initial state
        car.state = [
            s_x_dist[i],
            s_y_dist[i],
            delta_dist[i],
            v_dist[i],
            epsylon_dist[i],
            epsylon_dot_dist[i],
            beta_dist[i],
        ]

        mu, sigma = 0, 0.5 # mean and standard deviation
        u0_dist = np.random.normal(mu, sigma, number_of_steps_per_trajectory)
        mu, sigma = 0, 0.1 # mean and standard deviation
        u1_dist = np.random.normal(mu, sigma, number_of_steps_per_trajectory)

        for j in range(number_of_steps_per_trajectory):
            control = [u0_dist[j], u1_dist[j]]
            nn_input = np.append(car.state.copy(), control)
            nn_inputs.append(nn_input)

            car.step(control)

            nn_result = car.state
            nn_results.append(car.state)


    df = pd.DataFrame(nn_inputs)
    df.to_csv('NeuralNetworkPredictor/data/train/random_inputs.csv', index=False, float_format='%.3f')
    df = pd.DataFrame(nn_results)
    df.to_csv('NeuralNetworkPredictor/data/train/random_results.csv', index=False, float_format='%.3f')


def generate_every_possibility():
    car.state = [0,0,0,0,0,0,0]

    nn_inputs = []
    nn_results = []

    for i in range(0, 10):
        car.state = [i,0,0,0,0,0,0]
        for j in range(100):
            for k in range(100):
                control = [0.01 * j, 0.01 * k]
                #The NN input is an array containing the current state and the control
                nn_input = np.append(car.state.copy(), control)
                nn_inputs.append(nn_input)

                car.step(control)

                #The nn output is the next state
                nn_result = car.state
                nn_results.append(car.state)


    df = pd.DataFrame(nn_inputs)
    df.to_csv('NeuralNetworkPredictor/data/train/inputs.csv', index=False, float_format='%.3f')
    df = pd.DataFrame(nn_results)
    df.to_csv('NeuralNetworkPredictor/data/train/results.csv', index=False, float_format='%.3f')



    #Test Data

    for i in range(0, 10):
        car.state = [i,0,0,0,0,0,0]
        for j in range(5):
            for k in range(5):
                control = [0.01 * j, 0.01 * k]
                #The NN input is an array containing the current state and the control
                nn_input = np.append(car.state.copy(), control)
                nn_inputs.append(nn_input)

                car.step(control)

                #The nn output is the next state
                nn_result = car.state
                nn_results.append(car.state)
    df = pd.DataFrame(nn_inputs)
    df.to_csv('NeuralNetworkPredictor/data/test/inputs.csv', index=False, float_format='%.3f')
    df = pd.DataFrame(nn_results)
    df.to_csv('NeuralNetworkPredictor/data/test/results.csv', index=False, float_format='%.3f')




generate_random_walk(1000)