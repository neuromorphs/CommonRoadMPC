from itertools import accumulate
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController

from constants import *
from globals import *

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import trange


def run_simulation(number_of_steps):

    track = Track()
    car = Car(track)

    car_controller = CarController(track=track, predictor=CONTROLLER_PREDICTIOR, model_name=CONTROLLER_MODEL_NAME)

    costs = []
    lap_times = []

    for i in trange(number_of_steps):

        car_controller.set_state(car.state)
        next_control_sequence = car_controller.control_step()

        chosen_trajectory, cost = car_controller.simulate_trajectory( next_control_sequence )
        costs.append(cost)

        if DRAW_LIVE_ROLLOUTS:
            car_controller.draw_simulated_history(0, chosen_trajectory)

        if DRAW_LIVE_HISTORY:
            car.draw_history("live_history.png")

        # Do the first step of the best control sequence
        original_car_state = car.state
        car.step(next_control_sequence[0])

        # Check if lap completed
        # if(original_car_state[0] < track.waypoints_x[0] and car.state[0] >= track.waypoints_x[0]): #If passes finish line
        #     x_difference = car.state[0] - track.waypoints_x[0] #How much is car further than finish line
        #     x_speed = math.cos(original_car_state[4]) * original_car_state[3]   #X component of velocity
        #     time_difference = x_difference / x_speed    #how long ago did the car cross the finish line
        #     lap_time = car.time - time_difference # Accurate lap time is the current time of the car - the time passed since crossing the line

        #     print("completed track, lap_time: ", lap_time)
        #     lap_times.append(lap_time)
        #     exit()
        
            


    accumulated_cost = np.sum(costs)
    print("accumulated cost", accumulated_cost)
    print("lap_times", lap_times)
    car.draw_history()
    car.save_history()
    



if __name__ == "__main__":

    run_simulation(SIMULATION_LENGTH)

