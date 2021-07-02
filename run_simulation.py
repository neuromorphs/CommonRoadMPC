#THIS IS AN EXAMPLE FOR HOW TO RUN A NEW CONTROLLER INSIDE THIS FRAMEWORK
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController

from constants import *
from globals import *

import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange



def run_simulation(number_of_steps):

    track = Track()
    car = Car(track)

    # Init car controller
    # car_controller = CarController()

    for i in trange(number_of_steps):

        # Update Cat state for controller
        # car_controller.set_state(car.state)

        #######################################
        ###### IMPLEMENT CONTROLLER HERE ######
        #######################################

        # Find out the next control
        # control = car_controller.get_next_control()
        control = [0,0] #Steering and acceleration between -1,-1 and 1,1

        car.step(control)
        car.draw_history("live_history.png")

    car.draw_history()
    car.save_history()

  


if __name__ == "__main__":

    run_simulation(SIMULATION_LENGTH)

