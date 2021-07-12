from itertools import accumulate
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController

from constants import *
from globals import *

import numpy as np
from tqdm import trange


def run_simulation(number_of_steps):

    track = Track()
    car = Car(track)

    car_controller = CarController(track=track, predictor=CONTROLLER_PREDICTIOR, model_name=CONTROLLER_MODEL_NAME)


    for i in trange(number_of_steps):

        car_controller.set_state(car.state)
        next_control_sequence = car_controller.control_step()

        chosen_trajectory, cost = car_controller.simulate_trajectory( next_control_sequence )

        if DRAW_LIVE_ROLLOUTS:
            car_controller.draw_simulated_history(0, chosen_trajectory)

        # Do the first step of the best control sequence
        car.step(next_control_sequence[0])

    car.draw_history()
    car.save_history()
    print(car.lap_times)
    



if __name__ == "__main__":

    run_simulation(SIMULATION_LENGTH)

