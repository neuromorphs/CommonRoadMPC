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
    car_controller = CarController(car, predictor="nn", model_name="Dense-128-128-128-128-uniform-20")
    # car_controller = CarController(car, predictor="euler")

    for i in trange(number_of_steps):
        
        next_control_sequence = car_controller.control_step()
      
        # Do the first step of the best control sequence
        car.step(next_control_sequence[0])
        car_controller.set_state(car.state)

        if(DRAW_LIVE_ROLLOUTS):
            chosen_trajectory, cost = car_controller.simulate_trajectory(next_control_sequence)
            car_controller.draw_simulated_history(0, chosen_trajectory)
        
        if(DRAW_LIVE_HISTORY):
            car.draw_history("live_history.png")

    car.draw_history()
    car.save_history()



    """
    This is only for getting a feeling of how hight the different costs are... can be ignored
    """

    # Show cost distributions
    # collect_distance_costs = car_controller.collect_distance_costs
    # collect_speed_costs = car_controller.collect_speed_costs
    # collect_progress_costs = car_controller.collect_progress_costs

    # mean_distance_cost = np.array(collect_distance_costs).mean()
    # mean_speed_cost = np.array(collect_speed_costs).mean()
    # mean_progress_cost = np.array(collect_progress_costs).mean()

    # print("Mean distance cost: " ,mean_distance_cost)
    # print("Mean speed cost: " ,mean_speed_cost)
    # print("Mean progress cost: " ,mean_progress_cost)

    # plt.clf()
    # fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=False)
    # fig.set_size_inches(12, 16)

    # axs[0].set_title('Distance Cost Distribution')
    # axs[0].hist(collect_distance_costs, bins=20)

    # axs[1].set_title('Speed Cost Distribution')
    # axs[1].hist(collect_speed_costs, bins=20)

    # axs[2].set_title('Progress Cost Distribution')
    # axs[2].hist(collect_progress_costs, bins=20)

    # plt.savefig("cost_distributiuon.png")


if __name__ == "__main__":

    run_simulation(SIMULATION_LENGTH)

