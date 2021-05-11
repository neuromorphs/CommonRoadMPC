import numpy as np
import math
from car import Car
from car_controller import CarController
from track import Track
from constants import *
import matplotlib.pyplot as plt


track = Track()
car = Car(track)
car_controller = CarController(car)
waypoint_index = 1
last_control_input = [0,0]


for i in range(150):

    next_control_input = car_controller.control_step()
    chosen_trajectory = car_controller.simulate_trajectory([next_control_input] *20)

    # print("Next control", next_control_input)

    # Comment this out for speedup
    # car_controller.draw_simulated_history(waypoint_index,chosen_trajectory)

    car.step(next_control_input)
    car_controller.set_state(car.state)

    car.draw_history()

car.draw_history()



'''
This is only for getting a feeling of how hight the different costs are... can be ignored
'''

#Show cost distributions
collect_distance_costs = car_controller.collect_distance_costs
collect_speed_costs = car_controller.collect_speed_costs
collect_progress_costs = car_controller.collect_progress_costs

mean_distance_cost = np.array(collect_distance_costs).mean()
mean_speed_cost = np.array(collect_speed_costs).mean()
mean_progress_cost = np.array(collect_progress_costs).mean()

print("Mean distance cost: " ,mean_distance_cost)
print("Mean speed cost: " ,mean_speed_cost)
print("Mean progress cost: " ,mean_progress_cost)

plt.clf()
fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=False)
fig.set_size_inches(12, 16)

axs[0].set_title('Distance Cost Distribution')
axs[0].hist(collect_distance_costs, bins=20)

axs[1].set_title('Speed Cost Distribution')
axs[1].hist(collect_speed_costs, bins=20)

axs[2].set_title('Progress Cost Distribution')
axs[2].hist(collect_progress_costs, bins=20)


plt.savefig("cost_distributiuon.png")






 




