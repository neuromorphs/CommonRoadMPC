import numpy as np
import math
from car import Car
from car_controller import CarController
from track import Track
from constants import *
import matplotlib.pyplot as plt


u1 = 10 * [[1,3]]
print( u1 )

track = Track()
car = Car(track)
car_controller = CarController(car)
waypoint_index = 1
last_control_input = [0,0]


for i in range(50):

    # input_samples = u_dist
    next_control_input = car_controller.control_step([0,0])

    weighted_avg_trajectory = car_controller.simulate_trajectory([next_control_input] *10)
    print("next ctr input",  next_control_input)
   
    print("weighted_avg_trajectory",weighted_avg_trajectory)

    plt = car_controller.draw_simulated_history(waypoint_index)

    weighted_avg_trajectories = car_controller.simulated_history[-1]
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

    print("car state", car.state)

    car.step(next_control_input)
    car_controller.set_state(car.state)

    car.draw_history()

car.draw_history()







 




