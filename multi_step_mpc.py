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


for i in range(200):

    next_control_input = car_controller.control_step()
    chosen_trajectory = car_controller.simulate_trajectory([next_control_input] *10)

    car_controller.draw_simulated_history(waypoint_index,chosen_trajectory)
    car.step(next_control_input)
    car_controller.set_state(car.state)

    car.draw_history()

car.draw_history()







 




