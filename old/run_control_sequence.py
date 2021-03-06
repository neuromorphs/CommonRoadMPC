# This file is for simulating a saved control sequence

import numpy as np
import math
from racing.car import Car
from mppi_mpc.car_controller import CarController
from racing.track import Track
from constants import *
import matplotlib.pyplot as plt
from tqdm import trange

FILE_NAME = "ControlSequences/control_history.csv"

control_sequence = np.loadtxt(open(FILE_NAME, "rb"), delimiter=",", skiprows=1)

track =  Track()
car = Car(track)

for control_input in control_sequence:
    car.step(control_input)
car.draw_history("history.png")
car.save_history("history.csv")
