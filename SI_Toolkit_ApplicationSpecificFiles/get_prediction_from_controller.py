from tqdm import trange

import numpy as np
from racing.car import Car
from racing.track import Track
from mppi_mpc.car_controller import CarController
from constants import *
from mppi_mpc.car_controller import *

DEFAULT_SAMPLING_INTERVAL = 0.2  # s

def get_prediction_for_testing_gui_from_controller(a, dataset, dt_sampling, predictor, dt_sampling_by_dt_fine=1):

    # predictor => "euler", "odeint", "Dense-128-256-128_delta" or any other Neural network name
    model_name = None
    if(predictor != "euler" and predictor != "odeint"):
        model_name = predictor
        predictor = "nn"

    # region In either case testing is done on a data collected offline
    output_array = np.zeros(shape=(a.test_max_horizon+1, a.test_len, len(a.features)+1))

    print('Calculating predictions with {}'.format(predictor))

    track = Track()
    car = Car(track)
    car_controller = CarController(track, predictor=predictor, model_name = model_name)
    initial_state = dataset.loc[dataset.index[[0]], :].values[0][1:-2]
    car.state = initial_state
    car_controller.set_state(initial_state)

    for timestep in trange(a.test_len):
    
        # x1,x2,x3,x4,x5,x6,x7,u1,u2
        state_and_control = dataset.loc[dataset.index[[timestep]], :].values[0][1:]
        # x1,x2,x3,x4,x5,x6,x7
        car_state = state_and_control[:-2]
        # u1, u2
        next_control = state_and_control[-2:]

        car.state = car_state
        car_controller.set_state(car_state)
        
        simulated_state = car.state

        # Autoregressive prediction over the max_horizon for every test point
        for i in range(0, a.test_max_horizon):
            next_control = dataset.loc[dataset.index[[timestep + i ]], :].values[0][1:][-2:]
            simulated_state = car_controller.simulate_step(simulated_state, next_control)
            output_array[i+1, timestep, :-1] = simulated_state


    output_array = np.swapaxes(output_array, 0, 1)
    return output_array


