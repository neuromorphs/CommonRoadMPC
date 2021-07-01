import sys

sys.path.insert(0, "./commonroad-vehicle-models/PYTHON/")

from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from constants import *
from globals import *
from util import *

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import shapely.geometry as geom
import matplotlib.pyplot as plt

import time
import math


"""
Can calculate the next control step due to the cost function
"""


class CarController:
    def __init__(self, track, predictor="euler", model_name=None):

        # Global
        self.tControlSequence = T_CONTROL  # [s] How long is a control input applied

        # Racing objects
        self.car_state = [1, 1, 1, 1, 1, 1, 1]
        self.track = track  # Track waypoints for drawing

        # Control with common-road models
        self.predictior = predictor
        # self.parameters = parameters_vehicle1()
        self.parameters = parameters_vehicle2()
        self.tEulerStep = T_EULER_STEP  # [s] One step of solving the ODEINT or EULER

        # Control with neural network
        if predictor == "nn":
            print("Setting up controller with neural network")
            if model_name is not None:
                from nn_prediction.prediction import NeuralNetworkPredictor

                self.nn_predictor = NeuralNetworkPredictor(model_name=model_name)

        # MPPI data
        self.simulated_history = []  # Hostory of simulated car states
        self.simulated_costs = []  # Hostory of simulated car states
        self.last_control_input = [0, 0]
        self.best_control_sequenct = []
        self.emergency_brake = False

        # Data collection
        self.collect_distance_costs = []
        self.collect_acceleration_costs = []

        # Get track ready
        self.update_trackline()

    def set_state(self, state):
        self.car_state = state
        # print("steering angle",self.car_state[4])
        self.car_state[4] = self.car_state[4] % 6.28
        if self.car_state[4] > 3.14: 
            self.car_state[4] = self.car_state[4] - 6.28

    def update_trackline(self):
        # only Next NUMBER_OF_NEXT_WAYPOINTS points of the track
        waypoint_modulus = self.track.waypoints.copy()
        waypoint_modulus.extend(waypoint_modulus[:NUMBER_OF_NEXT_WAYPOINTS])

        closest_to_car_position = self.track.get_closest_index(self.car_state[:2])
        first_waypoint = closest_to_car_position + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        last_waypoint = (
            closest_to_car_position
            + NUMBER_OF_NEXT_WAYPOINTS
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        )

        # print("first/last wp", first_waypoint, last_waypoint)
        waypoint_modulus = waypoint_modulus[first_waypoint:last_waypoint]

        self.trackline = geom.LineString(waypoint_modulus)

    """
    Dynamics of the simulated car from common road
    We can define different models here
    """

    def car_dynamics(self, x, t, u, p):
        # f = vehicle_dynamics_ks(x, u, p)
        f = vehicle_dynamics_st(x, u, p)
        # f = vehicle_dynamics_std(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        return f

    """
    Calculates the next system state due to a given state and control input
    For one time step
    @state: the system's state
    @control input 
    returns the simulated car state after tControlSequence [s]
    """

    def simulate_step(self, state, control_input):
        t = np.arange(0, self.tControlSequence, self.tEulerStep)

        if self.predictior == "euler":
            x_next = solveEuler(
                self.car_dynamics, state, t, args=(control_input, self.parameters)
            )[-1]
        elif self.predictior == "odeint":
            x_next = odeint(
                self.car_dynamics, state, t, args=(control_input, self.parameters)
            )[-1]
        else:
            x_next = self.nn_predictor.predict_next_state(state, control_input)[
                :,
            ]

        return x_next

    """
    Simulates a hypothetical trajectory of the car due to a list of control inputs
    @control_inputs: list of control inputs, which last 0.01s each
    """

    def simulate_trajectory(self, control_inputs):

        simulated_state = self.car_state
        simulated_trajectory = []
        cost = 0
        index = 0

        for control_input in control_inputs:
            if cost > MAX_COST:
                cost = MAX_COST
                # continue

            simulated_state = self.simulate_step(simulated_state, control_input)

            simulated_trajectory.append(simulated_state)

            index += 1

        cost = self.cost_function(simulated_trajectory)

        return simulated_trajectory, cost

    """
    Returns a trajectory for every control input in control_inputs_distribution
    """

    def simulate_trajectory_distribution(self, control_inputs_distrubution):

        self.simulated_history = []
        results = []
        costs = np.zeros(len(control_inputs_distrubution))

        i = 0
        for control_input in control_inputs_distrubution:
            simulated_trajectory, cost = self.simulate_trajectory(control_input)

            results.append(simulated_trajectory)
            costs[i] = cost
            i += 1

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    """
    Returns a trajectory for every control input in control_inputs_distribution
    """

    def simulate_trajectory_distribution_nn(self, control_inputs_distrubution):

        # start = time.time()
        control_inputs_distrubution = np.swapaxes(control_inputs_distrubution, 0, 1)
        # print(control_inputs_distrubution.shape)

        results = []

        states = np.array(len(control_inputs_distrubution[0]) * [self.car_state])
        # print(states.shape)
        for control_inputs in control_inputs_distrubution:

            # print(control_inputs.shape)
            states = self.nn_predictor.predict_multiple_states(states, control_inputs)
            results.append(states)

        # end = time.time()
        # print("TIME FOR all {} Trajectories with nn".format(NUMBER_OF_TRAJECTORIES))
        # print(end - start)

        # start = time.time()

        results = np.array(results)
        results = np.swapaxes(results, 0, 1)

        costs = []
        for result in results:
            cost = self.cost_function(result)
            costs.append(cost)

        # end = time.time()
        # print("TIME FOR evaluating all Trajectories with cost function")
        # print(end - start)

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    def static_control_inputs(self):
        control_inputs = [
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, 0]],  # No input
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.2, 0]],  # little left
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[-1, 0]],  # hard left
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.2, 0]],  # little right
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[1, 0]],  # hard right
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, -1]],  # brake
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[0, 1]],  # accelerate
            # NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.4, -1]],  # brake and left
            # NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.4, -1]],  # brake and right
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.4, 1]],  # accelerate and left
            NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.4, 1]],  # accelerate and right
        ]

        return control_inputs
        exit()

    """
    Returns a gaussian distribution around the last control input
    """

    def sample_control_inputs(self):

        steering = np.random.uniform(
            -1, 1, NUMBER_OF_INITIAL_TRAJECTORIES * NUMBER_OF_STEPS_PER_TRAJECTORY
        )
        acceleration = np.random.uniform(
            -0.5, 0.5, NUMBER_OF_INITIAL_TRAJECTORIES * NUMBER_OF_STEPS_PER_TRAJECTORY
        )
        # if(self.emergency_brake):
        # else:
        #     acceleration = np.random.uniform(-1,0, NUMBER_OF_INITIAL_TRAJECTORIES *NUMBER_OF_STEPS_PER_TRAJECTORY)

        control_input_sequences = np.column_stack((steering, acceleration))
        control_input_sequences = np.reshape(
            control_input_sequences,
            (NUMBER_OF_INITIAL_TRAJECTORIES, NUMBER_OF_STEPS_PER_TRAJECTORY, 2),
        )
        # print("control sequences",control_input_sequences)
        return control_input_sequences

    def sample_control_inputs_similar_to_last(self, last_control_sequence):
        # return self.sample_control_inputs()
        # return self.static_control_inputs()
        # In case we always want the initial variance
        if self.emergency_brake:
            return self.sample_control_inputs()

        # Not initialized
        if len(last_control_sequence) == 0:
            return self.sample_control_inputs()

        # Delete the first step of the last control sequence because it is already done
        # To keep the length of the control sequence add one to the end
        last_control_sequence = last_control_sequence[1:]
        last_control_sequence = np.append(last_control_sequence, [0, 0]).reshape(
            NUMBER_OF_STEPS_PER_TRAJECTORY, 2
        )

        last_steerings = last_control_sequence[:, 0]
        last_accelerations = last_control_sequence[:, 1]

        control_input_sequences = np.zeros(
            [NUMBER_OF_TRAJECTORIES, NUMBER_OF_STEPS_PER_TRAJECTORY, 2]
        )

        for i in range(NUMBER_OF_TRAJECTORIES):

            steering_noise = np.random.normal(0, 0.5, NUMBER_OF_STEPS_PER_TRAJECTORY)
            acceleration_noise = np.random.normal(0, 0.4, NUMBER_OF_STEPS_PER_TRAJECTORY)

            next_steerings = last_steerings + steering_noise
            next_accelerations = last_accelerations + acceleration_noise

            # Filter for smoother control
            # next_steerings = signal.medfilt(next_steerings, 3)
            # next_accelerations = signal.medfilt(next_accelerations, 3)

            next_control_inputs = np.vstack((next_steerings, next_accelerations)).T
            control_input_sequences[i] = next_control_inputs

        return control_input_sequences

    """
    calculates the cost of a trajectory
    """

    def cost_function(self, trajectory):

        distance_cost = 0
        acceleration_cost = 0

        distance_cost_weight = 1
        terminal_cost_weight = 0
        acceleration_cost_weight = 1 * (
            MAX_SPEED - self.car_state[3]
        )  # Adaptive speed cost weight
        if self.emergency_brake:
            acceleration_cost_weight = 0
        number_of_critical_states = 10
        number_of_states = len(trajectory)
        index = 0

        for state in trajectory:
            discount = (number_of_states - 0.5 * index) / number_of_states

            simulated_position = geom.Point(state[0], state[1])
            distance_to_track = simulated_position.distance(self.trackline)
            # distance_to_track = distance_to_track ** 2
            acceleration = state[3] - self.car_state[3]

            acceleration_cost += abs((acceleration + 1) / 2)
            distance_cost += abs(discount * distance_to_track)

            # Don't leave track!
            if distance_to_track > TRACK_WIDTH:
                # if(index < number_of_critical_states):
                distance_cost += 1000
            index += 1

        acceleration_cost = 1 / acceleration_cost

        # Terminal Cost
        terminal_state = trajectory[-1]
        terminal_cost = self.terminal_cost(terminal_state)

        self.collect_distance_costs.append(distance_cost)
        self.collect_acceleration_costs.append(acceleration_cost)

        cost = (
            distance_cost_weight * distance_cost
            + acceleration_cost_weight * acceleration_cost
            + terminal_cost_weight * terminal_cost
        )
        if False:
            print("acceleration_cost", acceleration_cost_weight * acceleration_cost)
            print("distance_cost", distance_cost_weight * distance_cost)
            print("terminal_cost", terminal_cost_weight * terminal_cost)
        # print("Cost", cost)

        return cost

    def terminal_cost(self, terminal_state):
        terminal_cost = 0
        # print("terminal_state",terminal_state)
        terminal_position = geom.Point(terminal_state[0], terminal_state[1])
        terminal_distance_to_track = terminal_position.distance(self.trackline)

        # original_waypoint_index = self.track.get_closest_index(self.car_state[:2])
        # terminal_waypoint_index = self.track.get_closest_index(terminal_state[:2])
        # progress =terminal_waypoint_index - original_waypoint_index
        # progress = math.sqrt(squared_distance(terminal_state[:2], terminal_state[:2]))

        braking = max(self.car_state[3] - terminal_state[3], 0)
        # print("braking", braking)
        # print("progress", progress)
        # if(progress ==  0):
        #     progress = 1

        terminal_cost += 3 * abs(terminal_distance_to_track)
        terminal_cost += 0.15 * abs(15 - terminal_state[3])

        if terminal_state[3] < 5:
            terminal_cost += 3 * abs(5 - terminal_state[3])
        # terminal_cost += 100 * braking
        # print("terminal_cost", terminal_cost)
        return max(0, terminal_cost)

    """
    Does one step of control and returns the best control input for the next time step
    """

    def control_step(self):
        # print("Emergency brake", self.emergency_brake)
        self.update_trackline()
        self.last_control_input[0] = min(self.last_control_input[0], 1)
        self.last_control_input[1] = min(self.last_control_input[1], 0.5)

        control_sequences = self.sample_control_inputs_similar_to_last(
            self.best_control_sequenct
        )
        # control_sequences = u_dist #those are the handcrafted inputs

        if self.predictior == "nn":
            simulated_history, costs = self.simulate_trajectory_distribution_nn(
                control_sequences
            )
        else:
            simulated_history, costs = self.simulate_trajectory_distribution(
                control_sequences
            )

        lowest_cost = 100000
        best_index = 0

        weights = np.zeros(len(control_sequences))

        for i in range(len(control_sequences)):
            cost = costs[i]
            # find best
            if cost < lowest_cost:
                best_index = i
                lowest_cost = cost

            if cost < MAX_COST:
                weight = math.exp((-1 / INVERSE_TEMP) * cost)
            else:
                weight = 0
            weights[i] = weight

        best_conrol_sequence = control_sequences[best_index]

        # Finding weighted avg input
        if weights.max() != 0:
            self.emergency_brake = False
            next_control_sequence = np.average(
                control_sequences, axis=0, weights=weights
            )
        else:
            # self.emergency_brake = True
            next_control_sequence = best_conrol_sequence

        next_control_sequence = best_conrol_sequence
        self.best_control_sequenct = next_control_sequence
        return next_control_sequence

    """
    draws the simulated history (position and speed) of the car into a plot for a trajectory distribution resp. the history of all trajectory distributions
    """

    def draw_simulated_history(self, waypoint_index=0, chosen_trajectory=[]):

        plt.clf()

        fig, position_ax = plt.subplots()

        plt.title("Static Control")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

        s_x = []
        s_y = []
        costs = []
        i = 0
        ind = 0
        indices = []
        for trajectory in self.simulated_history:
            cost = self.simulated_costs[i]
            if cost < MAX_COST:
                for state in trajectory:
                    if state[0] > 1:
                        s_x.append(state[0])
                        s_y.append(state[1])
                        costs.append(cost)
                        indices.append(cost)
                        ind += 1
            i += 1

        trajectory_costs = position_ax.scatter(s_x, s_y, c=indices)
        colorbar = fig.colorbar(trajectory_costs)
        colorbar.set_label("Trajectory costs")

        # Draw car position
        p_x = self.car_state[0]
        p_y = self.car_state[1]
        position_ax.scatter(p_x, p_y, c="#FF0000", label="Current car position")

        # Draw waypoints
        waypoint_index = self.track.get_closest_index(self.car_state[:2])

        waypoints = np.array(self.track.waypoints)
        w_x = waypoints[
            waypoint_index
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS : waypoint_index
            + NUMBER_OF_NEXT_WAYPOINTS
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
            0,
        ]
        w_y = waypoints[
            waypoint_index
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS : waypoint_index
            + NUMBER_OF_NEXT_WAYPOINTS
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
            1,
        ]
        position_ax.scatter(w_x, w_y, c="#000000", label="Next waypoints")

        # Plot Chosen Trajectory
        t_x = []
        t_y = []
        for state in chosen_trajectory:
            t_x.append(state[0])
            t_y.append(state[1])

        plt.scatter(t_x, t_y, c="#D94496", label="Chosen control")
        plt.legend(fancybox=True, shadow=True, loc="best")

        plt.savefig("live_rollouts.png")
        return plt

