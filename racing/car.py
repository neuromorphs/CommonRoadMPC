
import sys
sys.path.insert(0, './commonroad-vehicle-models/PYTHON/')

from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.init_std import init_std
from vehiclemodels.init_mb import init_mb
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from globals import *
from util import *
import shapely.geometry as geom

import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime
from pathlib import Path

import csv  




'''
Represents the "real car", calculated by the l2race server
'''
class Car:
    def __init__(self,track = [], stay_on_track = True):

        initial_position = track.initial_position

        # self.parameters = parameters_vehicle1()
        self.parameters = parameters_vehicle2()
        self.state = init_st([initial_position[0], initial_position[1], 0, INITIAL_SPEED, 0, 0,0])
        self.stay_on_track = stay_on_track
        # self.state = init_std([initial_position[0], initial_position[1], 0, INITIAL_SPEED, 0, 0,0], p= self.parameters)
        # self.state = init_mb([419, 136, 0, INITIAL_SPEED, 0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0], self.parameters)
        
        self.time = 0 # The car's clock
        self.tControlSequence = T_CONTROL  # [s] How long is a control input applied
        self.tEulerStep = 0.01
        self.state_history = [] #Hostory of real car states
        self.control_history = [] #History of controls applied every timestep
        self.track = track #Track waypoints for drawing
        self.trackline  = geom.LineString(track.waypoints.copy())
        self.lap_times = []

        # For continuous experiment, we can start, where we last ended
        if CONTINUE_FROM_LAST_STATE:
            my_file = Path("racing/last_car_state.csv")
            if my_file.is_file():
                # file exists
                initial_state = np.loadtxt(open("racing/last_car_state.csv", "rb"), delimiter=",", skiprows=1)
                print("Initial_state" , initial_state)
                self.state = initial_state
            else:
                print("racing/last_car_state.csv not found, chosing hardcoded initial value")





    '''
    Dynamics of the real car
    '''
    def car_dynamics(self,x, t, u, p):
        # f = vehicle_dynamics_ks(x, u, p)
        f = vehicle_dynamics_st(x, u, p)
        # f = vehicle_dynamics_std(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        return f

   

    '''
    Moves the car one step due to a given control input
    '''
    def step(self, control_input):
        t = np.arange(0, self.tControlSequence, self.tEulerStep) 
        original_state = self.state.copy()
        
        # Next car position can be solved with euler or odeint
        # x_next = odeint(self.car_dynamics, self.state, t, args=(control_input, self.parameters))
        x_next = solveEuler(self.car_dynamics, self.state, t, args=(control_input, self.parameters))

        self.time += self.tControlSequence
        self.state = x_next[-1]
        self.state_history.append(x_next)
        self.control_history.append(control_input)

        # Check if car is still on the track
        if(self.stay_on_track):
            car_position = geom.Point(self.state[:2])
            distance_to_track = car_position.distance(self.trackline)
            
            if(distance_to_track > TRACK_WIDTH):
                np.savetxt("racing/last_car_state.csv", self.state, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")
                self.save_history()
                self.draw_history()
                print("Car left the track", distance_to_track)
                exit()
            
        # Check if lap completed
        if(original_state[0] < self.track.waypoints_x[0] and self.state[0] >= self.track.waypoints_x[0]): #If passes finish line
            x_difference = self.state[0] - self.track.waypoints_x[0] #How much is car further than finish line
            x_speed = math.cos(original_state[4]) * original_state[3]   #X component of velocity
            time_difference = x_difference / x_speed    #how long ago did the car cross the finish line
            lap_time = self.time - time_difference # Accurate lap time is the current time of the car - the time passed since crossing the line

            print("completed track, lap_time: ", lap_time)
            self.lap_times.append(lap_time)
            if EXIT_AFTER_ONE_LAP:
                exit()

        if DRAW_LIVE_HISTORY:
            self.draw_history("live_history.png")

        if ALWAYS_SAVE_LAST_STATE:
            np.savetxt("racing/last_car_state.csv", self.state, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")




    def save_history(self, filename = None):
        print("Saving history...")
        
        np.savetxt("racing/last_car_state.csv", self.state, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")
        
        control_history = np.array(self.control_history)
        np.savetxt("ExperimentRecordings/control_history.csv", control_history, delimiter=",", header="u1,u2")

        header=["time", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "u1", "u2"]
        state_history = np.array(self.state_history)
        state_history = state_history.reshape(state_history.shape[0] * state_history.shape[1],len(self.state))
        np.savetxt("ExperimentRecordings/car_state_history.csv", state_history, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")


        cut_state_history = state_history[0::20]
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
    
        file = 'ExperimentRecordings/history-{}.csv'.format(now)
        if filename is not None:
            file = filename

        with open(file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)


            #Meta data
            f.write("# Datetime: {} \n".format(now))
            f.write("# Track: {} \n".format( self.track.track_name))
            f.write("# Saving: {}\n".format("{}s".format(self.tControlSequence)))
            f.write("# Model: {}\n".format("MPPI, Ground truth model"))

            writer.writerow(header)
            time = 0
            for i in range(len(cut_state_history)):

                state_and_control = np.append(cut_state_history[i],control_history[i])               
                time_state_and_control = np.append(time, state_and_control)
                writer.writerow(time_state_and_control)
                time = round(time+self.tControlSequence, 2)


    

    """
    draws the history (position and speed) of the car into a plot
    """    
    def draw_history(self, filename = None):
        plt.clf()


        angles = np.absolute(self.track.AngleNextCheckpointRelative)
        # plt.scatter(self.track.waypoints_x,self.track.waypoints_y, c=angles)
        plt.scatter(self.track.waypoints_x,self.track.waypoints_y, color="#000")

        plt.ylabel('Position History')
        s_x = []
        s_y = []
        velocity = []
    
        for trajectory in self.state_history:
            for state in trajectory:
                s_x.append(state[0])
                s_y.append(state[1])
                velocity.append(state[3]) 

        index = 0
        scatter = plt.scatter(s_x,s_y, c=velocity, cmap = cm.jet)
        index += 1

        colorbar = plt.colorbar(scatter)
        colorbar.set_label('speed')

        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")

        file = 'ExperimentRecordings/history-{}.png'.format(now)
        if filename is not None:
            file = filename

        plt.savefig(file)
    
