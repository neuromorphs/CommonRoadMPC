
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
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import math
import csv  
from datetime import datetime
from globals import *

from scipy.integrate import odeint

CONTINUE_FROM_LAST_STATE = True

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    return history


PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"

'''
Represents the "real car", calculated by the l2race server
'''
class Car:
    def __init__(self,track = []):

        initial_position = track.initial_position

        self.parameters = parameters_vehicle2()
        # self.state = init_ks([0, 0, 0, 20, 0])
        self.state = init_st([initial_position[0], initial_position[1], 0, 7, 0, 0,0])
        # self.state = init_std([initial_position[0], initial_position[1], 0, 8, 0, 0,0], p= self.parameters)
        # self.state = init_mb([419, 136, 0, 5, 0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0], self.parameters)
        self.time = 0 #TODO:
        self.tControlSequence = T_CONTROL  # [s] How long is a control input applied
        self.tEulerStep = 0.01
        self.state_history = [] #Hostory of real car states
        self.control_history = [] #History of controls applied every timestep
        self.track = track #Track waypoints for drawing


        if CONTINUE_FROM_LAST_STATE:
            initial_state = np.loadtxt(open("car_state.csv", "rb"), delimiter=",", skiprows=1)
            print("Initial_state" , initial_state)
            self.state = initial_state





    '''
    Dynamics of the real car
    '''
    def func_KS(self,x, t, u, p):
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
        x_next = odeint(self.func_KS, self.state, t, args=(control_input, self.parameters))
        # x_next = solveEuler(self.func_KS, self.state, t, args=(control_input, self.parameters))

        self.time += self.tControlSequence
        # print(self.time)
        self.state = x_next[-1]
        self.state_history.append(x_next)
        self.control_history.append(control_input)



    def save_history(self, filename = None):
        print("Saving history...")
        
        np.savetxt("car_state.csv", self.state, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")

        
        control_history = np.array(self.control_history)
        # np.savetxt("ExperimentRecordings/control_history.csv", control_history, delimiter=",", header="u1,u2")

        header=["time", "dx","dy","x1", "x2", "x3", "x4", "x5", "x6", "x7", "u1", "u2"]
        state_history = np.array(self.state_history)
        state_history = state_history.reshape(state_history.shape[0] * state_history.shape[1],len(self.state))
        # np.savetxt("ExperimentRecordings/car_state_history.csv", state_history, delimiter=",", header="x1,x2,x3,x4,x5,x6,x7")


        cut_state_history = state_history[0::20]
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Today's date:", now)
    
        file = 'ExperimentRecordings/history-{}.csv'.format(now)
        if filename is not None:
            file = filename

        with open(file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)


            #Meta data
            f.write("# Datetime: {} \n".format(now))
            f.write("# Track: {} \n".format( self.track.track_name))
            f.write("# Saving: {}\n".format("0.2s"))
            f.write("# Model: {}\n".format("MPPI, Ground truth model"))

            writer.writerow(header)
            time = 0
            dx = 0
            dy = 0
            for i in range(len(cut_state_history)):
                
                if(i > 1):
                    dx = cut_state_history[i][0] - cut_state_history[i-1][0]
                    dy = cut_state_history[i][1] - cut_state_history[i-1][1]

                state_and_control = np.append(cut_state_history[i],control_history[i])
                state_and_control = np.append(dy, state_and_control)
                state_and_control = np.append(dx, state_and_control)
                time_state_and_control = np.append(time, state_and_control)
                writer.writerow(time_state_and_control)
                time = round(time+0.2, 2)




    """
    draws the history (position and speed) of the car into a plot
    """    
    def draw_history(self, filename = None):
        plt.clf()
        plt.scatter(self.track.waypoints_x,self.track.waypoints_y)

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
        color_index = format(int(255) , '02x') 
        color = "#5500%s" % (color_index)
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
    
