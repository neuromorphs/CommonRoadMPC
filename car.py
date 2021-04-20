
from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.init_mb import init_mb
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom


from scipy.integrate import odeint

class Car:
    def __init__(self, track):
        self.parameters = parameters_vehicle2()
        # self.state = init_ks([0, 0, 0, 20, 0])
        self.state = init_st([40, 10, 0, 10, 0, 0,0])
        # self.state = init_mb([419, 136, 0, 5, 0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0], self.parameters)
        self.time = 0 #TODO:
        self.tControlSequence = 0.2  # [s] How long is a control input applied
        self.tEulerStep = 0.01 # [s] One step of solving the ODEINT
        self.tHorizon = 4 # [s] How long should routes be calculated
        self.history = [] #Hostory of real car states
        self.simulated_history = [] #Hostory of simulated car states
        self.track = track #Track waypoints for drawing



    '''
    Dynamics of the car (Kinematic Single Track)
    '''
    def func_KS(self,x, t, u, p):
        # f = vehicle_dynamics_ks(x, u, p)
        f = vehicle_dynamics_st(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        return f

   
    '''
    Calculates the next system state due to a given state and control input
    For one time step
    @state: the system's state
    @control input 
    returns the simulated car state after tControlSequence [s]
    '''
    def simulate_step(self, state, control_input):
        t = np.arange(0, self.tControlSequence, self.tEulerStep) 
        x_next = odeint(self.func_KS, state, t, args=(control_input, self.parameters))
        return x_next[-1]

    '''
    Simulates a hypothetical trajectory of the car due to a list of control inputs
    @control_inputs: list of control inputs, which last 0.01s each
    '''
    def simulate_trajectory(self, control_inputs):

        #Generate 20 timesteps
        # t = np.arange(0, tHorizon, tControlSequence) 
        simulated_time = self.time
        simulated_state = self.state
        simulated_trajectory = []
        for control_input in control_inputs:
            simulated_state = self.simulate_step(simulated_state, control_input)
            simulated_trajectory.append(simulated_state)
        self.simulated_history.append(simulated_trajectory)
        return simulated_state


    def simulate_trajectory_distribution(self, control_inputs_distrubution):
        self.simulated_history = []
        for control_inputs in control_inputs_distrubution:
            step = self.simulate_trajectory(control_inputs)
        return self.simulated_history

    '''
    calculates the cost of a certain trajectory
    '''
    def cost_function(self, trajectory, waypoint_index):

        #Next 30 points of the track
        track = geom.LineString(self.track.waypoints[waypoint_index: waypoint_index + 30])

        cost = 0
        factor = 1
        for state in trajectory:
            simulated_position = geom.Point(state[0],state[1])
            distance_to_track = simulated_position.distance(track)
            
            # closest_waypoint_index = self.track.get_closest_index([state[0], state[1]])

            cost += factor * distance_to_track

        return cost


    '''
    Moves the "real" car one step due to a given control input
    '''
    def step(self, control_input):
        t = np.arange(0, self.tControlSequence, self.tEulerStep) 
        x_next = odeint(self.func_KS, self.state, t, args=(control_input, self.parameters))

        self.time += self.tControlSequence
        # print(self.time)
        self.state = x_next[-1]
        self.history.append(x_next)

    def draw_simulated_history(self, waypoint_index = 0):

        plt.clf()

        fig, position_ax = plt.subplots()
        # fig, axs = plt.subplots(2)
        # position_ax = axs[0]
        # velocity_ax = axs[1]

        plt.title("Model Predictive Path Integral")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

        s_x = []
        s_y = []
        costs = []
        for trajectory in self.simulated_history:
            cost = self.cost_function(trajectory, waypoint_index)
            for state in trajectory:
                if(state[0] > 1):
                    s_x.append(state[0])
                    s_y.append(state[1])
                    costs.append(cost)
                   

        index = 0
        trajectory_costs = position_ax.scatter(s_x,s_y, c=costs)
        colorbar = fig.colorbar(trajectory_costs)
        colorbar.set_label('Trajectory costs')

        #Draw waypoints
        w_x = self.track.waypoints_x[waypoint_index:waypoint_index+20]
        w_y = self.track.waypoints_y[waypoint_index:waypoint_index+20]
        position_ax.scatter(w_x,w_y, c ="#000000", label="Next waypoints")


        #Draw car position
        p_x = self.state[0]
        p_y = self.state[1]
        print("car state: ", self.state)
        position_ax.scatter(p_x,p_y, c ="#FF0000", label="Current car position")


        #Plot car state
        # velocity_ax.bar(range(5),  self.state[2:])
        # velocity_ax.set_xticklabels(["Delta", "Speed", "Phi", "Phi Dot", "Beta"])

        index += 1
        fig.legend()
        plt.savefig('sim_history.png')
        return plt

    def draw_history(self):
        plt.clf()
        plt.scatter(self.track.waypoints_x,self.track.waypoints_y)

        plt.ylabel('Position History')
        s_x = []
        s_y = []
        velocity = []
    
        for trajectory in self.history:
            for state in trajectory:
                s_x.append(state[0])
                s_y.append(state[1])
                velocity.append(state[3]) 

        index = 0
        color_index = format(int(255) , '02x') 
        color = "#5500%s" % (color_index)
        plt.scatter(s_x,s_y, c=velocity, cmap = cm.jet)
        index += 1

        plt.savefig('history.png')
    
