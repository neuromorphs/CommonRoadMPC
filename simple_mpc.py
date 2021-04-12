from scipy.integrate import odeint

import numpy as np
from cycler import cycler
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
import matplotlib.pyplot as plt


def column(matrix, i):
    return [row[i] for row in matrix]

def func_KS(x, t, u, p):
    f = vehicle_dynamics_ks(x, u, p)
    return f

tStart = 0  # start time
tFinal = 2  # start time

# load vehicle parameters
p = parameters_vehicle1()

# initial state for simulation
delta0 = 0
vel0 = 2
Psi0 = 0
sy0 = 0
initialState = [0, sy0, delta0, vel0, Psi0]
x0_KS = init_ks(initialState)

t = np.arange(0, tFinal, 0.05)


sample_waypoints = [
    [5,5],
    [8,9],
    [15,15]
]

print(column(sample_waypoints,0))
u_seq = [
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
]


u_dist =  [
    [0.0, 0],
    [0.1, 0],
    [0.2, 0],
    [0.3, 0],
    [0.4, 0],
    [1, 0],
    [-0.1, 0],
    [-0.2, 0],
    [-0.3, 0],
    [-0.4, 0],


    # [0.0, 1],
    # [0.1, 1],
    # [0.2, 1],
    # [0.3, 1],
    # [0.4, 1],
    # [1, 1],
    # [-0.1, 1],
    # [-0.2, 1],
    # [-0.3, 1],
    # [-0.4, 1],


    # [0.0, -1],
    # [0.1, -1],
    # [0.2, -1],
    # [0.3, -1],
    # [0.4, -1],
    # [1, -1],
    # [-0.1, -1],
    # [-0.2, -1],
    # [-0.3, -1],
    # [-0.4, -1],

]


def cost_function(w_x, w_y, pos_x, pos_y):
    
    squared_distance = abs(w_x - pos_x) ** 2 + abs(w_y - pos_y) ** 2

    return squared_distance



x_real = x0_KS
next_waypoint_index = 0

for step in range(200):
    selected_index_u = 0
    lowest_cost = 100000000
    index = 0
    for u in u_dist:
        x_sim = odeint(func_KS, x_real, t, args=(u, p))
    
        s_x = x_sim[:,0]
        s_y = x_sim[:,1]

        
        
        for i in range(len(s_y)):
            waypoint = sample_waypoints[next_waypoint_index]
            cost = cost_function(waypoint[0],waypoint[1], s_x[i], s_y[i])
            
            if(cost < lowest_cost):
                # print("Cost",cost, index)
                lowest_cost = cost
                selected_index_u = index


            if(cost < 0.01):
                next_waypoint_index += 1
                print("NEXT WAYPOINT")
        


        index = index + 1

    plt.ylabel('some numbers')
    plt.savefig('plt.png')


    color_index = format(int(256/len(u_dist)) * index , '02x') 
    color = "#5500%s" % (color_index)
    plt.scatter(s_x,s_y, c=color)
    plt.scatter(column(sample_waypoints,0),column(sample_waypoints,1), c="#000000")


    print("Selected route")
    print(selected_index_u)
    t_real = np.arange(0, 0.1, 0.01)
    x_real = odeint(func_KS, x_real, t_real, args=(u_dist[selected_index_u], p))[-1]
    print(x_real)

    real_cost = cost_function(waypoint[0],waypoint[1], s_x[-1], s_y[-1])
    print("real_cost", real_cost)
   
        
 

