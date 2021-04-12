from scipy.integrate import odeint

import numpy as np
from cycler import cycler
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
import matplotlib.pyplot as plt



def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    print(history)
    return history
    


def func_KS(x, t, u, p):
    f = vehicle_dynamics_ks(x, u, p)
    return f

tStart = 0  # start time
tFinal = 1  # start time

# load vehicle parameters
p = parameters_vehicle1()

# initial state for simulation
delta0 = 0
vel0 = 5
Psi0 = 0
sy0 = 0
initialState = [0, sy0, delta0, vel0, Psi0]
x0_KS = init_ks(initialState)

t = np.arange(0, tFinal, 0.05)

u_seq = [
    [0.0, 1],
    [0.0, 1],
    [0.0, 1],
    [1 , 0],
    [-1 , 0],
    [-1 , 0],
    [-1 , 0],
    [1 , 0],
    [1 , 0],
    [1 , 0],
    [1 , 0],


]

index = 0
x = solveEuler(func_KS, x0_KS, t, args=([0,0], p))
for u in u_seq:
    x = solveEuler(func_KS, x[-1], t, args=(u, p))

    s_x = x[:,0]
    s_y = x[:,1]
    # print(s_x)
    # 
    color_index = format(int(256/len(u_seq)) * index , '02x') 
    color = "#5500%s" % (color_index)
    plt.scatter(s_x,s_y, c=color)
    index = index + 1


# plt.plot(s_x)
# plt.plot(s_y)

plt.ylabel('some numbers')
plt.savefig('plt.png')