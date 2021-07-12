from scipy.integrate import odeint

import numpy as np
from cycler import cycler
from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
import matplotlib.pyplot as plt

from nn_prediction.keras_test import predict_next_state




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
    f = vehicle_dynamics_st(x, u, p)
    return f

tStart = 0  # start time
tFinal = 1  # start time

# load vehicle parameters
p = parameters_vehicle1()

# initial state for simulation
sx0 = 40
sy0 = 15
delta0 = 0
vel0 = 8
Psi0 = 1
initialState = [43.23173334436362,13.60142774208227,0.004662434370796332,7.0182585218596,0.000861189740078239,0.010506119824329887,0.0018568201206292032]
x0_KS = init_st(initialState)

t = np.arange(0, tFinal, 0.05)

u_seq = [
    [0.03228715794315807,-0.21562029606707006],
    [0.053498444071493984,-0.17183121112629401],
    [0.00904375556023907,0.030898644749977545],
    [-0.05045244035950034,-0.08589216564103928],
    [0.10415926464786728,0.017182947624049986],
    [0.11549411802308145,-0.030044060640355672],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],
    [0.0, 0],





]

#Euler
index = 0
x = solveEuler(func_KS, x0_KS, [0.02], args=([0,0], p))
for u in u_seq:
    x = solveEuler(func_KS, x[-1], t, args=(u, p))

    s_x = x[:,0]
    s_y = x[:,1]
    # print(s_x)
    # 
    color_index = format(int(256/len(u_seq)) * index , '02x') 
    color = "#5500%s" % (color_index)
    plt.scatter(s_x,s_y, c="#FF0000")
    index = index + 1



x0_KS = init_st(initialState)
x = np.array(x0_KS)
#NN_prediction
index = 0


for u in u_seq:
    u = np.array(u)
    # print("u1", u)
    x = predict_next_state(x,u)
    # print("x1", x)


    s_x = x[0]
    s_y = x[1]
    # print(s_x)
    # 
    color_index = format(int(256/len(u_seq)) * index , '02x') 
    color = "#5500%s" % (color_index)
    plt.scatter(s_x,s_y, c="#00FF00")
    index = index + 1



# plt.plot(s_x)
# plt.plot(s_y)

plt.ylabel('some numbers')
plt.savefig('plt.png')