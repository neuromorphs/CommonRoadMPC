from scipy.integrate import odeint
from scipy.integrate import solve_ivp  # Methods tried before, now not uesed anymore: RK23, RK45, LSODA, BDF, DOP853

import numpy as np
import time
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks


def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    # print(history)
    return x


def model_func(t, y):
    f = func_KS(y, t, [0, 1], parameters_vehicle1())
    return f


def ivp(func, x0, t) :

    RTOL = 1e-2 #Relative tolerance for IVP solver
    ATOL = 1e-4 #Absolute tolerance for IVP solver

    return solve_ivp( fun=model_func,
                t_span=[0, 1],
                method="LSODA",
                y0=x0,
                atol=ATOL,
                rtol=RTOL)
        


def func_KS(x, t, u, p):
    f = vehicle_dynamics_ks(x, u, p)
    return f

tStart = 0  # start time
tFinal = 1  # start time

# load vehicle parameters
p = parameters_vehicle1()

# initial state for simulation
delta0 = 0
vel0 = 2
Psi0 = 0
sy0 = 0
initialState = [0, sy0, delta0, vel0, Psi0]
x0_KS = init_ks(initialState)

t = np.arange(0, tFinal, 0.01)
u = [0, 1]

start = time.time()
x = odeint(func_KS, x0_KS, t, args=(u, p))
end = time.time()
print("TIME FOR SCIPY ODE")
print(end - start)

# print(x)

start = time.time()
x = solveEuler(func_KS, x0_KS, t, args=(u, p))
end = time.time()
print("TIME FOR EULER")
print(end - start)


start = time.time()
x = ivp(func_KS, x0_KS, t)
end = time.time()
print("TIME FOR IVP")
print(end - start)



print(x.y[-2])