from scipy.integrate import odeint
from scipy.integrate import solve_ivp  # Methods tried before, now not uesed anymore: RK23, RK45, LSODA, BDF, DOP853
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from matplotlib import cm


EULER_STEP = 0.05

def column(matrix, i):
        return [row[i] for row in matrix]
        

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    # print(history)
    return history


def model_func(t, y):
    f = func_KS(y, t, [0.1, 1], parameters_vehicle1())
    return f


def ivp(func, x0, t, atol = 1e-4, rtol =1e-2 ) :

   

    return solve_ivp( fun=model_func,
                t_span=[0, 1],
                method="LSODA",
                y0=x0,
                atol=atol,
                rtol=rtol)
        


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

t = np.arange(0, tFinal, EULER_STEP)
u = [0.1, 1]

number_of_test_runs = 1000

x0_KS = init_ks(initialState)
start = time.time()
for i in range(number_of_test_runs):
    x = []
    x0_KS[0] = x0_KS[0] + 0.1
    x0_KS[1] = x0_KS[1] + 0.1
    x0_KS[3] = x0_KS[3] + 0.001
    x = odeint(func_KS, x0_KS, t, args=(u, p))

end = time.time()
print("TIME FOR SCIPY ODE")
print(end - start)

plt.scatter(column(x, 0), column(x,1), c="#0000FF", label="ODEInt")



x0_KS = init_ks(initialState)
start = time.time()
for i in range(number_of_test_runs):
    x = []
    x0_KS[0] = x0_KS[0] + 0.1
    x0_KS[1] = x0_KS[1] + 0.1
    x0_KS[3] = x0_KS[3] + 0.001
    x = solveEuler(func_KS, x0_KS, t, args=(u, p))
end = time.time()

print("TIME FOR EULER")
print(end - start)

plt.scatter(column(x, 0), column(x,1), c="#00FF00", label="Euler")


plt.clf()


for atol in [1e-2, 2e-2, 3e-2, 4e-2, 5e-2,6e-2,7e-2,8e-2, 1e-1, 1.5e-1,1]:
    x0_KS = init_ks(initialState)
    start = time.time()
    for i in range(number_of_test_runs):
        x = []
        x0_KS[0] = x0_KS[0] + 0.1
        x0_KS[1] = x0_KS[1] + 0.1
        x0_KS[3] = x0_KS[3] + 0.001
        x = ivp(func_KS, x0_KS, t, atol = atol/100, rtol= atol)
    end = time.time()
    print("TIME FOR IVP with atol = ", atol)
    print(round(end - start, 4))
    x = x.y
    plt.scatter(x[0], x[1], c="#123123", label = atol)



plt.legend()
plt.savefig("T.png")


# print(x.y[-2])