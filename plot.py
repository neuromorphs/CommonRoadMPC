from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import math
from numpy import genfromtxt


control_history = genfromtxt('ExperimentRecordings/control_history.csv', delimiter=',')
print(control_history.shape)

fig, axs = plt.subplots(2)
fig.suptitle('Control History')
axs[0].title.set_text('Steering')
axs[0].plot(control_history[:,0])
axs[1].title.set_text('Thrust')
axs[1].plot(control_history[:,1])

plt.savefig("control_histry.png")
plt.clf()

state_history = genfromtxt('ExperimentRecordings/car_state_history.csv', delimiter=',')
print(state_history.shape)

fig, axs = plt.subplots(7)
fig.suptitle('Control History')
axs[0].title.set_text('x')
axs[0].plot(state_history[:,0])
axs[1].title.set_text('y')
axs[1].plot(state_history[:,1])
axs[2].title.set_text('Car angle')
axs[2].plot(state_history[:,2])
axs[3].title.set_text('Speed')
axs[3].plot(state_history[:,3])
axs[4].title.set_text('Steering angle')
axs[4].plot(state_history[:,4])
axs[5].title.set_text('Steering rate')
axs[5].plot(state_history[:,5])
axs[6].title.set_text('Slip angle')
axs[6].plot(state_history[:,6])


plt.savefig("car_state_histry.png")