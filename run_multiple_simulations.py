from multi_step_mpc import *


NUMBER_OF_SIMULATIONS = 10
NUMBER_OF_STEPS_PER_SIMULATION = 1000


for i in range(NUMBER_OF_SIMULATIONS):
    print("Running Simulation {}".format(i))
    run_simulation(NUMBER_OF_STEPS_PER_SIMULATION)


print("Experiments done.")