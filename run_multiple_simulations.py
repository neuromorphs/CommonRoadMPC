from multi_step_mpc import *


NUMBER_OF_SIMULATIONS = 2
NUMBER_OF_STEPS_PER_SIMULATION = 100


for i in range(NUMBER_OF_SIMULATIONS):
    print("Running Simulation {}".format(i))
    run_simulation(NUMBER_OF_STEPS_PER_SIMULATION)


print("Experiments done.")