##########################################
################## Global ################
##########################################
T_CONTROL = 0.2
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"


##########################################
################### Car ##################
##########################################

CONTINUE_FROM_LAST_STATE = True


##########################################
############# Car Controller #############
##########################################

#Covariance Matrix for the input distribution
INITIAL_COVARIANCE = [[0.1, 0], [0, 0.0]] 

#Covariance matrix for every next-step in the trajectory prediction
STEP_COVARIANCE = [[0.01, 0],[0,0.1]]
NUMBER_OF_STEPS_PER_TRAJECTORY = 20

# Must be between 0 and 1 and defines how close to a random walk the trajectories are
# 0: The trajectories are calculated by a completely random control with mean = 0 and variance = step_covariance0 
# 1: The trajectories are a random walk with mean = last control and variance = step_covariance
RANDOM_WALK = 0.3

#Number of trajectories that are calculated for every step
NUMBER_OF_INITIAL_TRAJECTORIES = 50

#Covariance for calculating simmilar controls to the last one
STRATEGY_COVARIANCE=[[0.02, 0],[0, 0.2]]   
NUMBER_OF_TRAJECTORIES = 50

NUMBER_OF_NEXT_WAYPOINTS = 20

#With of the race track (ToDo) 
DIST_TOLLERANCE = 4 
INVERSE_TEMP = 5
DRAW_CHOSEN_SEQUENCE = True
DRAW_TRAJECTORIES = True

MAX_COST = 1000


##########################################
############## NN Predictor ##############
##########################################

TRAIN_NEW_MODEL = False
MODEL_NAME = "Dense-128-256-256-128_augumented"

NUMBER_OF_EPOCHS = 3
BATCH_SIZE = 64