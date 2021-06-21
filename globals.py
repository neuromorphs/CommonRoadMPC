##########################################
################## Global ################
##########################################
T_CONTROL = 0.2
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"


##########################################
################### Car ##################
##########################################

CONTINUE_FROM_LAST_STATE = False


##########################################
############# Car Controller #############
##########################################

# Covariance Matrix for the input distribution
INITIAL_COVARIANCE = [[0.2, 0], [0, 0.2]]

# Covariance matrix for every next-step in the trajectory prediction
STEP_COVARIANCE = [[0.0, 0], [0, 0.0]]
NUMBER_OF_STEPS_PER_TRAJECTORY = 15

# Must be between 0 and 1 and defines how close to a random walk the trajectories are
# 0: The trajectories are calculated by a completely random control with mean = 0 and variance = step_covariance0
# 1: The trajectories are a random walk with mean = last control and variance = step_covariance
RANDOM_WALK = 0.0

# Number of trajectories that are calculated for every step
NUMBER_OF_INITIAL_TRAJECTORIES = 1000

# Covariance for calculating simmilar controls to the last one
STRATEGY_COVARIANCE = [[0.02, 0], [0, 0.1]]
NUMBER_OF_TRAJECTORIES = 500

NUMBER_OF_NEXT_WAYPOINTS = 20

# With of the race track (ToDo)
DIST_TOLLERANCE = 4
INVERSE_TEMP = 5
DRAW_CHOSEN_SEQUENCE = True
DRAW_TRAJECTORIES = True

MAX_COST = 1000


##########################################
############## NN Predictor ##############
##########################################


MODEL_NAME = "Dense-128-128-128-128-uniform-40"
# MODEL_NAME = "Dense-128-256-256-128_0.1s_delta"
# MODEL_NAME = "Dense-128-256-256-128_0.1s"

# Training parameters
TRAINING_DATA_FILE = "training_data_1000x1000x10.csv"
NUMBER_OF_EPOCHS = 40
BATCH_SIZE = 128
PREDICT_DELTA = True
NORMALITE_DATA = True
