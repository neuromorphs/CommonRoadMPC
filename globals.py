##########################################
########         Constants        ########
##########################################

from re import T


T_CONTROL = 0.2
T_EULER_STEP = 0.01


##########################################
#######         Experiments        #######
##########################################

SIMULATION_LENGTH = 400
DRAW_LIVE_HISTORY = True
DRAW_LIVE_ROLLOUTS = False
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"



##########################################
#########       Car & Track     ##########
##########################################

INITIAL_SPEED = 8
CONTINUE_FROM_LAST_STATE = False
ALWAYS_SAVE_LAST_STATE = False

TRACK_NAME = "track_2"
M_TO_PIXEL = 0.15
TRACK_WIDTH = 3


##########################################
#########     Car Controller     #########
##########################################

# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 200
INITIAL_STEERING_VARIANCE = 0.2
INITIAL_ACCELERATION_VARIANCE = 0.0


# Parameters for rollout
NUMBER_OF_TRAJECTORIES = 2000
STRATEGY_COVARIANCE = [[0.2, 0], [0, 0.2]]
NUMBER_OF_STEPS_PER_TRAJECTORY = 10
INVERSE_TEMP = 5

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 20
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 2
DIST_TOLLERANCE = 4

# Relations to car
MAX_SPEED = 12
MAX_COST = 400


##########################################
#########       NN Training     ##########
##########################################

# Training parameters
MODEL_NAME = "Dense-128-128-128-128-uniform-40-3"
TRAINING_DATA_FILE = "training_data_uniform_500x500x10.csv"
NUMBER_OF_EPOCHS = 40
BATCH_SIZE = 64
PREDICT_DELTA = True
NORMALITE_DATA = True
