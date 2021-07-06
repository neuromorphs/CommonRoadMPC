##########################################
########         Constants        ########
##########################################

T_CONTROL = 0.2
T_EULER_STEP = 0.01


##########################################
#######         Experiments        #######
##########################################

SIMULATION_LENGTH = 1000
DRAW_LIVE_HISTORY = True
DRAW_LIVE_ROLLOUTS = False
PATH_TO_EXPERIMENT_RECORDINGS = "./ExperimentRecordings"


##########################################
#########       Car & Track     ##########
##########################################

INITIAL_SPEED = 13
CONTINUE_FROM_LAST_STATE = False
ALWAYS_SAVE_LAST_STATE = False
EXIT_AFTER_ONE_LAP = False

TRACK_NAME = "track_2"
M_TO_PIXEL = 0.1
TRACK_WIDTH = 4


##########################################
####   Neural MPC Car Controller     #####
##########################################

# Path Prediction
CONTROLLER_PREDICTIOR = "nn"
CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-invariant-10" # Accurate
# CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-small" # Small training data


# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 2500
INITIAL_STEERING_VARIANCE = 0.5
INITIAL_ACCELERATION_VARIANCE = 0.5


# Parameters for rollout
NUMBER_OF_TRAJECTORIES = 250
STEP_STEERING_VARIANCE = 0.1
STEP_ACCELERATION_VARIANCE = 0.1
NUMBER_OF_STEPS_PER_TRAJECTORY = 15
INVERSE_TEMP = 5

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 20
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 1
ANGLE_COST_INDEX_START = 5
ANGLE_COST_INDEX_STOP = 15

# Relations to car
MAX_SPEED = 15
MAX_COST = 1000


##########################################
#########       NN Training     ##########
##########################################

# Artificial data generation
DATA_GENERATION_FILE = "training_data_5-15_50x30x10.csv"

# Training parameters
MODEL_NAME = "Dense-128-128-128-128-small"
TRAINING_DATA_FILE = "training_data_5-15_50x30x10.csv"
NUMBER_OF_EPOCHS = 150
BATCH_SIZE = 128
PREDICT_DELTA = True
NORMALITE_DATA = True
CUT_INVARIANTS = True
