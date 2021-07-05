##########################################
########         Constants        ########
##########################################

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

INITIAL_SPEED = 9
CONTINUE_FROM_LAST_STATE = False
ALWAYS_SAVE_LAST_STATE = False

TRACK_NAME = "track_3"
M_TO_PIXEL = 0.1
TRACK_WIDTH = 6


##########################################
####   Neural MPC Car Controller     #####
##########################################

# Path Prediction
CONTROLLER_PREDICTIOR = "nn"
CONTROLLER_MODEL_NAME = "Dense-128-128-128-128-invariant-10"

# Initializing parameters
NUMBER_OF_INITIAL_TRAJECTORIES = 500
INITIAL_STEERING_VARIANCE = 0.5
INITIAL_ACCELERATION_VARIANCE = 0.5


# Parameters for rollout
NUMBER_OF_TRAJECTORIES = 250
STEP_STEERING_VARIANCE = 0.1
STEP_ACCELERATION_VARIANCE = 0.1
NUMBER_OF_STEPS_PER_TRAJECTORY = 10
INVERSE_TEMP = 5

# Relation to track
NUMBER_OF_NEXT_WAYPOINTS = 20
NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS = 2

# Relations to car
MAX_SPEED = 15
MAX_COST = 1000


##########################################
#########       NN Training     ##########
##########################################

# Artificial data generation
DATA_GENERATION_FILE = "training_data_0-25_1500x500x10.csv"

# Training parameters
MODEL_NAME = "Dense-256-256-256-128-high_speed_var-10-64"
TRAINING_DATA_FILE = "training_data_0-25_1500x500x10.csv"
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 64
PREDICT_DELTA = True
NORMALITE_DATA = True
CUT_INVARIANTS = True
