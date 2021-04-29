# CommonRoadMPC

## Installation
I recommend using a virtual environement like conda.
'pip install -r requirements.txt '

## Run
python multistep_mpc.py

## Terminology:
* control_input: array(float, 2) The input to control the car models, control_input[0] is the steeting [-1,1]
* control_input_distribution: array(control_input, n) A pack of n different control_inputs
* state: array (float, 6?), car state from commonRoad model. Length depends on which model is used
* trajectory: array (state, n), n states of the same model calculated each 0.2s following a control_input
* trajectory_distribution: array(trajectory, n) A pack of n trajectories, following a control_input_distribution
