# CommonRoadMPC
This project is a lite version of the L2Race project. It is optimized for implementing controllers.

There is a neural network based MPC already implemented. 
For further information about the MPC contact me: bollif@ethz.ch

## Setup

I recommend using VSCode and a virtual environement like conda.
Make sure you add the git submodules.
```shell script
git clone --recursive https://github.com/neuromorphs/CommonRoadMPC.git
cd CommonRoadMPC
```

Create the conda environment:
```shell script
conda create --name cr
conda activate cr
```

Install requirements:
```shell script
pip install -r requirements.txt
```
If you use miniconda, there might be some missing dependancies, just install them via pip.


## Run
Make sure your conda environment is activated.
If you work from terminal, run:
```shell script
export PYTHONPATH=./
```
To start the simulation:
```shell script
python run_simulation.py
```

This will create an image live_history.png which shows the trajectory of the car in comparison to the race track.
Once the simulation is complete, there will be a record of the car's states and controls in the folder ExperimentRecordings.

## Information
You can implement your own controller and test it in the run_simulation.py
For settings of the Car/Track, please check globals.py

For implementing an own contoller you won't need the following subfolders(they only belong to the neural MPC): mppi_mpc, nn_prediction, SI_Toolkit, SI_Tookit_ApplicationSpecificFiles, Old.

If you need to generate training data for a neural network, you can check out nn_prediction/generate_data.py.


## Neural MPC controller
You can run the simulation with the already implemented NueralMPC controller:

Download the Keras models from: https://www.dropbox.com/sh/wxihelna93wnl24/AAC-79Yo3M6xw5G2O832aiPia?dl=0 \
and place them into the folder nn_prediction/models
(the models folder might not exist: just create it.)

To start the simulation:
```shell script
python run_mpc.py
```


## Terminology:
* control_input: array(float, 2) The input to control the car models, control_input[0] is the steeting [-1,1]
* control_input_distribution: array(control_input, n) A pack of n different control_inputs
* state: array (float, 7?), car state from commonRoad model. Length depends on which model is used. 7 for ST model
* trajectory: array (state, n), n states of the same model calculated each 0.2s following a control_input
* trajectory_distribution: array(trajectory, n) A package of n trajectories, following a control_input_distribution
