import numpy as np
from matplotlib import colors

from tqdm import trange


from SI_Toolkit.TF.TF_Functions.predictor_autoregressive_tf import predictor_autoregressive_tf

# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm


STATE_VARIABLES = ['x1','x2','x3','x4','x5','x6','x7']
cartpole_state_varnames_to_indices = [0,1,2,3,4,5,6]
if get_backend() != 'module://backend_interagg':
    use('Agg')


cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

def get_data_for_gui_TF(a, dataset, net_name):

    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    #  states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]
    Inputs = dataset[['u1', 'u2']].copy()
    Q = Inputs.to_numpy()
    # Q = dataset['U1'].to_numpy()
    print("Q[i:-a.test_max_horizon+i]", Q[-a.test_max_horizon])
    print("a.test_max_horizon", a.test_max_horizon)
  
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.array(Q_array).transpose(1, 0, 2)

    #Q array is the next TEST_MAX_HORIZON controls for each time step 
    # Shape is 80,20, (2?)

    print("Q", Q_array)
    print("Q", Q_array.shape)


    # mode = 'batch'
    mode = 'sequential'
    if mode == 'batch':
        # All at once
        predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=a.test_len, net_name=net_name)
        predictor.setup(initial_state=states_0, prediction_denorm=True)
        output_array = predictor.predict(Q_array)
    elif mode == 'sequential':
        print("sequential")
        # predictor = predictor_autoregressive_tf(a=a, batch_size=1)
        predictor = predictor_autoregressive_tf(horizon=a.test_max_horizon, batch_size=1, net_name=net_name)
        # Iteratively (to test internal state update)
        output_array = np.zeros([a.test_len, a.test_max_horizon, len(STATE_VARIABLES)], dtype=np.float32)
      
        for timestep in trange(a.test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :]
            s_current_timestep = states_0[timestep, np.newaxis]
            predictor.setup(initial_state=s_current_timestep, prediction_denorm=True)
            prediction = predictor.predict(Q_current_timestep)

            print("Q_current_timestep", Q_current_timestep)
            print("s_current_timestep", s_current_timestep)

            output_array[timestep,:,:] = prediction
           

            predictor.update_internal_state(Q_current_timestep[0, 0])
            print("prediction",prediction)
          
          

    # output_array = output_array[..., list(cartpole_state_varnames_to_indices(a.features))+[-1]]

    # time_axis is a time axis for ground truth
    return output_array
