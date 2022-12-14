from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import set_sumo, set_train_path
from constants import *

sumo_cmd = set_sumo(TRAIN_GUI, S_CONF_FILE, MAX_STEPS)
path = set_train_path(MDL_PATH)

Model = TrainModel()

Memory = Memory()
    
Simulation = Simulation(
    Model,
    Memory,
    sumo_cmd,
    0.75,
    MAX_STEPS,
    G_DUR_SEC,
    Y_DUR_SEC,
    N_STATES,
    N_ACTIONS,
    TRAIN_EPOCHS,
)

episode = 0
timestamp_start = datetime.datetime.now()

while episode < TRAIN_TOT_EPIS:
    print('\n----- Episode', str(episode+1), 'of', str(TRAIN_TOT_EPIS))
    epsilon = 1.0 - (episode / TRAIN_TOT_EPIS)  # set the epsilon for this episode according to epsilon-greedy policy
    simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
    print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
    episode += 1

print("\n----- Start time:", timestamp_start)
print("----- End time:", datetime.datetime.now())
print("----- Session info saved at:", path)

Model.save_model(path)

copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
