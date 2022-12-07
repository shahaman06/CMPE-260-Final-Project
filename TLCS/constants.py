## TESTING
GUI = True # gui
MAX_STEPS = 5400 # max_steps
CARS_GEN = 1000 # n_cars_generated
SEED = 10000 # episode_seed
Y_DUR_SEC = 4 # yellow_duration
G_DUR_SEC = 10 # green_duration

N_STATES = 80 # num_states
N_ACTIONS = 4 # num_actions

MDL_PATH = 'models' # models_path_name
S_CONF_FILE = 'sumo_config.sumocfg' # sumocfg_file_name
MDL_TEST_NUM = 2 # model_to_test


##TRAINING 
TRAIN_GUI = False # gui 
TRAIN_TOT_EPIS = 100 # total_episodes

# [model]
TRAIN_N_LAYERS = 4 # num_layers
TRAIN_W_LAYERS = 400 # width_layers
TRAIN_BATCH_SIZE = 100 # batch_size
TRAIN_LR = 0.001 # learning_rate
TRAIN_EPOCHS = 800 # training_epochs

# [memory]
TRAIN_MEM_MIN = 600 # memory_size_min
TRAIN_MEM_MAX = 50000 # memory_size_max
