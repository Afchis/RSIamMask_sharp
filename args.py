import torch


BATCH_SIZE = 16
# TIMESTEPS = 1
INPUT_CHANNELS = 3
SEARCH_SIZE = 256
TARGET_SIZE = 128
NUM_CLASSES = 1

# MODE = 'Standart'
# CELL_MODEL = 'Rnn'


DEVICE = "cuda:0"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# tensorboard --logdir=runs
