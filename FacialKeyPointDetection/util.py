import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 30
NUM_EPOCHS = 1
RANGE = 5

def convert_to_matrix(data):
    return np.array(data.split(' '), dtype=float).reshape(96, 96)  # Images are 96 * 96