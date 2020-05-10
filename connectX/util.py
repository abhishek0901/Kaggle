from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition',('state','action','reward','next_state','done'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_board(board):
    board_1 = board
    board_2 = board.copy()
    board_1[board_1 == 2.] = 0.
    board_2[board_2 == 1.] = 0.
    board_2[board_2 == 2.] = 1.
    return np.array([board_1,board_2])