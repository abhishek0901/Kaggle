from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition',('state','action','reward','next_state','done'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#needs to be updated by configuration
ROWS = 6
COLUMNS = 7
K = 4

def process_board(board):
    board_1 = board
    board_2 = board.copy()
    board_1[board_1 == 2.] = 0.
    board_2[board_2 == 1.] = 0.
    board_2[board_2 == 2.] = 1.
    return np.array([board_1,board_2])

def get_list(board,MARK,ROWS,COLUMNS,K):
    column_list = []
    THREATS = []
    n_in_row = K
    # Get first avaialble point for that column
    i=-1
    for j in range(COLUMNS):
        for row_i in reversed(range(ROWS)):
            if board[row_i][j] == 0:
                i = row_i
                break

        if i==-1:
            continue
        # Horizontal
        cnt = 1
        for k in range(1, n_in_row):
            if j + k < COLUMNS and board[i][j + k] == MARK:
                cnt += 1
            else:
                break

        for k in range(1, n_in_row):
            if j - k >= 0 and board[i][j - k] == MARK:
                cnt += 1
            else:
                break

        if cnt >= n_in_row:
            column_list.append(j)
            continue

        # Vertical
        cnt = 1
        for k in range(1, n_in_row):
            if i + k < ROWS and board[i + k][j] == MARK:
                cnt += 1
            else:
                break

        # This part will never work -> have put here just for symmetry
        for k in range(1, n_in_row):
            if i - k >= 0 and board[i - k][j] == MARK:
                cnt += 1
            else:
                break

        if cnt >= n_in_row:
            column_list.append(j)
            continue

        # Diagonal 1
        cnt = 1
        for k in range(1, n_in_row):
            if i + k < ROWS and j + k < COLUMNS and board[i + k][j + k] == MARK:
                cnt += 1
            else:
                break

        for k in range(1, n_in_row):
            if i - k >= 0 and j - k >= 0 and board[i - k][j - k] == MARK:
                cnt += 1
            else:
                break
        if cnt >= n_in_row:
            column_list.append(j)
            continue

        # Diagonal 2
        cnt = 1
        for k in range(1, n_in_row):
            if i - k >= 0 and j + k < COLUMNS and board[i - k][j + k] == MARK:
                cnt += 1
            else:
                break

        for k in range(1, n_in_row):
            if i + k < ROWS and j - k >= 0 and board[i + k][j - k] == MARK:
                cnt += 1
            else:
                break

        if cnt >= n_in_row:
            column_list.append(j)
            continue

    return column_list

def get_threats_and_column_list(board):

    board = board.reshape((ROWS,COLUMNS))
    OPPORTUNITY = get_list(board,MARK=1,ROWS=ROWS,COLUMNS=COLUMNS,K=K)
    THREATS = get_list(board, MARK=2, ROWS=ROWS, COLUMNS=COLUMNS,K=K)

    THREATS = list(set(THREATS) - set(OPPORTUNITY))

    return OPPORTUNITY, THREATS

#B = np.zeros(42)
#B[36] = 1
#B[37] = 2
#get_threats_and_column_list(B)