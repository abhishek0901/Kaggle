from torch.utils.data import Dataset
import util as util
import pandas as pd
import torch
from skimage.color import gray2rgb
from skimage.transform import resize


class FacialDataset(Dataset):
    def __init__(self, filename):
        input_data = pd.read_csv(filename)
        self.X = input_data['Image'].to_numpy()
        input_data.drop(['Image'], axis=1, inplace=True)
        self.Y = input_data.to_numpy()
        self.len = input_data.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(gray2rgb(resize(util.convert_to_matrix(self.X[index]), (299, 299)))).type(
            torch.FloatTensor).permute(2, 0, 1) / 255
        x = x.to(device=util.device).to(torch.float32)
        y = torch.from_numpy(self.Y[index]).to(device=util.device)
        y = y.to(torch.float32)
        return x, y

    def __len__(self):
        return self.len
