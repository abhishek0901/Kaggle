import util as util
import FacialDetectModel as FDM
import torch
import FacialDataset as FD
from skimage.color import gray2rgb
from skimage.transform import resize

model = FDM.FacialDetectModel(util.NUM_CLASSES).to(util.device)
model.load_state_dict(torch.load('trained_model.dt'))
dataset=FD.FacialDataset("./data/training.zip")
model.eval()
for row in range(util.RANGE):
    x = dataset.X[row]
    y_o = dataset.Y[row]
    x = torch.from_numpy(gray2rgb(resize(util.convert_to_matrix(x), (299, 299)))).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0) / 255
    x = x.to(device=util.device).to(torch.float32)
    y_pred = model(x)
    print(y_o,y_pred)