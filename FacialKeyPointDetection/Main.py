import FacialDataset as FD
from torch.utils.data import DataLoader
import util as util
import FacialDetectModel as FDM
import TrainModel as TM
import math
import torch

dataset=FD.FacialDataset("./data/training.zip")
dataset_loader = DataLoader(dataset=dataset,batch_size=util.BATCH_SIZE)
model = FDM.FacialDetectModel(util.NUM_CLASSES).to(util.device)
train_model = TM.TrainModel(model,dataset_loader)
model = train_model.train(util.NUM_EPOCHS,math.ceil(len(dataset)/util.BATCH_SIZE))
torch.save(model.state_dict(),'trained_model.dt')