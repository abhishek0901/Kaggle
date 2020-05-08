import time
import torch.nn as nn
import torch.optim as optim

class TrainModel:
    def __init__(self,model,dataloader,lr = 0.003):
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        self.optimizer = optim.Adam(params_to_update, lr=lr)
    def train(self,num_epochs,total_batch):
        self.model.train()
        since = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            running_loss = 0.0
            batch = 1
            for x,y in self.dataloader:
                print('Batch {}/{}'.format(batch, total_batch))
                batch += 1
                self.optimizer.zero_grad()
                outputs = self.model(x)
                non_final_mask = (y != y)
                if non_final_mask.any():
                    y[non_final_mask] = outputs[non_final_mask]
                loss = self.criterion(outputs,y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss
            print("Epoch Loss : ", epoch_loss)
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return self.model