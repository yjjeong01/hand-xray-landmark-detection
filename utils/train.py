from torch.utils.data import DataLoader
from collections import defaultdict
import torch.optim as optim
import time

from dataload import dataload
from utils import *


H = 800
W = 640
pow_n = 4
batch_size = 1
num_workers = 5
dataloaders = {
    'train': DataLoader(dataload(path='../dataset/train', H=H, W=W, pow_n=pow_n, aug=True, phase='train'), batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'valid': DataLoader(dataload(path='../dataset/valid', H=H, W=W, pow_n=pow_n, aug=False, phase='train'), batch_size=batch_size, shuffle=False, num_workers=num_workers)
}


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = load_model()

    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    print("****************************GPU : ", device)

    best_loss = 1e10

    for epoch in range(1, num_epochs + 1):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('------------------------' * 10)
        start_time = time.time()
        phases = ['train', 'valid'] if epoch % 2 == 0 and epoch > num_epochs // 2 else ['train']

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    LOSS = L2_loss(outputs, labels)
                    metrics['Jointloss'] += LOSS

                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            for param_group in optimizer.param_groups:
                lr_rate = param_group['lr']
            print(phase, "Joint loss :", epoch_Jointloss.item(), 'lr rate', lr_rate)

            savepath = 'model/net_{}_E_{}.pth'
            if phase == 'valid' and epoch_Jointloss < best_loss:
                print("model saved")
                best_loss = epoch_Jointloss
                torch.save(model.state_dict(), savepath.format(best_loss, epoch))

        print(time.time() - start_time)
