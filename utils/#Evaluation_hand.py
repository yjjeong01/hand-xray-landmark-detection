from torch.utils.data import DataLoader

from utils import *
from dataload import *

H = 800
W = 640
pow_n = 8
batch_size = 1
num_workers = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dataloaders = {
    'train': DataLoader(dataload(path='../dataset/train', H=H, W=W, pow_n=pow_n, aug=False), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    'valid': DataLoader(dataload(path=r"../dataset/valid", H=H, W=W, pow_n=pow_n, aug=False), batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

if __name__ == '__main__':
    model = load_model('valid')
    img_size = np.load('../img_size.npy')

    distances = []

    for (inputs, labels), size in zip(dataloaders['valid'], img_size):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        pred = get_heatmap(outputs, 1e-10, size[0], size[1])
        gt = get_heatmap(labels, 1e-10, size[0], size[1])
        distance = (pred - gt).pow(2).sum(1).sqrt().cpu().numpy()
        distances.append(distance)

    mm = 20
    mtx = np.array(distances)
    print("2mm:", np.mean(mtx < mm), "2.5mm:", np.mean(mtx < mm * 1.25),
          "3mm:", np.mean(mtx < mm * 1.5), "4mm:", np.mean(mtx < mm * 2), "avg:", np.mean(mtx))
