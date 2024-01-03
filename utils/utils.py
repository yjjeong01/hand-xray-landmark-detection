import sys
sys.path.append(f'{sys.path[0]}/utils')

import numpy as np
import torch
import UNet as network


def load_model(phase='train', num_classes=37):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = network.UNet(1, num_classes).to(device)
    if phase != 'train':
        model.load_state_dict(torch.load(r"net_2.0276949726394378e-05_E_941.pth", map_location=device))
        model = model.eval()

    return model


def get_heatmap(outputs, beta, origin_h, origin_w, H=800, W=640):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
    Ymap, Xmap = (torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to(device),
                  torch.tensor(Xmap.flatten(), dtype=torch.float).unsqueeze(1).to(device))

    heat = torch.cat([argsoftmax(outputs[0].view(-1, H * W), Ymap, beta=beta) * (origin_h / H),
                      argsoftmax(outputs[0].view(-1, H * W), Xmap, beta=beta) * (origin_w / W)],
                     dim=1).detach().cpu()

    return heat


def argsoftmax(x, index, beta=1e-2):
    a = torch.exp(-torch.abs(x - x.max(dim=1).values.unsqueeze(1)) / beta)
    b = torch.sum(a, dim=1).unsqueeze(1)
    softmax = a / b
    return torch.mm(softmax, index)
