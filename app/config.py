import torch
from model import *

CHECKPOINT_PATH = './checkpoint'
MODEL_PATH = './bestmodel.pt'
LR = 1e-3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']

    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
