import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import floor, ceil

def create_dataloaders(filepath, batch_size=32):

    # creates a train and val dl with an 80/20 split, train data is shuffled
    raw_data = np.load(filepath, allow_pickle=True)['arr_0']

    len_data = len(raw_data)
    train_length = floor(len_data * .8)
    val_length = ceil(len_data * .2)

    raw_x = []
    raw_y = []

    for i in range(len(raw_data)):
        raw_x.append(raw_data[i][1])
        raw_y.append(raw_data[i][0])

    n_x = np.array(raw_x)
    n_y = np.array(raw_y)

    x_train = torch.from_numpy(n_x)
    y_train = torch.from_numpy(n_y)

    ds = TensorDataset(x_train, y_train)
    train_ds, val_ds = random_split(ds, [train_length, val_length])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    return train_dl, val_dl

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

if __name__ == "__main__":
    train_dl, val_dl = create_dataloaders('data/10_games.npz')


