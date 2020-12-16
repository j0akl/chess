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

    x_train = torch.from_numpy(n_x).float()
    y_train = torch.from_numpy(n_y).float().view(-1, 1)

    # get fraction of positions that didnt end in draw
    # 6% on first iteration
    # output = [idx for idx, element in enumerate(y_train) if element[0] == 1 or
    #          element[0] == -1]
    # print(len(output) / len(raw_data))


    ds = TensorDataset(x_train, y_train)
    train_ds, val_ds = random_split(ds, [train_length, val_length])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    return train_dl, val_dl

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, 3, 1)
        self.conv2 = nn.Conv1d(32, 32, 3, 1)
        self.conv3 = nn.Conv1d(32, 32, 3, 1)
        self.conv4 = nn.Conv1d(32, 32, 3, 1)
        self.conv5 = nn.Conv1d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(864, 128)
        self.fc2 = nn.Linear(128, 1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.flat(x)
        output = torch.tanh(x)
        return output

def train(model, train_loader, optimizer, epoch, device="cpu"):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='mean').item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == "__main__":
    train_dl, val_dl = create_dataloaders('data/1k_games.npz', 128)

    model = Net().float()

    optimizer = optim.Adadelta(model.parameters())

    num_epochs = 1

    for epoch in range(num_epochs):
        train(model, train_dl, optimizer, epoch)
        test(model, val_dl)

    f = open('model/v1.pt', 'w')
    torch.save(model.state_dict(), 'model/v1.pt')
    f.close()


