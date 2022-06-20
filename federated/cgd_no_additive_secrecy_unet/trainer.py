import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from collections import OrderedDict
from BraTSdataParticipantSplit import BraTStrainingNoBlankTrainValCGDSplit
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet_model import UNet


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


BASE_PATH = '/home/Student/s4606685/BraTS2021_Training_Data/BraTS2021_Training_Data'
BATCH_SIZE = 16
device = 'cuda'
num_participants = 3
lr = 1e-3

model_p1 = UNet()
model_p1 = model_p1.to(device)
optim1 = torch.optim.Adam(model_p1.parameters(), lr=lr)
loss_1 = DiceLoss()
ds1 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl1 = DataLoader(ds1, batch_size=16, num_workers=1)

model_p2 = UNet()

model_p2 = model_p2.to(device)
optim2 = torch.optim.Adam(model_p2.parameters(), lr=lr)
loss_2 = DiceLoss()
ds2 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl2 = DataLoader(ds2, batch_size=16, num_workers=1)

model_p3 = UNet()

model_p3 = model_p3.to(device)
optim3 = torch.optim.Adam(model_p3.parameters(), lr=lr)
loss_3 = DiceLoss()
ds3 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=1, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl3 = DataLoader(ds3, batch_size=16, num_workers=1)

train_loss_p1 = []
train_loss_p2 = []
train_loss_p3 = []

writer = SummaryWriter(f"logs/cgd")
for j in range(10):
    # 1
    running_loss = 0
    for batch_idx, (X, y_true) in enumerate(dl1):
        X = X.to(device)
        y_true = y_true.to(device)
        optim1.zero_grad()
        y_pred = model_p1(X)
        loss1 = loss_1(y_pred, y_true)
        running_loss += loss1.item()
        loss1.backward()
    running_loss = running_loss / len(dl1)
    train_loss_p1.append(running_loss)

    running_loss = 0
    for batch_idx, (X, y_true) in enumerate(dl2):
        X = X.to(device)
        y_true = y_true.to(device)
        optim2.zero_grad()
        y_pred = model_p2(X)
        loss2 = loss_2(y_pred, y_true)
        running_loss += loss2.item()
        loss2.backward()
    running_loss = running_loss / len(dl2)
    train_loss_p2.append(running_loss)

    for pA, pB in zip(model_p1.parameters(), model_p2.parameters()):
        sum_grads = pA.grad.clone() + pB.grad.clone()
        pA.grad = sum_grads.clone()
        pB.grad = sum_grads.clone()

    optim1.step()
    optim2.step()

    running_loss = 0
    for batch_idx, (X, y_true) in enumerate(dl3):
        X = X.to(device)
        y_true = y_true.to(device)
        optim3.zero_grad()
        y_pred = model_p3(X)
        loss = loss_3(y_pred, y_true)
        running_loss += loss.item()
        loss.backward()
        optim3.step()
    running_loss = running_loss / len(dl3)
    train_loss_p3.append(running_loss)

    print(train_loss_p1[j])
    print(train_loss_p2[j])
    print(train_loss_p3[j])

    writer.add_scalars('17_5_22_1', {
        '1': train_loss_p1[j],
        '2': train_loss_p2[j],
        '3': train_loss_p3[j],
    }, j)
