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
dl1 = DataLoader(ds1, batch_size=16, num_workers=0)

model_p2 = UNet()

model_p2 = model_p2.to(device)
optim2 = torch.optim.Adam(model_p2.parameters(), lr=lr)
loss_2 = DiceLoss()
ds2 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl2 = DataLoader(ds2, batch_size=16, num_workers=0)

model_p3 = UNet()

model_p3 = model_p3.to(device)
optim3 = torch.optim.Adam(model_p3.parameters(), lr=lr)
loss_3 = DiceLoss()
ds3 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=1, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl3 = DataLoader(ds3, batch_size=16, num_workers=0)

ds_val = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=1, this_participant=1,
                                              no_adjacent_slices=1, val_offset=200, train=False)
dl_val = DataLoader(ds_val, batch_size=16, num_workers=0)

writer = SummaryWriter(f"logs/cgd")

for j in range(1500):
    dl1_enumerator = enumerate(dl1)
    dl2_enumerator = enumerate(dl2)
    dl3_enumerator = enumerate(dl3)
    for i in range(len(dl3)):
        if i % len(dl1) == 0:
            dl1_enumerator = enumerate(dl1)
            dl2_enumerator = enumerate(dl2)

        print(f'Training iter {(i + 1) * (j + 1)}')
        batch1, (X1, y1) = next(dl1_enumerator)
        batch2, (X2, y2) = next(dl2_enumerator)
        batch3, (X3, y3) = next(dl3_enumerator)
        optim1.zero_grad()
        optim2.zero_grad()

        X1 = X1.to(device)
        X2 = X2.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        X3 = X3.to(device)
        y3 = y3.to(device)

        y1_pred = model_p1(X1)
        loss1 = loss_1(y1_pred, y1)
        loss1.backward()
        train_loss_1 = loss1.item()

        y2_pred = model_p2(X2)
        loss2 = loss_2(y2_pred, y2)
        loss2.backward()
        train_loss_2 = loss2.item()

        for pA, pB in zip(model_p1.parameters(), model_p2.parameters()):
            sum_grads = pA.grad + pB.grad
            pA.grad = sum_grads
            pB.grad = sum_grads.clone()

        optim1.step()
        optim2.step()

        optim3.zero_grad()
        y3_pred = model_p3(X3)
        loss3 = loss_3(y3_pred, y3)
        loss3.backward()
        train_loss_3 = loss3.item()
        optim3.step()

        # Validation
        for batch_val, (Xval, yval) in enumerate(dl_val):
            Xval = Xval.to(device)
            yval = yval.to(device)
            with torch.no_grad():
                pred1 = model_p1(Xval)
                pred2 = model_p2(Xval)
                pred3 = model_p3(Xval)
                loss1val = loss_1(pred1, yval)
                loss2val = loss_2(pred2, yval)
                loss3val = loss_3(pred3, yval)
                valloss1 = loss1val.item()
                valloss2 = loss2val.item()
                valloss3 = loss3val.item()

        outindex = (j+1)*(i+1)
        writer.add_scalars('696969', {
            '1': train_loss_1,
            '2': train_loss_2,
            '3': train_loss_3,
            '1val': valloss1,
            '2val': valloss2,
            '3val': valloss3
        }, outindex)
