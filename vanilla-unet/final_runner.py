"""
Run this file to train UNET
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from BraTSdata_traintest_split import BraTStrainingNoBlankTrainVal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
MODEL
"""


class UNet(nn.Module):
    """From https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py"""

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


# %%


"""
Dataset and Dataloader
"""
BASE_PATH = '/home/Student/s4606685/BraTS2021_Training_Data/BraTS2021_Training_Data'
BATCH_SIZE = 16
ds = BraTStrainingNoBlankTrainVal(data_path=BASE_PATH, no_adjacent_slices=1, val_offset=200, train=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)
val_ds = BraTStrainingNoBlankTrainVal(data_path=BASE_PATH, no_adjacent_slices=1, val_offset=200, train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1)
"""
Loss fn
"""


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


"""
Train, test loops
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
"""
model = UNet()
device = 'cuda'
lr = 1e-3
dsc_loss = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr)
epochs = 30

loss_train = []
loss_val = []
step = 0
model = model.to(device)

writer_real = SummaryWriter(f"logs/real2")
write_pred = SummaryWriter(f"logs/pred2")
write_train = SummaryWriter(f"logs/train2")
write_val = SummaryWriter(f"logs/val2")

for epoch in range(epochs):
    # Train
    print(epoch)
    running_loss = 0
    for batch_idx, (X, y_true) in enumerate(dl):
        X = X.to(device)
        y_true = y_true.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = dsc_loss(y_pred, y_true)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss_train.append(running_loss)
    # WRITE TRAIN LOOP
    write_train.add_scalar("train", running_loss, global_step=step)
    # Test TODO NEED TO SPLIT DATASET
    running_loss = 0
    for batch_idx, (X, y_true) in enumerate(val_dl):
        X = X.to(device)
        y_true = y_true.to(device)
        with torch.no_grad():
            y_pred = model(X)
            loss = dsc_loss(y_pred, y_true)
            running_loss += loss.item()
    loss_val.append(running_loss)
    write_val.add_scalar("val", running_loss, global_step=step)

    # Summary writer to show actual, predicted mask
    with torch.no_grad():
        real = next(enumerate(dl))
        real_y = real[1][1]
        real_y = real_y.to(device)
        real = real[1][0]
        real = real.to(device)
        pred = model(real)
        img_grid_real = torchvision.utils.make_grid(
            real_y[:5], normalize=True
        )
        img_grid_pred = torchvision.utils.make_grid(
            pred[:5], normalize=True
        )

        writer_real.add_image("Real", img_grid_real, global_step=step)
        write_pred.add_image("Pred", img_grid_pred, global_step=step)
    step += 1
