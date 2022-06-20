"""
Runner UNET with Confined Gradient descent
Num of adjacent slices=0
Num participants=3
"""

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


class Participant:

    def __init__(self, id):
        self.device = 'cuda'
        self.id = id
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.verbose = True
        self.step_size = 1e-3
        self.prev_step_train_loss = 0

    def set_verbose(self, v):
        self.verbose = v

    def init_model(self, model):
        """
        Sets model for this participant
        :param model: model
        """
        self.model = model

    def init_optim(self, lr):
        if self.model is None:
            raise ValueError("Model not initialized")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def init_dataloader(self, dl):
        if self.dataloader is None:
            self.dataloader = dl

    def init_loss(self):
        if self.loss is None:
            self.loss = DiceLoss()

    def compute_local_grads(self):
        """
        Computes an iter of grads based on local model
        """
        if self.verbose:
            print(f'Computing local grad for {str(self.id)}')
        self.optimizer.zero_grad()
        running_loss = 0
        for batch_idx, (X, y) in enumerate(self.dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            output = self.model(X)
            loss = self.loss(output, y)
            loss = loss / len(self.dataloader)  # normalize for accumulation
            running_loss += loss.item()
            loss.backward()
        self.prev_step_train_loss = running_loss

    def step_update(self):
        """
        Update new gradients
        """
        self.optimizer.step()

    def training_evaluations(self):
        """
        :return: Training loss
        """
        return self.prev_step_train_loss

    def val_loss(self, val_dl):
        """
        TODO: Evaluate and return loss of val dl on this set
        :param val_dl:
        :return:
        """
        running_loss = 0
        for batch_idx, (x, y_true) in enumerate(val_dl):
            x = x.to('cuda')
            y_true = y_true.to('cuda')
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_true)
                running_loss += loss.item()
        return running_loss


BASE_PATH = '/home/Student/s4606685/BraTS2021_Training_Data/BraTS2021_Training_Data'
BATCH_SIZE = 16
device = 'cuda'
num_participants = 3
lr = 1e-3

# Init P1
p1 = Participant(id=1)
model_p1 = UNet()
model_p1 = model_p1.to(device)
p1.init_model(model_p1)
ds1 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl1 = DataLoader(ds1, batch_size=16, num_workers=1)
p1.init_dataloader(dl1)
p1.init_optim(lr)
p1.init_loss()

# Init p2
p2 = Participant(id=2)
model_p2 = UNet()
model_p2 = model_p2.to(device)
p2.init_model(model_p2)
ds2 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=2,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl2 = DataLoader(ds2, batch_size=16, num_workers=1)
p2.init_dataloader(dl2)
p2.init_optim(lr)
p2.init_loss()

# Init p3
p3 = Participant(id=3)
model_p3 = UNet()
model_p3 = model_p3.to(device)
p3.init_model(model_p3)
ds3 = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=1, this_participant=1,
                                           no_adjacent_slices=1, val_offset=200, train=True)
dl3 = DataLoader(ds3, batch_size=16, num_workers=1)
p3.init_dataloader(dl3)
p3.init_optim(lr)
p3.init_loss()

# Training
epochs = 30

training_loss_p1 = []
training_loss_p2 = []
training_loss_p3 = []

val_loss_p1 = []
val_loss_p2 = []
val_loss_p3 = []

ds_val = BraTStrainingNoBlankTrainValCGDSplit(data_path=BASE_PATH, total_participants=3, this_participant=3,
                                              no_adjacent_slices=1, val_offset=200, train=False)
dl_val = DataLoader(ds_val, batch_size=16, num_workers=1)

writer = SummaryWriter(f"logs/cgd")

for j in range(epochs):
    print(f'Training epoch: {j + 1}')
    # Compute local gradients
    p1.compute_local_grads()
    p2.compute_local_grads()
    p3.compute_local_grads()

    print(f'Calculated local grads for {j + 1}')

    # Sum grads
    '''
    for pA, pB, pC in zip(p1.model.parameters(), p2.model.parameters(), p3.model.parameters()):
        sum_grads = pA.grad.clone() + pB.grad.clone() + pC.grad.clone()
        pA.grad = sum_grads.clone()
        pB.grad = sum_grads.clone()
        pC.grad = sum_grads.clone()
    '''
    for pA, pB in zip(p1.model.parameters(), p2.model.parameters()):
        sum_grads = pA.grad.clone() + pB.grad.clone()
        pA.grad = sum_grads.clone()
        pB.grad = sum_grads.clone()

    # Step
    p1.step_update()
    p2.step_update()
    p3.step_update()

    print(f'step updates applied for epoch {j + 1}')

    training_loss_p1.append(p1.training_evaluations())
    training_loss_p2.append(p2.training_evaluations())
    training_loss_p3.append(p3.training_evaluations())

    # Val error evaluation
    val_loss_p1.append(p1.val_loss(dl_val))
    val_loss_p2.append(p2.val_loss(dl_val))
    val_loss_p3.append(p3.val_loss(dl_val))

    print(training_loss_p1[j])
    print(training_loss_p2[j])
    print(training_loss_p3[j])
    # Write training loss
    writer.add_scalars('17_5_22', {
        'train_p1': training_loss_p1[j],
        'train_p2': training_loss_p2[j],
        'train_p3': training_loss_p3[j],
        'val_p1': val_loss_p1[j],
        'val_p2': val_loss_p2[j],
        'val_p3': val_loss_p3[j],
    }, j)
