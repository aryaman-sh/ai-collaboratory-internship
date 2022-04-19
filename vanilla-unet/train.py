import torch
import torch.nn as nn
import nibabel as nib
from BraTSdata import BraTStrainingNoBlank
from torch.utils.data import Dataset, DataLoader
from model import UNet
from train_fns import train_loop
from torch.utils.tensorboard import SummaryWriter
import torchvision

"""
Params
"""
DEVICE = 'cuda'
LEARNING_RATE = 1e-3
loss_fn = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

"""
Dataset and Dataloader
"""
BASE_PATH = '/home/Student/s4606685/BraTS2021_Training_Data/BraTS2021_Training_Data'
BATCH_SIZE = 16
ds = BraTStrainingNoBlank(data_path=BASE_PATH, no_adjacent_slices=1, val_offset=150)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1)

"""
MODEL
"""
model = UNet(in_channels=3, out_channels=1)
model = model.to(DEVICE)
"""
Training Function
"""
writer_real = SummaryWriter(f"logs/real")
write_pred = SummaryWriter(f"logs/pred")


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
step = 0
for j in range(5):

    print(f" Epoch = {j} \n +++++++++++++++++++++++++++++++++++++")
    train_loop(dataloader=dl, model=model, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler, device=DEVICE)

    # Summary writer to show actual, predicted mask
    with torch.no_grad():
        real = next(enumerate(dl))
        real = real[1][0]
        real = real.float().to(DEVICE)
        pred = model(real)
        img_grid_real = torchvision.utils.make_grid(
            real[:5], normalize=True
        )
        img_grid_pred = torchvision.utils.make_grid(
            pred[:5], normalize=True
        )

        writer_real.add_image("Real", img_grid_real, global_step=step)
        write_pred.add_image("Pred", img_grid_pred, global_step=step)

    step+=1

#%%
