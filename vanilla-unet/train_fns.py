"""
Training and val functions for model training
"""
import torch
import torch.nn as nn


def train_loop(dataloader, model, optimizer, loss_fn, scaler, device):
    size = len(dataloader.dataset)
    for batch_idx, (image, mask) in enumerate(dataloader):
        image = image.float().to(device)
        mask = mask.float().to(device)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(image)
            loss = loss_fn(prediction, mask)

        # backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
