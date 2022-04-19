import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from model import UNet, weights_init_relu


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        # save all args to self.hparams
        self.save_hyperparameters()
        self.args = self.hparams.args

        # Network
        self.print_("Using UNet")
        self.net = UNet(in_channels=self.args.slices_on_forward, out_channels=1, init_features=self.args.model_width)
        self.net.apply(weights_init_relu)

        # Example input array needed to log the graph in tensorboard
        # input_size = (1, args.img_size, args.img_size)
        input_size = (self.args.slices_on_forward, self.args.img_size, self.args.img_size)
        self.example_input_array = torch.randn(
            [5, *input_size])

        # Init Loss function
        self.loss_fn = torch.nn.BCELoss()

        if self.logger:
            self.logger.log_hyperparams(self.args)

    def forward(self, x):
        y = self.net(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=0.5 * 0.0005)
        return [optimizer]

    def print_(self, msg):
        if self.args.verbose:
            print(msg)

    def log_metric(self, name, value, on_step=None, on_epoch=None):
        if self.logger:
            self.log(name, value, on_step=on_step, on_epoch=on_epoch,
                     logger=True)
        if self.args.hparam_search:
            tune.report(value)

