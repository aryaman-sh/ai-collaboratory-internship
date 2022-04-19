"""
CDG implementation outline

Each participant l independently initialises the starting point of its confined model.
Every training iter, l independently compute gradient based on local data.
Jointly work out sum of gradient.
All participants update their confined models.

Initial implementation do normal addition for gradients not secure version and test if that works first
"""
import torch
import torch.nn as nn


class Participant:
    """
    A participant in CDG has its confined model
    """

    def __init__(self, id):
        self.id = id
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.loss = None

    def init_loss(self):
        self.loss = torch.nn.functional.nll_loss()

    def init_optim(self, lr):
        """
        Init optimizer
        :param lr: learning rate
        """
        if self.model is None:
            raise ValueError('Model not initialized')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_id(self):
        """
        :return: Id
        """
        return self.id

    def init_model(self, model_sig):
        """
        Initialise a model for this participant
        :param model_sig: model to init
        """
        self.model = model_sig

    def init_dataloader(self, dl):
        """
        Assign dl to this participant
        """
        self.dataloader = dl

    def perform_iter(self):
        """
        Perform iter and return the gradients, note no update
        :return: model.parameters gradients
        """
        self.optimizer.zero_grad()
        for batch_idx, (input, labels) in enumerate(self.dataloader):
            output = self.model(input)
            loss = self.loss(output, labels)
            loss = loss/len(self.dataloader) # normalize for accumulation
            loss.backward()
        grads = self.model.parameters()


    def update_model(self, gradients):
        """
        Update step on gradients
        :param gradients:
        :return:
        """
