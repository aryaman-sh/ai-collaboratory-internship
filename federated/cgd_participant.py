import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

"""
Notes:
1) No parameter updates take place and grads for whole dataset are collected in one go
"""


class Participant:

    def __init__(self, id):
        self.id = id
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.verbose = True
        self.step_size = 1e-3

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
            self.loss = torch.nn.NLLLoss()

    def compute_local_grads(self):
        """
        Computes an iter of grads based on local model
        """
        if self.verbose:
            print(f'Computing local grad for {str(self.id)}')
        self.optimizer.zero_grad()
        for batch_idx, (input, labels) in enumerate(self.dataloader):
            output = self.model(input)
            loss = self.loss(output, labels)
            loss = loss / len(self.dataloader)  # normalize for accumulation
            loss.backward()

    def step_update(self):
        """
        Update new gradients
        """
        self.optimizer.step()


n_epochs = 3
batch_size = 64
learning_rate = 1e-3

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model_p1 = Net()
model_p2 = Net()

# Start training steps here
p1 = Participant(id=1)
p1.init_model(model=model_p1)
p1.init_dataloader(train_loader)
p1.init_optim(lr=learning_rate)
p1.init_loss()

p2 = Participant(id=2)
p2.init_model(model=model_p2)
p2.init_dataloader(train_loader)
p2.init_optim(lr=learning_rate)
p2.init_loss()

# Compute local grads
p1.compute_local_grads()
p2.compute_local_grads()

# Add grads and update grads in models
# https://discuss.pytorch.org/t/manually-modify-gradients-of-two-models-average-them-and-put-them-back-in-both-models/80786
listGradA = []
listGradB = []
listGradAUpdated = []
listGradBUpdated = []
itr = 0

for pA, pB in zip(p1.model.parameters(), p2.model.parameters()):
    listGradA.append(pA)
    listGradB.append(pB)
    sum = pA.grad.clone() + pB.grad.clone()
    pA.grad = sum.clone()
    pB.grad = sum.clone()
    itr += 1

# Now
p1.step_update()
p2.step_update()

# %%
