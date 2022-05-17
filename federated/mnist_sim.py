"""
MNIST simulations CGD
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import threading


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
        self.loss = torch.nn.NLLLoss()

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
        train_losses = []
        for batch_idx, (input, labels) in enumerate(self.dataloader):
            output = self.model(input)
            loss = self.loss(output, labels)
            loss = loss / len(self.dataloader)  # normalize for accumulation
            loss.backward()
        grads = self.model.named_parameters()
        print(loss.item())
        return grads

    def step(self, grads=None):
        """
        Update step on gradients
        :param grads: update grads, to update confined model
        """
        if grads is not None:
            for name, param in grads:
                for name2, param2 in self.model.named_parameters():
                    if name == name2:
                        param2.grad = param.grad
        optimizer.step()
        optimizer.zero_grad()

    def init_all(self, model, dataloader, lr):
        self.init_model(model)
        self.init_optim(lr)
        self.init_loss()
        self.init_dataloader(dataloader)

    def get_model(self):
        return self.model

    def update_model(self, model):
        self.model = model

n_epochs = 3
batch_size_train = 64
learning_rate = 1e-3
log_interval = 10
device = "cpu"

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)


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


network = Net()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)


################ Single Participant training
"""
model_p1 = Net()

p1 = Participant(id=1)
p1.init_model(model_p1)
p1.init_dataloader(train_loader)
p1.init_loss()
p1.init_optim(lr=0.01)

for i in range(10):
    grads = p1.perform_iter()
    p1.update_model(grads)
print("done")
"""
################ Multi participant training Simple Addition
# Between three particints
model_p1 = Net()
model_p2 = Net()
model_p3 = Net()

p1 = Participant(id=1)
p1.init_all(model=model_p1, dataloader=train_loader, lr=learning_rate)

p2 = Participant(id=2)
p2.init_all(model=model_p2, dataloader=train_loader, lr=learning_rate)

p3 = Participant(id=3)
p3.init_all(model=model_p3, dataloader=train_loader, lr=learning_rate)


grad1 = p1.perform_iter()
grad2 = p2.perform_iter()
grad3 = p3.perform_iter()

mp1 = p1.get_model()
mp2 = p2.get_model()
mp3 = p3.get_model()

# Verbose verification
pa1grads = []
pa2grads = []
pa3grads = []
sumgrads = []

for pa1, pa2, pa3 in zip(mp1.parameters(), mp2.parameters(), mp3.parameters()):
    pa1grads.append(pa1)
    pa2grads.append(pa2)
    pa3grads.append(pa3)

    sum_grad = (pa1.grad + pa2.grad + pa3.grad)
    sumgrads.append(sum_grad)
    pa1.grad = sum_grad
    pa2.grad = sum_grad.clone()
    pa3.grad = sum_grad.clone()

p1.update_model(mp1)
p2.update_model(mp2)
p3.update_model(mp3)

#p1.step()
#p2.step()
#p3.step()
print("Done")

#%%
