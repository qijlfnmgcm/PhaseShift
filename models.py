import torch.nn as nn
import torch.nn.functional as F

image_size = 28 * 28
hidden = 2048


class deep_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, image_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class shallow_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_size, hidden)
        self.fc2 = nn.Linear(hidden, image_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
