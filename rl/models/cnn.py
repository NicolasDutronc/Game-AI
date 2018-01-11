import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self, action_space):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # 20
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # 9
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 7
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=action_space)
        # self.out = nn.Linear(in_features=256, out_features=action_space)
        # self.weights_init()
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.out(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def weights_init(self):
        for p in self.parameters():
            print(p)
            nn.init.xavier_normal(p, gain=nn.init.calculate_gain('relu'))


class DuelingCNNModel(nn.Module):

    def __init__(self, action_space):
        super(DuelingCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # 20
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # 9
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 7
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.fcValue = nn.Linear(in_features=7*7*64, out_features=512)
        self.fcAdvantage = nn.Linear(in_features=7*7*64, out_features=512)
        self.value = nn.Linear(in_features=512, out_features=1)
        self.advantages = nn.Linear(in_features=512, out_features=action_space)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        value = F.relu(self.fcValue(x))
        value = self.value(value)
        advantages = F.relu(self.fcAdvantage(x))
        advantages = self.advantages(advantages)
        out = value + (advantages - advantages.mean())
        return out
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def weights_init(self):
        for p in self.parameters():
            print(p)
            nn.init.xavier_normal(p, gain=nn.init.calculate_gain('relu'))

model = CNNModel(2)
print(model)