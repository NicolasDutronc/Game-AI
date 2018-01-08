from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet import ndarray as nd


class Model(Block):

    def __init__(self, action_space, debug=False, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=32, kernel_size=8, strides=3)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels=64, kernel_size=5, strides=3)
            self.bn2 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels=64, kernel_size=3, strides=2)
            self.bn3 = nn.BatchNorm()
            self.fc1 = nn.Dense(units=512)         
            self.fc2 = nn.Dense(units=256)
            self.out = nn.Dense(units=action_space)
    
    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = nd.relu(self.fc1(x))
        x = nd.relu(self.fc2(x))
        return self.out(x)