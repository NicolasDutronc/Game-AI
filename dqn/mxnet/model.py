import mxnet as mx
import numpy as np


def get_symbol(action_space=None, label_names=None):
    if action_space is None:
        print('Input the action space please')
        return None
    else:
        print('Shapes:')
        shape = (100, 1, 128, 128)

        # input data
        print('input:')
        net = mx.sym.var('data')
        print(net.infer_shape(data=shape)[1][0])

        # normalization
        net = mx.sym.BatchNorm(data=net)

        # convolution layer 1
        print('conv1:')
        net = mx.sym.Convolution(data=net, kernel=(8,8), stride=(3,3), num_filter=32)
        net = mx.sym.Activation(data=net, act_type='relu')
        print(net.infer_shape(data=shape)[1][0])

        # convolution layer 2
        print('conv2:')
        net = mx.sym.Convolution(data=net, kernel=(5,5), stride=(3,3), num_filter=64)
        net = mx.sym.Activation(data=net, act_type='relu')
        print(net.infer_shape(data=shape)[1][0])

        # convolution layer 3
        print('conv3:')
        net = mx.sym.Convolution(data=net, kernel=(3,3), stride=(2,2), num_filter=64)
        net = mx.sym.Activation(data=net, act_type='relu')
        print(net.infer_shape(data=shape)[1][0])

        # flatten
        print('flatten:')
        net = mx.sym.flatten(data=net)
        print(net.infer_shape(data=shape)[1][0])

        # hidden layer
        print('hidden layer:')
        net = mx.sym.FullyConnected(data=net, num_hidden=512)
        net = mx.sym.Activation(data=net, act_type='relu')
        print(net.infer_shape(data=shape)[1][0])

        # hidden layer
        print('hidden layer:')
        net = mx.sym.FullyConnected(data=net, num_hidden=256)
        net = mx.sym.Activation(data=net, act_type='relu')
        print(net.infer_shape(data=shape)[1][0])

        # output layer
        print('output layer:')
        net = mx.sym.FullyConnected(data=net, num_hidden=action_space)
        net = mx.sym.LinearRegressionOutput(data=net, name=label_names)
        print(net.infer_shape(data=shape)[1][0])

        return net

def update(model, batch, batch_size, episode_num, discount_factor):
    print('update network')
    x = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], batch[0][0].shape[2]))
    print('x shape:', x.shape)
    y = np.zeros((len(batch), 2))
    print('y shape:', y.shape)
    for i in range(len(batch)):
        current_obs = batch[i][0]
        action = batch[i][1]
        reward = batch[i][2]
        next_obs = batch[i][3]

        print(current_obs.shape)
        print(action)
        print(reward)
        print(next_obs.shape)
        print(type(current_obs))

        model.forward(current_obs)
        print(model.get_outputs())

        x[i:i+1] = current_obs
        y[i] = model.forward(current_obs)[0].asnumpy()
        next_Q = model.forward(next_obs)[0].asnumpy()

        if done:
            y[i, action] = reward
        else:
            y[i, action] = reward + discount_factor * np.max(next_Q)

    train_iter = mx.io.NDArrayIter(data=x, label=y, batch_size=batch_size, suffle=True)
    model.fit(
        train_data=train_iter,
        eval_metric='mse',
        optimizer='adagrad',
        optimizer_params={'learning_rate':0.01},
        num_epoch=10
    )
    model.save_params('checkpoint_{}'.format(episode_num))

'''
model = get_symbol(2)
#'''
