import mxnet as mx


def get_symbol(action_space=None):
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
        net = mx.sym.LinearRegressionOutput(data=net)
        print(net.infer_shape(data=shape)[1][0])

        return net


model = get_symbol(2)

