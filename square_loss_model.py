import tensorflow as tf 

class SimpleModel: 
    def __init__(self, size = 10): 
        '''
        :param size: the dimension of the vector theta, which is always equal to the dimension of output
        '''
        self.size = size 
        self.theta = tf.Variable(tf.random.normal((size,), mean = 0.0, stddev = 0.1, dtype = tf.float32))

        return 

    def call(self, input = None): 
        return self.theta 

    def loss_function(self, predictions, labels): 
        '''
        :param predictions: the theta returned by model.call 
        :param labels: a tuple (W, y)
        :return the loss: |W theta - y|^2 
        '''
        W, y = labels
        diff = tf.matmul(W, predictions) - y 
        loss = tf.tensordot(diff, diff, 1)

        return loss 