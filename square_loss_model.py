import tensorflow as tf 

class Square_Loss_Optimizee(tf.keras.Model): 
    def __init__(self, size = 10, params = None): 
        '''
        :param size: the dimension of the vector theta, which is always equal to the dimension of output
        :param params: a dictionary for preset values for trainable params
        '''
        super(Square_Loss_Optimizee, self).__init__()
        
        # Hyperparameters
        self.size = size 

        # Trainable parameters 
        if params == None: 
            self.theta = tf.Variable(tf.random.normal((size,), mean = 0.0, stddev = 0.1, dtype = tf.float32))
        else: 
            self.theta = params['theta']

        return 

    def call(self, input = None): 
        return self.theta 

    def get_params(self): 
        # TODO: This is incorrect!! We need a dictionary with one 
        #       Key-Value pair per COORDINATE of theta
        return {'theta': self.theta} 

    def loss_function(self, predictions, loss_computer): 
        '''
        :param predictions: the theta returned by model.call 
        :param loss_computer: a Square_Loss object 
        :return the loss: |W theta - y|^2 
        '''
        return loss_computer.compute_loss(predictions)

class Square_Loss(): 
    def __init__(self, W, b): 
        '''
        :param W: an [n, n] tensor 
        :param b: an [n] tensor 
        '''
        self.W = W 
        self.b = b

    def compute_loss(self, theta): 
        diff = tf.matmul(self.W, theta) - self.W
        loss = tf.einsum('i,i', diff, diff) 
        return loss 