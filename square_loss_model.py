import tensorflow as tf 

class Square_Loss_Optimizee(tf.keras.Model): 
    def __init__(self, size = 2, params = None): 
        '''
        :param size: the dimension of the vector theta, which is always equal to the dimension of output
        :param params: a dictionary for preset values for trainable params
        '''
        super(Square_Loss_Optimizee, self).__init__()
        
        # Hyperparameters
        self.size = size 
        self.batch_size = 1

        # Trainable parameters 
        if params == None: 
            param_ids = range(0, size)
            self.params = {id:0.0 for id in param_ids} 
        else: 
            self.params = params

        self.theta = tf.Variable(list(self.params.values()))

        return 

    def get_params(self): 
        return self.params

    def get_param_tensors(self): 
        return [self.theta]

    def set_params(self, params): 
        self.params = params
        self.theta = tf.Variable(list(self.params.values()))

    def call(self, inputs): 
        '''
        :param inputs: None 
        :return theta
        '''
        return self.theta

    def loss_function(self, predictions, labels, loss_computer): 
        '''
        :param predictions: the theta returned by model.call 
        :param labels: None
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
        diff = tf.einsum('ij,j->i', self.W, theta) - self.b
        loss = tf.reduce_sum(diff * diff)
        return loss 