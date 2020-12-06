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
            self.theta = tf.Variable(tf.zeros((self.size,)), name = 'Theta')
        else: 
            self.theta = params[0]

        return 

    def get_param_indices(self): 
        return [[0, i] for i in range(0, self.size)]

    def get_param_shapes(self): 
        return [(self.size,)]

    def get_param_tensors(self): 
        return [self.theta]

    def update_params(self, change_tensors): 
        '''
        :param change_tensors: a list containing just one tensor, which is to be added to self.theta
        '''
        self.theta.assign_add(change_tensors[0])

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