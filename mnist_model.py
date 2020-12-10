import tensorflow as tf

class MNIST_Model(tf.keras.Model): 
    def __init__(self, params = None): 
        super(MNIST_Model, self).__init__()
        # initialize hyperparameters 
        self.fc1_unit = 784 
        self.num_classes = 10
        self.batch_size = 100
        
        # initialize trainable parameters

        if params == None:
            self.fc1 = tf.zeros((self.fc1_unit,), name = 'fc1')
            self.fc2 = tf.zeros((self.num_classes,), name = 'fc2')
        else:
            self.fc1 = params['fc1']
            self.fc2 = params['fc2']
        
        #self.fc1 = tf.keras.layers.Dense(
        #    self.fc1_unit, 
        #    activation = 'relu'
        #)
        #self.fc2 = tf.keras.layers.Dense(
        #    self.num_classes, 
        #    activation = 'softmax' 
        #)

        return  

    def get_params(self):
        return [('theta', self.theta)]

    def update_params(self, change_tensors): 
        '''
        :param change_tensors: a list containing just one tensor, which is to be added to self.theta
        '''
        self.theta += change_tensors['theta']

    def call(self, input): 
        '''
        Forward pass 
        :param input: shape = (batch_sz, input_sz)
        :return the probabilities of each label, shape = (batch_sz, num_classes)
        '''
        fc1_output = self.fc1(input) 
        fc2_output = self.fc2(fc1_output) 

        prbs = fc2_output 

        return prbs

    def loss_function(self, predictions, labels): 
        '''
        :param predictions: the outputs from model.call. shape = (batch_sz, num_classes)
        :param labels: the ground truth labels. shape = (batch_sz)
        :return The Sparse Categorical Cross Entropy Loss
        '''
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))
        
