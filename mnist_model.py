import tensorflow as tf

class MNIST_Model(tf.keras.Model): 
    def __init__(self, params = None): 
        super(MNIST_Model, self).__init__()
        # initialize hyperparameters 
        self.fc1_unit = 100
        self.num_classes = 10
        self.batch_size = 100
        
        # initialize trainable parameters 
        if params == None:
            self.fc1 = tf.random.truncated_normal([784, self.fc1_unit], stddev=.1, dtype=tf.float32, name='fc1')
            self.fc2 = tf.random.truncated_normal([self.fc1_unit, self.num_classes], stddev=.1, dtype=tf.float32, name='fc2')
        else:
            self.fc1 = params['fc1']
            self.fc2 = params['fc2']

        return  
        
    def get_params(self): 
        return [('fc1', self.fc1), ('fc2', self.fc2)]

    def update_params(self, change_tensors): 
        '''
        :param change_tensors: a list containing just one tensor, which is to be added to self.theta
        '''
        self.fc1 += change_tensors['fc1']
        self.fc2 += change_tensors['fc2']

    def call(self, input): 
        '''
        Forward pass 
        :param input: shape = (batch_sz, input_sz)
        :return the probabilities of each label, shape = (batch_sz, num_classes)
        '''
        fc1_output = tf.linalg.matmul(input, self.fc1)
        fc1_output = tf.nn.relu(fc1_output)
        fc2_output = tf.linalg.matmul(fc1_output, self.fc2)
        fc2_output = tf.nn.softmax(fc2_output)
        
        return fc2_output

    def loss_function(self, predictions, labels, loss_computer = None): 
        '''
        :param predictions: the outputs from model.call. shape = (batch_sz, num_classes)
        :param labels: the ground truth labels. shape = (batch_sz)
        :return The Sparse Categorical Cross Entropy Loss
        '''
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))
        