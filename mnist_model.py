import tensorflow as tf

class MNIST_Model(tf.keras.Model): 
    def __init__(self): 
        super(MNIST_Model, self).__init__()
        # initialize hyperparameters 
        self.fc1_unit = 784 
        self.num_classes = 10
        self.batch_size = 100
        
        # initialize trainable parameters 
        self.fc1 = tf.keras.layers.Dense(
            self.fc1_unit, 
            activation = 'relu'
        )
        self.fc2 = tf.keras.layers.Dense(
            self.num_classes, 
            activation = 'softmax' 
        )

        return  

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
        