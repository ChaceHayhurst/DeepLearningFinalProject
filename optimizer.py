import tensorflow as tf

class RNN_Optimizer(tf.keras.Model): 
    def __init__(self, units, unroll_factor = 20, meta_learning_rate = 0.01): 
        '''
        :param units: a list of three ints [lstm1_units, lstm2_units, dense_units]
        '''
        super(RNN_Optimizer, self).__init__()
        # TODO: Initialize Hyperparameters 
        self.layer1_units = units[0]
        self.layer2_units = units[1]
        self.dense_units = units[2]
        self.unroll_factor = unroll_factor
        
        # TODO: Initialize Trainable Parameters 
        self.lstm1 = tf.keras.layers.LSTM(
            units = self.layer1_units, 
            return_state = True, 
            return_sequences = True
        )
        self.lstm2 = tf.keras.layers.LSTM(
            units = self.layer2_units, 
            return_state = True, 
            return_sequences = False
        )
        self.dense1 = tf.keras.layers.Dense(
            units = self.dense_units, 
            activation = 'relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units = 1
        )

        self.learning_rate = meta_learning_rate
        self.adam_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        return  

    def call(self, optimizee_grads, initial_state_for_1, initial_state_for_2): 
        '''
        Given a sequence of optimizee gradients for one optimizee trainable parameter
        Tell how the optimizee parameter should update 

        :param optimizee_grads: optimizee gradients for one optimizee trainable param
                                [param_shape]
        :param initial_states_for_1: initial states for lstm1. A list containing
                                [initial_hidden_state, initial_cell_state]
        :param initial_states_for_1: initial states for lstm2. A list containing
                                [initial_hidden_state, initial_cell_state]
        :return optimizee_updates: optimizee param update for one optimizee trainable param
                                [param_shape]
                new_states_for_1: a list containing 
                                [new_hidden_state_1, new_cell_state_1]
                new_states_for_2: a list containing 
                                [new_hidden_state_2, new_cell_state_2]
        '''
        # TODO: implement this 
        original_gradients_shape = optimizee_grads.shape 
        lstm_input = tf.reshape(optimizee_grads, shape = (-1, 1, 1))
        
        lstm1_output, new_hidden_state_1, new_cell_state_1 = self.lstm1(lstm_input, initial_state = initial_state_for_1) 
        lstm2_output, new_hidden_state_2, new_cell_state_2 = self.lstm2(lstm1_output, initial_state = initial_state_for_2)
        
        dense1_output = self.dense1(lstm2_output) 
        dense2_output = self.dense2(dense1_output)

        result = tf.reshape(dense2_output, shape = original_gradients_shape)

        return result, [new_hidden_state_1, new_cell_state_1], [new_hidden_state_2, new_cell_state_2]
