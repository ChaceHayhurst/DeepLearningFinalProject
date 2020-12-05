import tensorflow as tf

class rnn_optimizer: 
    def __init__(self): 
        # TODO: Initialize Hyperparameters 
        self.layer1_units = 1
        self.layer2_units = 1
        self.unroll_factor = 20
        
        # TODO: Initialize Trainable Parameters 
        self.lstm1 = tf.keras.layers.LSTM(
            units = self.layer1_units, 
            return_state = True, 
            return_sequences = False
        )
        self.lstm2 = tf.keras.layers.LSTM(
            units = self.layer2_units, 
            return_state = True, 
            return_sequences = False
        )

        return  

    def call(self, optimizee_grads, initial_states_for_1, initial_states_for_2): 
        '''
        Given a sequence of optimizee gradients for one optimizee trainable parameter
        Tell how the optimizee parameter should update 

        :param optimizee_grads: optimizee gradients for one optimizee trainable param
                                [param_shape (= 1)]
        :param initial_states_for_1: initial states for lstm1. A list containing
                                [initial_hidden_state, initial_cell_state]
        :param initial_states_for_1: initial states for lstm2. A list containing
                                [initial_hidden_state, initial_cell_state]
        :return optimizee_updates: optimizee param update for one optimizee trainable param
                                [param_shape (= 1)]
                new_states_for_1: a list containing 
                                [new_hidden_state_1, new_cell_state_1]
                new_states_for_2: a list containing 
                                [new_hidden_state_2, new_cell_state_2]
        '''
        # TODO: implement this 

        lstm1_output, new_hidden_state_1, new_cell_state_1 = self.lstm1.call(inputs, initial_state = initial_state_for_1) 
        lstm2_output, new_hidden_state_2, new_cell_state_2 = self.lstm2.call(lstm1_output, initial_state = initial_states_for_2)

        return lstm2_output, [new_hidden_state_1, new_cell_state_1], [new_hidden_state_2, new_cell_state_2]