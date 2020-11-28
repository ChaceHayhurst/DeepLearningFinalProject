class rnn_optimizer: 
    def __init__(self): 
        # TODO: Initialize Hyperparameters 

        # TODO: Initialize Trainable Parameters 

        # TODO: Initialize States
        self.hidden = None 

        pass 

    def call_rnn(self, optimizee_grads): 
        '''
        Given a sequence of optimizee gradients for one optimizee trainable parameter
        Tell how the optimizee parameter should update 

        :param optimizee_grads: (a sequence of) optimizee gradients for one optimizee trainable param
                                [window_sz, param_shape]
        :return optimizee_updates: (a sequence of) optimizee param update for one optimizee trainable param
                                [window_sz, param_shape]
        '''
        # TODO: implement this 

        pass 

    def apply_gradients(self, grads_and_vars): 
        '''
        Use call_rnn to determine updates for optimizee params
        Then applies these updates 

        :param grads_and_vars: a zipped list of optimizee gradients and optimizee params 
        :return None 
        '''
        # TODO: implement this 

        pass 

    def loss_function(self, optimizee_losses): 
        '''
        Given optimizee_losses over time, calculate 
        optimizer loss 

        :param optimizee_losses: a list of optimizee losses over time 
        :return optimizer_loss: a scalar 
        '''
        #TODO: implement this 

        pass 
