class rnn_optimizer: 
    def __init__(self): 
        # TODO: Initialize Hyperparameters 

        # TODO: Initialize Trainable Parameters 

        pass 

    def call_rnn(self, optimizee_grads): 
        '''
        Given a sequence of optimizee gradients 
        Give how the optimizee parameters should update 

        :param optimizee_grads: (a sequence of) optimizee gradients 
                                [window_sz, optimizee_trainable_params]
        :return optimizee_updates: (a sequence of) optimizee update rules
                                [window_sz, optimizee_trainable_params_shape]
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
