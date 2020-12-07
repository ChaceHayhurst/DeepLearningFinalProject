import tensorflow as tf 
import numpy as np
import copy 
from matplotlib import pyplot as plt

from square_loss_model import Square_Loss_Optimizee, Square_Loss
from mnist_model import MNIST_Model
from optimizer import RNN_Optimizer

#tf.compat.v1.disable_eager_execution()

def train(optimizee, optimizer, num_examples, train_inputs = None, train_labels = None, loss_computer = None): 
    unroll_factor = optimizer.unroll_factor
    # unroll_factor = 1
    batch_size = optimizee.batch_size 

    # Grab Optimizee Params ids for making the dicts 
    optimizee_params = optimizee.get_params() #NOTE: optimizee_params is a list of tuples (param_name, param_tensor)

    # Initialize hidden_states and cell_states 
    state_size_for_1 = optimizer.layer1_units
    state_size_for_2 = optimizer.layer2_units
    initial_states_for_1 = {
        param_name:[tf.zeros(param_tensor.shape.concatenate(state_size_for_1), dtype = tf.float32), tf.zeros(param_tensor.shape.concatenate(state_size_for_1), dtype = tf.float32)] for param_name, param_tensor in optimizee_params
    }
    initial_states_for_2 = {
        param_name:[tf.zeros(param_tensor.shape.concatenate(state_size_for_2), dtype = tf.float32), tf.zeros(param_tensor.shape.concatenate(state_size_for_2), dtype = tf.float32)] for param_name, param_tensor in optimizee_params
    } # NOTE: in the initial_states dicts, param_ids are hashed after passing to string form 

    # Variables for keeping track of historic optimizee losses
    all_optimizee_losses_ever = []

    optimizer_tape = tf.GradientTape(persistent = True) 
    optimizer_tape.__enter__() 
        
    optimizee_losses_sum = tf.Variable(tf.zeros(()), name = 'Optimizee Losses Sum')
    optimizer_tape.watch(optimizee_losses_sum)

    for batch in range(0, num_examples // batch_size): 
        print("Starting batch {}".format(batch))

        train_inputs_batch = train_inputs[batch*batch_size:batch*(batch_size+1)] if not train_inputs is None else None
        train_labels_batch = train_labels[batch*batch_size:batch*(batch_size+1)] if not train_labels is None else None

        # Optimizee Forward Pass 
        with tf.GradientTape() as optimizee_tape: 
            optimizee_output = optimizee.call(train_inputs_batch)
            loss = optimizee.loss_function(optimizee_output, train_labels_batch, loss_computer) 

        # Preparation for Backprop 
        optimizee_losses_sum.assign_add(loss)
        print("Optimizee Loss: {}".format(loss))
        all_optimizee_losses_ever += [float(loss)]

        new_states_for_1 = {
            param_name:[tf.zeros(param_tensor.shape.concatenate(state_size_for_1), dtype = tf.float32), tf.zeros(param_tensor.shape.concatenate(state_size_for_1), dtype = tf.float32)] for param_name, param_tensor in optimizee_params       
        }
        new_states_for_2 = {
            param_name:[tf.zeros(param_tensor.shape.concatenate(state_size_for_2), dtype = tf.float32), tf.zeros(param_tensor.shape.concatenate(state_size_for_2), dtype = tf.float32)] for param_name, param_tensor in optimizee_params
        }
        new_optimizee_params = {param_name:tf.identity(param_tensor, name = 'theta') for param_name, param_tensor in optimizee_params}
        optimizee_param_changes = {}

        # Coordinate-wise Backprop on Optimizee
        for param_name, param_tensor in optimizee_params: 
            gradient = optimizee_tape.gradient(loss, param_tensor)
            gradient = tf.stop_gradient(gradient) 

            param_initial_state_for_1 = initial_states_for_1[param_name] 
            param_initial_state_for_2 = initial_states_for_2[param_name]

            change, new_state_for_1, new_state_for_2 = optimizer.call(gradient, param_initial_state_for_1, param_initial_state_for_2)
            # NOTE: new_state_for_1 is [new_hidden_state_1, new_cell_state_1], same for 2

            new_states_for_1[param_name] = new_state_for_1 
            new_states_for_2[param_name] = new_state_for_2 
            
            optimizee_param_changes[param_name] = change
            new_optimizee_params[param_name] += change 

        # Backprop on Optimizer
        if (batch + 1) % unroll_factor == 0: 
            optimizer_params = optimizer.trainable_variables 
            print(optimizer_tape.watched_variables())

            optimizer_tape.__exit__(None, None, None)
            optimizer_gradients = optimizer_tape.gradient(optimizee_losses_sum, optimizer_params) 
            optimizer.adam_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_params))
            
            optimizee_losses_sum = tf.constant(0, dtype = tf.float32)
            #new optimizee
            if optimizee.__class__.__name__ == 'Square_Loss_Optimizee':
                optimizee = Square_Loss_Optimizee(params=new_optimizee_params)
            else:
                optimizee = MNIST_Model(params=new_optimizee_params) 
            #new cell and hidden states
            for param_name, _ in optimizee_params: #not sure if this is right
                [new_hidden_1, new_cell_1] = new_states_for_1[param_name]
                initial_states_for_1[param_name] = [tf.constant(new_hidden_1.numpy()), tf.constant(new_cell_1.numpy())]
                
                [new_hidden_2, new_cell_2] = new_states_for_2[param_name]
                initial_states_for_2[param_name] = [tf.constant(new_hidden_2.numpy()), tf.constant(new_cell_2.numpy())]             

            # New Optimizer Tape
            optimizer_tape = tf.GradientTape()
            optimizer_tape.__enter__()

        else:
            initial_states_for_1 = new_states_for_1
            initial_states_for_2 = new_states_for_2
            optimizee.update_params(optimizee_param_changes)
            
        visualize_train_loss(all_optimizee_losses_ever, [])

def test(optimizee, test_inputs, test_labels): 
    # TODO: Form Data 

    # TODO: Compute test loss

    pass 

def visualize_train_loss(optimizee_loss, optimizer_loss): 
    # TODO: Figure out what to do here... 
    x1 = np.arange(1, len(optimizee_loss)+1) 
    x2 = np.arange(1, len(optimizer_loss)+1)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Loss for optimizee, optimizer')
    ax1.plot(x1, optimizee_loss)
    ax2.plot(x2, optimizer_loss)

def visualize_test_loss(ADAM_trained_loss, optimizee_trained_loss):
    # TODO: Figure out what to do here... 
    x = np.arange(1, len(ADAM_trained_loss)+1)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss for Adam-trained model and Optimizee-trained model')
    plt.plot(x, ADAM_trained_loss, color='red')
    plt.plot(x, optimizee_trained_loss, color='green')
    plt.show()

def main(model_name): 
    # Grab Data 
    train_inputs = None
    test_inputs = None 
    train_labels = None 
    test_labels = None 
    train_num_examples = 0
    test_num_examples = 0

    if (model_name == "SIMPLE_SQUARE"): 
        train_num_examples = 1000
        test_num_examples = 1000

        train_W = tf.random.normal((2, 2), mean = 1.0, stddev = 1.0)
        train_b = tf.random.normal((2,), mean = 1.0, stddev = 1.0)
        train_loss_computer = Square_Loss(train_W, train_b)

        test_W = tf.random.normal((2, 2), mean = 1.0, stddev = 1)
        test_b = tf.random.normal((2,), mean = 1.0, stddev = 1)
        test_loss_computer = Square_Loss(train_W, train_b)

    if (model_name == "MNIST"): 
        # Load mnist data 
        mnist = tf.keras.datasets.mnist 
        (train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()
        
        # Normalize data 
        train_inputs, test_inputs = train_inputs / 255.0, test_inputs / 255.0 
        train_inputs = train_inputs.astype(np.float32)

        # Compute number of examples 
        train_num_examples = train_inputs.shape[0]
        test_num_examples = test_inputs.shape[0]

        # Reshape inputs 
        train_inputs = np.reshape(train_inputs, (train_num_examples, -1)) 
        test_inputs = np.reshape(test_inputs, (test_num_examples, -1))

    # Initialize several optimizee-optimizer pairs: 
    # one for our optimizer, several more for benchmark optimizers
    if (model_name == "SIMPLE_SQUARE"): 
        optimizee = Square_Loss_Optimizee(size = 2) 
    elif (model_name == "MNIST"): 
        optimizee = MNIST_Model()

    rnn_optimizer = RNN_Optimizer()

    # TODO: Train each optimizee-optimizer pair for several epochs 
    num_epochs = 10
    for i in range(0, num_epochs): 
        print("Starting epoch {}".format(i+1))
        train(
            optimizee, 
            rnn_optimizer, 
            train_num_examples, 
            loss_computer = train_loss_computer if model_name == "SIMPLE_SQUARE" else None, 
            train_inputs = train_inputs if model_name == "MNIST" else None, 
            train_labels = train_labels if model_name == "MNIST" else None 
        ) 

    # TODO: Test each model 

    # TODO: Visualize results 
    return  

if __name__ == "__main__": 
    main("SIMPLE_SQUARE")