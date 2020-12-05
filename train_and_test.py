import tensorflow as tf 
import numpy as np
from square_loss_model import Square_Loss_Optimizee 
from mnist_model import MNIST_Model
from matplotlib import pyplot as plt

def train(optimizee, optimizer, num_examples, train_inputs = None, train_labels = None, loss_computer = None): 
    unroll_factor = optimizer.unroll_factor
    batch_size = optimizee.batch_size 

    # Grab Optimizee Params
    optimizee_params = optimizee.get_params()

    # Initialize hidden_states and cell_states 
    initial_states_for_1 = {
        param_id:[tf.constant([0.0]), tf.constant([0.0])] for param_id in optimizee_params.keys()
    }
    initial_states_for_2 = {
        param_id:[tf.constant([0.0]), tf.constant([0.0])] for param_id in optimizee_params.keys()
    }
    
    # Initialize the list containing historic optimizee losses
    optimizee_losses = []
    all_optimizee_losses_ever = []
    
    # Train Loop 
    for batch in range(0, num_examples // batch_size): 
        # TODO: Add logic for cases with train_input and train_labels
        train_inputs_batch = train_inputs[batch*batch_size:batch*(batch_size+1)]
        train_labels_batch = train_labels[batch*batch_size:batch*(batch_size+1)]

        # Optimizee Forward Pass 
        optimizee_params = optimizee.get_params() 
        with tf.GradientTape() as tape: 
            optimizee_param_tensors = optimizee.get_param_tensors() #different for MNIST?
            optimizee_output = optimizee.call(train_inputs_batch)
            loss = optimizee.loss_function(optimizee_output, train_labels_batch, loss_computer) #this would need to take in the labels, would it not?

        # Preparation for Backprop 
        optimizee_losses.append(loss)     
        new_states_for_1 = {
            param_id:[tf.constant([0.0]), tf.constant([0.0])] for param_id in optimizee_params.keys()
        }
        new_states_for_2 = {
            param_id:[tf.constant([0.0]), tf.constant([0.0])] for param_id in optimizee_params.keys()
        }

        # Coordinate-wise Backprop on Optimizee
        gradients = tape.gradient(loss, optimizee_param_tensors) #can we just call tape.gradient like this?
        for param_id in optimizee_params.keys(): 
            grad = tf.stop_gradient(gradients[param_id])
            param_initial_state_for_1 = initial_states_for_1[param_id] #identical naming before was confusing
            param_initial_state_for_2 = initial_states_for_2[param_id]

            change, new_state_for_1, new_state_for_2 = optimizer.call(grad, param_initial_state_for_1, param_initial_state_for_2)
            #Note new_state_for_1 is [new_hidden_state_1, new_cell_state_1], same for 2

            new_states_for_1[param_id] = new_state_for_1 
            new_states_for_2[param_id] = new_state_for_2 

            new_optimizee_params = optimizee_params.copy() #we may need copy.deepcopy
            new_optimizee_params[param_id] += float(change) #where are we keeping all these new_optimizee_params dictionaries?

        # Backprop on Optimizer
        if batch % unroll_factor == 0: 
            optimizer_loss = sum(optimizee_losses[-unroll_factor:]) #not sure what you were doing here, do we not just sum the entire list?
            optimizer_params = optimizer.trainable_variables #I assume this is all the params we want?
            optimizer_gradients = tape.gradient(optimizer_loss, optimizer_params) #but we can't use the same tape, right?
            optimizer.adam_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_params))
            
            all_optimizee_losses_ever.append(optimizee_losses) #just for the purposes of visualization
            optimizee_losses = []
            #new optimizee
            if optimizee.__class__.__name__ == 'Square_Loss_Optimizee':
                optimizee = Square_Loss_Optimizee(params=new_optimizee_params)
            else:
                optimizee = MNIST_Model(params=new_optimizee_params) #still need to implement manual param setting for MNIST
            #new cell and hidden states
            for param_id in optimizee_params.keys(): #not sure if this is right
                [new_hidden_1, new_cell_1] = new_states_for_1[param_id]
                initial_states_for_1[param_id] = [tf.constant(new_hidden_1.numpy()), tf.constant(new_cell_1.numpy())]
                
                [new_hidden_2, new_cell_2] = new_states_for_2[param_id]
                initial_states_for_2[param_id] = [tf.constant(new_hidden_2.numpy()), tf.constant(new_cell_2.numpy())]             
        else:
            initial_states_for_1 = new_states_for_1
            initial_states_for_2 = new_states_for_2
            
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
        train_y = tf.random.normal((2,), mean = 1.0, stddev = 1.0)
        train_labels = [(train_W, train_y) for i in range(0, train_num_examples)]

        test_W = tf.random.normal((2, 2), mean = 1.0, stddev = 1)
        test_y = tf.random.normal((2,), mean = 1.0, stddev = 1)
        test_labels = [(test_W, test_y) for i in range(0, test_num_examples)]

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

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01) 

    # TODO: Train each optimizee-optimizer pair for several epochs 
    num_epochs = 10
    for i in range(0, num_epochs): 
        print("Starting epoch {}".format(i+1))
        train(optimizee, adam_optimizer, train_inputs, train_labels, train_num_examples) 

    # TODO: Test each model 

    # TODO: Visualize results 
    return  

if __name__ == "__main__": 
    main("MNIST")