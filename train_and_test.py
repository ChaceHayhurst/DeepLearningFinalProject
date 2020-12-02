import tensorflow as tf 
import numpy as np
from square_loss_model import SimpleModel 
from mnist_model import MNIST_Model
from matplotlib import pyplot as plt

def train(optimizee, optimizer, train_inputs, train_labels, num_examples): 
    # Grab metrics 
    # NOTE: All optimizees must take labels, but some take no inputs
    if (not train_inputs is None): assert(num_examples == train_inputs.shape[0])
    # If our model takes no inputs, set batch_sz to be 1
    batch_sz = optimizee.batch_size if (not train_inputs is None) else 1
    num_batches = num_examples // batch_sz 

    # Loop through batches to train 
    for i in range(0, num_batches): 
        if not train_inputs is None: 
            batch_inputs = train_inputs[i*batch_sz:(i+1)*batch_sz,]
        else: batch_inputs = None
        batch_labels = train_labels[i*batch_sz:(i+1)*batch_sz] 

        with tf.GradientTape() as tape: 
            preds = optimizee.call(batch_inputs) 
            optimizee_loss = optimizee.loss_function(preds, batch_labels)
        
        print("At batch {}. Optimizee loss = {}".format(i, optimizee_loss))
        optimizee_grads = tape.gradient(optimizee_loss, optimizee.trainable_variables)
        optimizer.apply_gradients(zip(optimizee_grads, optimizee.trainable_variables))

        # TODO: Insert optimizer training logic here!! 

    return  

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
        optimizee = SimpleModel(size = 2) 
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