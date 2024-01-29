from tensorflow import keras
from keras.datasets import mnist
from keras import layers
from keras.layers import Dense
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid','sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):
                                         
    """Define a dense model with a hidden layer with the following parameters:
    input_length: the number of inputs for the first layer
    activation_func_array: the activation function for the hidden layers
    hidden_layer_sizes: the number of neurons in each of the hidden layer
    output_function: the activation function for the output layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential()
    # Create the input layer
    model.add(Dense(units=hidden_layers_sizes[0], input_dim=input_length, activation=activation_func_array[0]))

    # Create the hidden layers
    for size, activation_func in zip(hidden_layers_sizes[1:], activation_func_array[1:]):
        model.add(Dense(units=size, activation=activation_func))

    # Create the output layer
    model.add(Dense(units=output_length, activation=output_function))

    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    """Set the layers of the model to trainable or not trainable.
    model: the model
    trainable_layer_numbers: the numbers of the layers to be set to trainable. set the other layers to not trainable
    """
    for i, layer in enumerate(model.layers):
        # Set the trainable attribute based on the layer number
        layer.trainable = i in trainable_layer_numbers
    return model
