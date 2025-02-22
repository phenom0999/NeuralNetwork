from Matrix import Matrix
import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def dsigmoid(y):
    return y * (1 - y) # y is already passed though the sigmoid function

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.01

        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []

        # Create layer sizes including input, hidden layers and output
        layer_sizes = [self.input_nodes] + self.hidden_nodes + [self.output_nodes]

        # Initialize weights and biases for each pair of consecutive layers
        for i in range(len(layer_sizes) - 1):
            # Weight matrix between layer i and i + 1
            self.weights.append(Matrix(layer_sizes[i+1], layer_sizes[i], randomize = True))
            # Bias vector for layer i+1 (excluding input layer)
            self.biases.append(Matrix(layer_sizes[i+1], 1, randomize=True))

    def set_learning_rate(self, lr):
        self.learning_rate = lr
    
    def feedforward(self, input_array):

        """ Convert the inputs array into nx1 matrix """
        inputs = Matrix.toMatrix(input_array)
        current_activation = inputs 

        for i in range(len(self.weights)):
            # Compute weighted sum and apply activation function
            current_activation = Matrix.multiply(self.weights[i], current_activation)
            current_activation.add(self.biases[i])
            current_activation.map_n(sigmoid)
        
        return current_activation.toArray()


    def train(self, input_array, target_array):
        
        """ Convert the inputs array into nx1 matrix """
        inputs = Matrix.toMatrix(input_array) 
        targets = Matrix.toMatrix(target_array)

        # Store layer activations during forward pass
        activations = [inputs]
        current_activation = inputs

        # Forward pass through all layers
        for i in range(len(self.weights)):
            current_activation = Matrix.multiply(self.weights[i], current_activation)
            current_activation.add(self.biases[i])
            current_activation.map_n(sigmoid)
            activations.append(current_activation)
        
        # Calculate output error
        output_error = Matrix.subtract(targets, activations[-1])
        errors = output_error

        # Backpropagate through layers in reverse order
        for layer_idx in reversed(range(len(self.weights))):
            # Calculate gradient for current layer
            current_activation = activations[layer_idx + 1]
            gradient = Matrix.map(current_activation, dsigmoid)
            gradient.multiply_hadamard(errors)
            gradient.multiply_scalar(self.learning_rate)

            # Calculate weight deltas and update weights
            activation_prev = activations[layer_idx]
            activation_prev_t = Matrix.transpose(activation_prev)
            weight_deltas = Matrix.multiply(gradient, activation_prev_t)

            self.weights[layer_idx].add(weight_deltas)
            self.biases[layer_idx].add(gradient)

            # Calculate error for previous layer (if not input layer)
            if layer_idx > 0:
                weights_t = Matrix.transpose(self.weights[layer_idx])
                errors_prev = Matrix.multiply(weights_t, errors)

                # Multiply by derivative of previous layer's activation
                #deriv_prev = Matrix.map(activations[layer_idx], dsigmoid)
                #errors_prev.multiply_hadamard(deriv_prev)
                errors = errors_prev

