from Matrix import Matrix
import math

def sigmoid(num):
    return 1 / (1 + math.exp(-num))

def dsigmoid(y):
    return y * (1 - y) # y is already passed though the sigmoid function

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.1

        self.weights_ih = Matrix(hidden_nodes, input_nodes, randomize=True)
        self.weights_ho = Matrix(output_nodes, hidden_nodes, randomize=True)
        self.bias_hidden = Matrix(hidden_nodes, 1, randomize=True)
        self.bias_output = Matrix(output_nodes, 1, randomize=True)
    
    def feedforward(self, input_array):

        """ Convert the inputs array into nx1 matrix """
        inputs = Matrix.toMatrix(input_array) 

        """ Calculations for the hidden layer """
        hidden_values = Matrix.multiply(self.weights_ih, inputs) # Perform matrix multiplication to get the values in the hidden layer
        hidden_values.add(self.bias_hidden) # Add bias to the hidden layer values
        hidden_values.map_n(sigmoid) # Pass the hidden layer values through the sigmoid function

        """ Calculations for the output layer """
        output_values = Matrix.multiply(self.weights_ho, hidden_values)
        output_values.add(self.bias_output)
        output_values.map_n(sigmoid)

        """ Convert the output_values back into an array """
        outputs = output_values.toArray()

        return outputs
    
    def train(self, input_array, target_array):
        
        """ Convert the inputs array into nx1 matrix """
        inputs = Matrix.toMatrix(input_array) 
        targets = Matrix.toMatrix(target_array)

        """ Calculations for the hidden layer """
        hidden_values = Matrix.multiply(self.weights_ih, inputs) # Perform matrix multiplication to get the values in the hidden layer
        hidden_values.add(self.bias_hidden) # Add bias to the hidden layer values
        hidden_values.map_n(sigmoid) # Pass the hidden layer values through the sigmoid function

        """ Calculations for the output layer """
        output_values = Matrix.multiply(self.weights_ho, hidden_values)
        output_values.add(self.bias_output)
        output_values.map_n(sigmoid)

        output_errors = Matrix.subtract(targets, output_values)

        """ Calculate the hidden->output gradients"""
        gradients_ho = Matrix.map(output_values, dsigmoid)
        gradients_ho.multiply_hadamard(output_errors)
        gradients_ho.multiply_scalar(self.learning_rate)

        """ Calculate the deltas """
        hidden_T = Matrix.transpose(hidden_values)
        del_weights_ho = Matrix.multiply(gradients_ho, hidden_T)

        """ Update the hidden->output weights and biases """
        self.weights_ho.add(del_weights_ho)
        self.bias_output.add(gradients_ho)


        weights_ho_transpose = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiply(weights_ho_transpose, output_errors)

        """ Calculate the input->hidden gradients"""
        gradients_ih = Matrix.map(hidden_values, dsigmoid)
        gradients_ih.multiply_hadamard(hidden_errors)
        gradients_ih.multiply_scalar(self.learning_rate)

        """ Calculate the deltas """
        input_T = Matrix.transpose(inputs)
        del_weights_ih = Matrix.multiply(gradients_ih, input_T)

        """ Update the hidden->output weights """
        self.weights_ih.add(del_weights_ih)
        self.bias_hidden.add(gradients_ih)

        return None


