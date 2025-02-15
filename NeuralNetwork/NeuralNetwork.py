from Matrix import Matrix
import math

def sigmoid(num):
    return 1 / (1 + math.exp(-num))

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

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
        hidden_values.map(sigmoid) # Pass the hidden layer values through the sigmoid function

        """ Calculations for the output layer """
        output_values = Matrix.multiply(self.weights_ho, hidden_values)
        output_values.add(self.bias_output)
        output_values.map(sigmoid)

        """ Convert the output_values back into an array """
        outputs = output_values.toArray()

        return outputs
    
    def train(self, inputs, targets):
        pass