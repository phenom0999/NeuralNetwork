import random

class Perceptron:
    def __init__(self, size, lr):
        # Initialize weights and bias with values centered around 0
        self.weights = [random.uniform(-1, 1) for _ in range(size)]
        self.learning_rate = lr
        self.bias = random.uniform(-1, 1)

    def feed_forward(self, inputs):
        total = self.bias
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]
        return self.activate(total)

    def activate(self, total):
        # Returns -1 for negative input, 1 for non-negative input
        return -1 if total < 0 else 1
    
    def train(self, inputs, desired):
        guess = self.feed_forward(inputs)
        error = desired - guess
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += inputs[i] * error * self.learning_rate
        
        # Update bias
        self.bias += error * self.learning_rate * 1000
