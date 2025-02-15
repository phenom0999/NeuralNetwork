from NeuralNetwork import NeuralNetwork

def main():
    nn = NeuralNetwork(2,2,1)
    inputs = [1,2]

    outputs = nn.feedforward(inputs)

    print(outputs)
    

if __name__ == "__main__":
    main()