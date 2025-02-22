from NeuralNetwork import NeuralNetwork
import random
import pygame

# Training data (example for XOR)
training_data = [
    {"inputs": [0, 1], "targets": [1]},
    {"inputs": [1, 0], "targets": [1]},
    {"inputs": [0, 0], "targets": [0]},
    {"inputs": [1, 1], "targets": [0]}
]

pygame.init()

WINDOW_SIZE = 600  # Window dimensions: 600x600 pixels
GRID_SIZE = 2   # 100 cells per row and column
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Each cell's size

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("XOR Neural Network")


nn = NeuralNetwork(2,[20,20],1)
nn.set_learning_rate(0.01)

def draw_grid():

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            """ Generate the color for each grid after training the Neural Network """
            x1 = col/(GRID_SIZE - 1)
            x2 = row/(GRID_SIZE - 1)
            inputs = [x1, x2]
            y = nn.feedforward(inputs)
            color_value = y[0] * 255
            color = (color_value, 0, color_value)

            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def main():    
    running = True
    i = 0
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        data = random.choice(training_data)
        nn.train(data["inputs"], data["targets"])
        print(i)
        i+= 1


        screen.fill((255, 255, 255))  # Clear the screen with white
        draw_grid()                   # Draw the grid with random colors
        pygame.display.flip()         # Update the display


    pygame.quit()  # Quit Pygame after the main loop ends

if __name__ == "__main__":
    main()
