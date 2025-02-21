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
GRID_SIZE = 100    # 100 cells per row and column
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Each cell's size

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("XOR Neural Network")

nn = NeuralNetwork(2,10,1)

def draw_grid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            """ Generate the color for each grid after training the Neural Network """
            x1 = col/GRID_SIZE
            x2 = row/GRID_SIZE
            inputs = [x1, x2]
            y = nn.feedforward(inputs)
            random_int = y[0] * 255
            random_color = (random_int, random_int, random_int)

            
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, random_color, rect)

def main():
    # Run training iterations (Consider reducing the frequency)
    for _ in range(5000):
        data = random.choice(training_data)
        nn.train(data["inputs"], data["targets"])
    
    print(nn.feedforward([1,1]))
    print(nn.feedforward([1,0]))
    print(nn.feedforward([0,0]))
    print(nn.feedforward([0,1]))

    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear the screen with white
        draw_grid()                   # Draw the grid with random colors
        pygame.display.flip()         # Update the display

       

    pygame.quit()  # Quit Pygame after the main loop ends

if __name__ == "__main__":
    main()
