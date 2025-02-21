import pygame
import random

pygame.init()

WINDOW_SIZE = 600  # Window dimensions: 600x600 pixels
GRID_SIZE = 100    # 100 cells per row and column
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Each cell's size

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Random Color 100x100 Grid")

def draw_grid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Generate a new random color for each cell every frame
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, random_color, rect)

running = True
while running:
    screen.fill((255, 255, 255))  # Clear the screen with white
    draw_grid()                   # Draw the grid with random colors
    pygame.display.flip()         # Update the display

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
