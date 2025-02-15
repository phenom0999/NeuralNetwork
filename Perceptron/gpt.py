import pygame
import random
import sys
from perceptron import Perceptron

# --- Pygame Setup ---
width, height = 600, 600  # window size
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Perceptron Learning Visualization")
clock = pygame.time.Clock()

# --- Helper: Convert Cartesian coordinates to Pygame screen coordinates ---
def to_screen(x, y):
    """
    Converts Cartesian coordinates (with origin at center and y-axis going up)
    into screen coordinates (with origin at top-left and y-axis going down).
    """
    return int(width / 2 + x), int(height / 2 - y)

# --- Define the target function f(x) ---
def f(x):
    """
    A line: f(x) = m * x + b. Adjust m and b as desired.
    For this example, we use m = 0.5 and b = 0.
    """
    m = -0.5
    b = 250
    return m * x + b

# --- Generate Training Data ---
# Create a list of random points in the Cartesian range.
# Our window’s Cartesian coordinate ranges roughly from -width/2 to width/2.
num_points = 10000
training = []
for _ in range(num_points):
    x = random.uniform(-width / 2, width / 2)
    y = random.uniform(-height / 2, height / 2)
    training.append([x, y])

# --- Create a Perceptron ---
# We have two inputs (x and y) and one bias, so we need 3 weights.
perceptron = Perceptron(2, lr=0.00001)

# --- Main Loop Variables ---
count = 0  # used to index the training points

# --- Main Loop ---
running = True
while running:
    print(-perceptron.weights[0]/perceptron.weights[1], -perceptron.bias/perceptron.weights[1], perceptron.bias, -perceptron.weights[1] * 100)
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Draw Background ---
    screen.fill((255, 255, 255))  # white background

    # --- Draw the Target Line ---
    # The line is drawn from x = -width/2 to x = width/2 using f(x)
    start = to_screen(-width / 2, f(-width / 2))
    end = to_screen(width / 2, f(width / 2))
    pygame.draw.line(screen, (0, 0, 0), start, end, 2)  # black line, thickness 2

    # --- Train the Perceptron on One Data Point at a Time ---
    # Get the current training point.
    x, y = training[count]
    # Determine the desired output:
    # If the point lies above the line (y > f(x)), desired output is 1;
    # otherwise, it's -1.
    desired = 1 if y > f(x) else -1
    perceptron.train([x, y], desired)
    # Update count for next frame
    count = (count + 1) % len(training)

    # --- Draw All Training Points ---
    for data in training:
        x_pt, y_pt = data
        # The perceptron’s guess will be 1 or -1.
        guess = perceptron.feed_forward([x_pt, y_pt])
        # Color the point based on the guess.
        # (127, 127, 127) for guess > 0, (255, 255, 255) for guess <= 0.
        color = (127, 127, 127) if guess > 0 else (255, 255, 255)
        # Convert Cartesian coordinate to screen coordinate.
        pos = to_screen(x_pt, y_pt)
        # Draw a circle with radius 4 (diameter ~8) and a black border.
        pygame.draw.circle(screen, color, pos, 4)
        pygame.draw.circle(screen, (0, 0, 0), pos, 4, 1)

    # --- Update the Display ---
    pygame.display.flip()
    clock.tick(60)  # run at 60 frames per second

pygame.quit()
sys.exit()
