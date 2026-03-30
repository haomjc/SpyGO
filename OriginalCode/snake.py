import pygame
import time
import random
import sys

# Initialize Pygame
pygame.init()

# Set up the game window
window_width = 800
window_height = 600
cell_size = 20
grid_width = window_width // cell_size
grid_height = window_height // cell_size

window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def draw_text(text, size, color, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(str(text), True, color)
    window.blit(text_surface, (x, y))

class Snake:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.length = 4
        self.direction = "right"
        self.body = [(10, 10)]
        self.last_update = time.time()
        self.update_speed(5)
        
    def update_speed(self, speed_factor):
        self.speed = 10 // (speed_factor + 2)
        
    def change_direction(self, direction):
        if self.direction != direction:
            self.direction = direction
            
    def move(self):
        current_time = time.time()
        if current_time - self.last_update >= self.speed:
            head = list(self.body[0])
            
            if self.direction == "up":
                head[1] -= 1
            elif self.direction == "down":
                head[1] += 1
            elif self.direction == "left":
                head[0] -= 1
            else:
                head[0] += 1
                
            # Check if snake hits itself or walls
            for body_part in self.body:
                if head == list(body_part):
                    return False
                    
            self.body.insert(0, tuple(head))
            self.last_update = current_time
            
            # If not growing, remove the last part to move forward
            if len(self.body) > self.length:
                self.body.pop()
                
        return True
        
    def grow(self):
        self.length += 1

def generate_food(snake_body):
    while True:
        food_pos = (random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))
        if food_pos not in snake_body:
            return food_pos

snake = Snake()
food = generate_food(snake.body)
score = 0

game_over = False
clock = pygame.time.Clock()

while not game_over:
    window.fill(BLACK)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake.direction != "down":
                snake.change_direction("up")
            elif event.key == pygame.K_DOWN and snake.direction != "up":
                snake.change_direction("down")
            elif event.key == pygame.K_LEFT and snake.direction != "right":
                snake.change_direction("left")
            elif event.key == pygame.K_RIGHT and snake.direction != "left":
                snake.change_direction("right")
                
    # Move the snake
    if not game_over:
        game_over = not snake.move()
        
    # Check if food eaten
    if snake.body[0] == food:
        score += 1
        snake.grow()
        food = generate_food(snake.body)
        
    # Draw snake
    for segment in snake.body:
        x = segment[0] * cell_size
        y = segment[1] * cell_size
        pygame.draw.rect(window, GREEN, (x, y, cell_size, cell_size))
        
    # Draw food
    x = food[0] * cell_size
    y = food[1] * cell_size
    pygame.draw.rect(window, RED, (x, y, cell_size, cell_size))
    
    # Update score display
    draw_text(score, 24, WHITE, 10, 10)
    
    # Draw grid lines
    for x in range(0, window_width, cell_size):
        pygame.draw.line(window, WHITE, (x, 0), (x, window_height))
    for y in range(0, window_height, cell_size):
        pygame.draw.line(window, WHITE, (0, y), (window_width, y))
        
    # Update display
    pygame.display.flip()
    
    # Game over condition
    if game_over:
        draw_text("Game Over! Score: {}".format(score), 48, RED, (window_width - 250)//2, (window_height - 100)//2)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    sys.exit()
    
    # Control game speed
    clock.tick(1000)

pygame.quit()
