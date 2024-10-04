import pygame
import random
import numpy as np
import tensorflow as tf
from collections import deque
import os
import sys

# Set TensorFlow log level to suppress warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
WIDTH = 600
HEIGHT = 400
BLOCK_SIZE = 20
SPEED = 1000

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE  = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()
pygame.display.set_caption('AI Snake Game')
font = pygame.font.SysFont('arial', 25)

# Game Window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Model File Path
MODEL_PATH = 'model/snake_ai_model.h5'

# Directions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

# Agent Class
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.trainer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(11,)),  # Added Input layer
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')
        ])
        model.compile(optimizer=self.trainer, loss='mean_squared_error')
        return model


    def get_state(self, game):
        head = game.snake[0]
        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)

        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food[0] < game.snake[0][0],  # food left
            game.food[0] > game.snake[0][0],  # food right
            game.food[1] < game.snake[0][1],  # food up
            game.food[1] > game.snake[0][1],  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_step(self, states, actions, rewards, next_states, dones):
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Predict Q values for current states
        with tf.GradientTape() as tape:
            q_preds = self.model(states)
            q_targets = q_preds.numpy()
            q_next = self.model(next_states)

            for idx in range(len(dones)):
                if dones[idx]:
                    q_targets[idx][np.argmax(actions[idx])] = rewards[idx]
                else:
                    q_targets[idx][np.argmax(actions[idx])] = rewards[idx] + self.gamma * np.amax(q_next[idx])

            loss = tf.keras.losses.MSE(q_preds, q_targets)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.trainer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state):
        # Random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.n_games  # Example: adjust epsilon
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self.model.predict(state.reshape(1, -1))
            move = np.argmax(prediction[0])
            final_move[move] = 1

        return final_move

# Game Class
class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.head = (self.w/2, self.h/2)
        self.snake = [self.head,
                    (self.head[0]-BLOCK_SIZE, self.head[1]),
                    (self.head[0]-(2*BLOCK_SIZE), self.head[1])]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = (x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input (event handling)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Move
        self.move(action)  # Update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self.update_ui()
        clock.tick(SPEED)
        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def update_ui(self):
        screen.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(screen, BLUE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, YELLOW, pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))

        pygame.draw.rect(screen, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        screen.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):
        # [straight, right, left]

        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # No change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Right turn r -> d -> l -> u
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Left turn r -> u -> l -> d
            next_idx = (idx -1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head[0]
        y = self.head[1]
        if self.direction == RIGHT:
            x += BLOCK_SIZE
        elif self.direction == LEFT:
            x -= BLOCK_SIZE
        elif self.direction == DOWN:
            y += BLOCK_SIZE
        elif self.direction == UP:
            y -= BLOCK_SIZE

        self.head = (x, y)

# Training Function
def train():
    agent = Agent()
    game = SnakeGameAI()

    # Could be useful for plotting
    scores = []
    mean_scores = []
    total_score = 0

    # Track the highest score
    record = 0

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(MODEL_PATH)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

# Play Function
def play():
    agent = Agent()
    if os.path.exists(MODEL_PATH):
        agent.model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No model found. Please train the model first.")
        return

    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)

        # Optional: Display the "brain" (decision-making process)
        # For example, show the Q-values
        prediction = agent.model.predict(state_old.reshape(1, -1))
        print("Q-values:", prediction[0])

        if done:
            game.reset()

# Main Function
if __name__ == '__main__':
    mode = input("Enter 'train' to train the model or 'play' to watch the AI play: ").strip().lower()
    if mode == 'train':
        train()
    elif mode == 'play':
        play()
    else:
        print("Invalid input. Please enter 'train' or 'play'.")
