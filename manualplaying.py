import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import ale_py
import time
import threading
from gymnasium.utils.play import play

class SlowBallWrapper(gym.Wrapper):
    def __init__(self, env, speed_factor=0.5):
        super().__init__(env)
        self.speed_factor = speed_factor
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        time.sleep(self.speed_factor)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

# Initialize and wrap the environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# Wrap the environment to slow the ball
env = SlowBallWrapper(env, speed_factor=0.02)  # Set the speed factor to reduce the speed

# Convert the action space to Discrete (since Pong has 6 actions but we want only one discrete action)
env = gym.wrappers.GrayScaleObservation(env)  # Convert the observations to grayscale for efficiency
env = gym.wrappers.ResizeObservation(env, 84)  # Resize for faster training
env = gym.wrappers.FrameStack(env, 4)  # Stack frames (a common practice in RL)

# Load the trained model (DQN agent for the opponent)
model = DQN.load("dqn_pong_model")

# Global variables to track whether the game is done and the current observation
done = False
obs = None

# Event to signal that the game is ready to start
game_ready = threading.Event()

# Function to handle AI control for the opponent's paddle
def ai_control():
    global done, obs
    game_ready.wait()  # Wait until the game is ready
    while not done:
        # Convert the LazyFrame observation to a numpy array
        obs_np = np.array(obs)  # Convert LazyFrame to numpy array
        action_opponent, _ = model.predict(obs_np, deterministic=True)
        # Take the AI action for the opponent
        obs, reward, done, truncated, info = env.step(action_opponent)
        time.sleep(0.05)  # Small delay to prevent excessive CPU usage

# Start AI control in a separate thread
control_thread = threading.Thread(target=ai_control)
control_thread.start()

# Function to handle manual control (player's input)
def manual_control():
    global done, obs
    obs, info = env.reset()  # Initialize the observation from the environment
    game_ready.set()  # Signal that the game is ready to start
    play(env, zoom=3)  # Use play for manual control

# Call the play function to start the game with manual control for the player
manual_control()

# Wait for the control thread to finish before closing the environment
control_thread.join()
env.close()
