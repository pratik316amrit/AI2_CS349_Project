import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import time
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize and wrap the environment
env = gym.make("ALE/Pong-v5",render_mode="human")

# Convert action space to Discrete (since Pong has 6 actions but we want only one discrete action)
# For Pong, typically you would reduce the multi-discrete action to one action: 0 (stay) or 1 (move up), etc.
env = gym.wrappers.GrayScaleObservation(env)  # Convert the observations to grayscale for efficiency
env = gym.wrappers.ResizeObservation(env, 84)  # Resize for faster training
env = gym.wrappers.FrameStack(env, 4)  # Stack frames (a common practice in RL)

# Convert the environment to a vectorized environment (required by SB3)
env = DummyVecEnv([lambda: env])

# Load the trained model
model = DQN.load("dqn_pong_model")

# Reset environment
obs = env.reset()
done = False


while not done:
    obs = np.array(obs)  # Convert LazyFrame to numpy array
    
    # Render the environment
    env.render()

    # Get the action for both paddles using the trained model
    action_left, _ = model.predict(obs, deterministic=True)  # Left paddle action
    action_right, _ = model.predict(obs, deterministic=True)  # Right paddle action

    # For Pong, we need to take one action each frame. Choose one paddle's action or alternate.
    # Here, you could alternate actions to simulate both paddles as AI
    action = action_left if np.random.rand() > 0.5 else action_right

    # Take the action and get feedback
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)  # Control speed to be visible

# Close the environment when done
env.close()
