"""rl_rmr.py"""
import gym
from gym import spaces
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """LSTM model with MLP for NASDAQ prediction"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        
        # Define the MLP
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass"""
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        
        # Pass the output of LSTM through the MLP
        x = F.relu(self.fc1(lstm_out.view(len(x), -1)[0]))
        output = self.fc2(x)
        
        return output

class NasdaqEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, data):
        super(NasdaqEnv, self).__init__()

        # Define the action space and observation space
        self.action_space = spaces.Discrete(2)  # Example: increase, decrease, no change

        # Load the NASDAQ data
        self.data = data  # Replace with your own data file

        self.observation_space = spaces.Box(low=-5000, high=np.inf, shape=(15,))

        # Define other necessary variables and parameters
        self.current_step = None
        self.max_steps = len(self.data) - 1

    def render(self):
        """Render the environment"""
        # Implement your rendering logic here
        return None

    def reset(self):
        """Reset the environment to the initial state"""
        # Reset the environment to the initial state
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """Execute one time step within the environment"""
        # Execute the given action and return the next observation, reward, done, and info
        if self.current_step >= self.max_steps:
            raise ValueError("Episode is done")

        # Execute the action and update the environment state
        # Example: Update portfolio, calculate reward, etc.

        self.current_step += 1
        observation = self._get_observation()

        if action == 0:
            # predicted icrease
            if observation[-1] > 0:
                reward = 15
            else:
                reward = -10
        elif action == 1:
            # predicted decrease
            if observation[-1] < 0:
                reward = 15
            else:
                reward = -10

        done = self.current_step >= self.max_steps
        info = {}  # Additional information

        return observation, reward, done, info

    def _get_observation(self):
        # Get the current observation from the data
        observation = self.data.iloc[self.current_step].values
        return observation


def custom_loss(output, target):
    return torch.mean(output - target)

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df_stock_1 = df[df["stock_id"] == 100]
    df_stock_1.index = df_stock_1["time_id"]
    df_stock_1.drop(columns=["row_id", "time_id", "stock_id"], inplace=True)

    # env = NasdaqEnv(df_stock_1)

    # observation_ = env.reset()
    # DONE_ = False

    # while not DONE_:
    #     action_ = (
    #         env.action_space.sample()
    #     )  # Replace with your own action selection logic
    #     observation_, reward_, DONE_, info_ = env.step(action_)

    #     print("Observation:", observation_[-1], "Action:", action_, "Reward:", reward_)

    # Initialize the environment and the model
    env = NasdaqEnv(df_stock_1)
    model = LSTMModel(
        input_dim=env.observation_space.shape[0] - 1,
        hidden_dim=32,
        output_dim=env.action_space.n,
    )
    # loss_function = nn.MSELoss()
    loss_function = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    total_reward = 0
    # Training loop
    for episode in range(10):  # number of episodes
        observation = env.reset()
        done = False
        count = 0
        rewards = torch.tensor([], dtype=torch.float)
        zeroes = torch.tensor([], dtype=torch.float)
        while not done:
            count += 1
            # Prepare the observation
            observation = torch.tensor(observation, dtype=torch.float)
            # Forward pass
            action_pred = model(observation)
            # Take action
            action = torch.argmax(action_pred).item()
            observation, reward, done, _ = env.step(action)
            
            reward = torch.tensor([reward], dtype=torch.float)
            rewards = torch.cat((rewards, reward))
            zeroes = torch.cat((zeroes, torch.tensor([0], dtype=torch.float, requires_grad=True)))
            if count >= 5000:
                count = 0
                # Compute loss
                loss = loss_function(
                    zeroes,
                    rewards,
                )
                total_reward += loss
                print(action, float(observation[-1]), action_pred, reward, loss.item(), episode, total_reward)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clear the rewards tensor
                rewards = torch.tensor([], dtype=torch.float)
                zeroes = torch.tensor([], dtype=torch.float)

    # Test the model
    observation = env.reset()
    done = False
    total_test_reward = 0
    count = 0
    while not done:
        count += 1
        # Prepare the observation
        observation = torch.tensor(observation, dtype=torch.float)
        # Forward pass
        action_pred = model(observation)
        # Take action
        action = torch.argmax(action_pred).item()
        observation, reward, done, _ = env.step(action)
        total_test_reward += reward
    
    print("Total test reward:", total_test_reward)
    print(total_test_reward/ count)
    print(21)
