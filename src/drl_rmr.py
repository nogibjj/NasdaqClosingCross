"""rl_rmr.py"""
import gym
from gym import spaces
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class LSTMModel(nn.Module):
    """LSTM model for NASDAQ prediction"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass"""
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        output = self.fc(lstm_out.view(len(x), -1)[0])
        return output


class NasdaqEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, data):
        super(NasdaqEnv, self).__init__()

        # Define the action space and observation space
        self.action_space = spaces.Discrete(3)  # Example: increase, decrease, no change

        # Load the NASDAQ data
        self.data = data  # Replace with your own data file

        self.observation_space = spaces.Box(low=-5000, high=np.inf, shape=(15,))

        # Define other necessary variables and parameters
        self.current_step = None
        # self.max_steps = len(self.data) - 1
        self.max_steps = 100

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
                reward = 1
            else:
                reward = -1
        elif action == 1:
            # predicted decrease
            if observation[-1] < 0:
                reward = 1
            else:
                reward = -1
        else:
            # predicted no change
            if observation[-1] == 0:
                reward = 2
            else:
                reward = -0.5

        done = self.current_step >= self.max_steps
        info = {}  # Additional information

        return observation, reward, done, info

    def _get_observation(self):
        # Get the current observation from the data
        observation = self.data.iloc[self.current_step].values
        return observation


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df_stock_1 = df[df["stock_id"] == 0]
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
        hidden_dim=1,
        output_dim=env.action_space.n,
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for episode in range(1000):  # number of episodes
        observation = env.reset()
        done = False
        while not done:
            # Prepare the observation
            observation = torch.tensor(observation, dtype=torch.float)
            # Forward pass
            action_pred = model(observation)
            # Take action
            action = torch.argmax(action_pred).item()
            observation, reward, done, _ = env.step(action)
            # Compute loss
            loss = loss_function(
                torch.tensor([0], dtype=torch.float, requires_grad=True),
                torch.tensor([reward], dtype=torch.float),
            )
            print(action, float(observation[-1]), action_pred, reward, loss.item(), episode)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test the model
    observation = env.reset()
    done = False
    while not done:
        # Prepare the observation
        observation = torch.tensor(observation, dtype=torch.float)
        # Forward pass
        action_pred = model(observation)
        # Take action
        action = torch.argmax(action_pred).item()
        observation, reward, done, _ = env.step(action)
        print("Observation:", observation[-1], "Action:", action, "Reward:", reward)

    print(21)
