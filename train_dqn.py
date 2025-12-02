import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cityflow_env import CityFlowSingleJunctionEnv


# --------- Q-network --------- #
"""
This is a fully connected neural network that approximates the Q-function:
Input: state vector of size state_dim (lane waiting counts).
Output: vector of size action_dim (one Q-value per possible traffic light phase).
Architecture:
Linear(state_dim → 128), ReLU
Linear(128 → 128), ReLU
Linear(128 → action_dim)
Action selection: take q_values.argmax(dim=1) to pick the action with max Q-value.
"""


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------- Replay buffer --------- #
"""
deque(maxlen=capacity) → when full, oldest entries are dropped automatically.
push stores one transition:
s: state (numpy array)
a: action (int)
r: reward (float)
s_next: next state (numpy array)
done: bool indicating episode termination

random.sample selects batch_size transitions uniformly at random.
zip(*batch) transposes list-of-tuples into tuple-of-lists.
map(np.array, ...) converts each into a numpy array.
Finally, converts each array into PyTorch tensors with appropriate dtypes:
s_batch: (batch_size, state_dim) float32
a_batch: (batch_size,) int64 (for indexing)
r_batch: (batch_size,) float32
s_next_batch: (batch_size, state_dim) float32
done_batch: (batch_size,) float32 (0 or 1)


Why replay buffer?
Breaks correlations between consecutive samples.
Reuses old experiences multiple times (sampled in random order).
Stabilizes DQN training compared to updating only on the most recent transition.
"""


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# --------- Training loop --------- #

"""
num_episodes: total episodes of training
gamma: discount factor 
batch_size: number of transitions per gradient update
lr: learning rate for Adam optimizer
epsilon_start, epsilon_end, epsilon_decay: parameters for ε-greedy exploration schedule
target_update_freq: how many steps between target network updates
"""


def train_dqn(
    num_episodes=200,
    gamma=0.99,
    batch_size=64,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5000,
    target_update_freq=1000,
):
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )

    state = env.reset()
    state_dim = state.shape[
        0
    ]  # state_dim: number of lanes used in state representation
    action_dim = (
        env.action_space_n
    )  # number of traffic light phases at the intersection.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Two networks:
    policy_net: Q-network being updated.
    target_net: Q-network used to compute target values.
    Initially, target_net is a clone of policy_net.
    target_net.eval() ensures it’s in evaluation mode (no gradients, consistent behavior).
    """
    """
    Why target network?
    Classic DQN trick to stabilize learning:
    Use a delayed copy of the Q-network to compute targets.
    So targets change more slowly.
    Reduces moving-target instability.
    optimizer: Adam with learning rate lr.
    replay_buffer: stores experience transitions while interacting with env.
    """
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    global_step = 0

    def epsilon_by_step(step):
        # simple exponential decay
        # So early training: heavy exploration.
        # Later: mostly exploitation.
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(
            -step / epsilon_decay
        )

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            global_step += 1
            eps = epsilon_by_step(global_step)

            # epsilon-greedy action selection
            if random.random() < eps:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q_values = policy_net(s_tensor)
                    action = int(q_values.argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # ---- update network ---- #
            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s_next_batch, done_batch = (
                    replay_buffer.sample(batch_size)
                )

                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                s_next_batch = s_next_batch.to(device)
                done_batch = done_batch.to(device)

                # Q(s,a)
                q_values = policy_net(s_batch)
                q_sa = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

                # target: r + gamma * max_a' Q_target(s', a') (1 - done)
                with torch.no_grad():
                    q_next = target_net(s_next_batch).max(dim=1)[0]
                    target = r_batch + gamma * q_next * (1.0 - done_batch)

                loss = nn.MSELoss()(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # ---- update target network ---- #
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(
            f"Episode {ep+1}/{num_episodes} "
            f"total_reward={total_reward:.2f}, epsilon={eps:.3f}"
        )

    # save trained weights
    torch.save(policy_net.state_dict(), "dqn_cityflow.pt")
    env.close()
    print("Training finished, model saved as dqn_cityflow.pt")


if __name__ == "__main__":
    train_dqn()
