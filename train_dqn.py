import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cityflow_env import CityFlowSingleJunctionEnv


# --------- Q-network --------- #

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# --------- Replay buffer --------- #

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

def train_dqn(
    num_episodes=200,
    gamma=0.99,
    batch_size=64,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5000,
    target_update_freq=1000
):
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300
    )

    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    global_step = 0

    def epsilon_by_step(step):
        # simple exponential decay
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)

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
                    s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s_tensor)
                    action = int(q_values.argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # ---- update network ---- #
            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s_next_batch, done_batch = replay_buffer.sample(batch_size)

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
