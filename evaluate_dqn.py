import numpy as np
import torch

from cityflow_env import CityFlowSingleJunctionEnv
from train_dqn import DQN   # uses the same network architecture


def evaluate(num_episodes=20):
    # Create environment (same settings as training)
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300
    )

    # Get state and action dimensions
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space_n

    # Build DQN and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load("dqn_cityflow.pt", map_location=device))
    policy_net.eval()

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # convert state to tensor
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(s_tensor)
                action = int(q_values.argmax(dim=1).item())  # greedy action

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        print(f"[EVAL] Episode {ep}/{num_episodes} total_reward={total_reward:.2f}")

    avg_reward = float(np.mean(episode_rewards))
    print(f"\nAverage eval reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate(num_episodes=20)
