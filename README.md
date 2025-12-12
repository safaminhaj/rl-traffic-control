Setup:
1. Windows/Ubuntu
2. Mac


SB3 DQN Agent
The parameters defined and what they correspond to is given below for understanding the agent and for further modifications:
a. Replay buffer size: buffer_size = 50_000, stores up to 50k transitions (s,a,r,s') enabling decorrelated updates and more stable learning.
b. Warmup: learning_starts = 1_000, for the first 1000 steps, the agent only explores and fills the buffer without updating the network. This prevents learning from a tiny, highly correlated dataset.
c. Batch size: batch_size = 64, each optimization step samples 64 transitions, balancing stability and efficiency. 
d. Training frequency: train_freq = 1, gradient_steps = 1, one gradient update per environment step, which is a standard choice in DQN.
e. Discount factor: gamma = 0.99, slightly prioritizes long-term reduction of congestion over immediate gain, appropriate for traffic where current signal choices affect downstream accumulation.
f. Target network update interval: target_update_interval = 2_000, every 2000 steps, the weights of the target Q-network are updated from the online network. This decoupling helps stabilize learning and prevents divergence.
g. exploration_initial_eps = 1.0, the agent initially acts fully at random, which encourages broad exploration.
h. exploration_final_eps = 0.05, epsilon decays to 0.05, meaning that even late in training, 5% of actions are still random, helping avoid overfitting to a narrow pattern.
i. exploration_fraction = 0.3, epsilon is linearly decayed over the first 30% of training steps (i.e., over 60,000 of 200,000 timesteps).
j. total_timesteps = 200_000, since each environment step corresponds to a 10-second control interval, this budget covers a large number of simulated episodes and exposes the agent to diverse traffic conditions.



