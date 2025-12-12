Setup:
1. Windows/Ubuntu
1. Clone the Repository
git clone https://github.com/safaminhaj/rl-traffic-control.git
cd rl-traffic-control
2. Install Conda (Recommended)

If conda is not installed, install Miniconda:

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh


Restart your terminal, then create and activate the environment:

conda create -n cityflow python=3.10 -y
conda activate cityflow


Python 3.12 is not supported because CityFlow’s pybind11 bindings require Python 3.10 or 3.11.
3. Install System Dependencies
sudo apt update
sudo apt install -y cmake g++ make git python3-pip
4. Install CityFlow from Source

CityFlow must be installed manually.

cd CityFlow
python -m pip install .
cd ..


Test installation:

python -c "import cityflow; print('CityFlow installed correctly')"
5. Install Python Dependencies
pip install torch numpy stable-baselines3 gymnasium
6. Verify the Simulation Setup (Smoke Test)

Run the provided test script:

python test_cityflow.py
7. Run Baseline Controllers
Random Agent
python run_random_agent.py

Fixed-Time Baseline
python run_baseline.py


These provide reference performance for comparison with RL agents.
8. Train the Custom DQN Agent
python train_dqn.py


After training, the model is saved as:

dqn_cityflow.pt

Evaluate the trained model:
python evaluate_dqn.py
9. Train SB3 Agents
SB3 DQN
python train_sb3_dqn.py
python eval_sb3_dqn.py

SB3 PPO (Best Performing Agent)
python train_sb3_ppo.py
python eval_sb3_ppo.py


Model files are saved automatically under the project directory.
10. Visualize Simulation in CityFlow Viewer

CityFlow can generate replay logs:

cityflow_scenario/replay.txt
cityflow_scenario/replay_roadnet.json


To visualize:

Open CityFlow’s web UI (index.html) locally in a browser

Upload both files

Watch the animated simulation of the intersection under RL control
11. Project Structure
rl-traffic-control/
│── CityFlow/                   # CityFlow engine (compiled here)
│── cityflow_scenario/          # Abu Dhabi intersection setup
│── test_cityflow.py            # Smoke test
│── cityflow_env.py             # Custom RL environment wrapper
│── run_random_agent.py         # Random baseline
│── run_baseline.py             # Fixed-time baseline
│── train_dqn.py                # Custom PyTorch DQN training
│── evaluate_dqn.py             # Custom DQN evaluation
│── train_sb3_dqn.py            # SB3 DQN agent
│── eval_sb3_dqn.py             # SB3 DQN evaluation
│── train_sb3_ppo.py            # SB3 PPO agent (best)
│── eval_sb3_ppo.py             # SB3 PPO evaluation
└── README.md

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



