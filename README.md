Frontend url CityFlow simulator: https://importly.github.io/CityFlow/frontend/

Setup:
1. Windows/Ubuntu
Clone the Repository
- git clone https://github.com/safaminhaj/rl-traffic-control.git
- cd rl-traffic-control

Install Conda (Recommended), If conda is not installed, install Miniconda:

- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- bash Miniconda3-latest-Linux-x86_64.sh
Restart your terminal, then create and activate the environment:

- conda create -n cityflow python=3.10 -y
- conda activate cityflow

Install System Dependencies
- sudo apt update
- sudo apt install -y cmake g++ make git python3-pip

Install CityFlow from Source
CityFlow must be installed manually.
- cd CityFlow
- python -m pip install .
- cd ..

Test installation:
- python -c "import cityflow; print('CityFlow installed correctly')"

Install Python Dependencies
- pip install torch numpy stable-baselines3 gymnasium

Verify the Simulation Setup (Smoke Test)
Run the provided test script:
- python test_cityflow.py

Run Baseline Controllers
Random Agent
- python run_random_agent.py
- python run_baseline.py

Train the Custom DQN Agent
- python train_dqn.py

After training, the model is saved as: "dqn_cityflow.pt"

Evaluate the trained model:
- python evaluate_dqn.py

Train SB3 Agents
- python train_sb3_dqn.py
- python eval_sb3_dqn.py
- python train_sb3_ppo.py
- python eval_sb3_ppo.py

Model files are saved automatically under the project directory.
Visualize Simulation in CityFlow Viewer
CityFlow can generate replay logs:

- cityflow_scenario/replay.txt
- cityflow_scenario/replay_roadnet.json

To visualize open CityFlow’s web UI (index.html) locally in a browser
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

On Mac, we ran CityFlow with Docker.

- docker run -it --name cityflow-rl \
  -v /Users/saima/Downloads/rl-traffic-control:/workspace \
  -w /workspace \
  cityflowproject/cityflow:latest \
  bash

Then we start the container
- docker start -ai cityflow-rl

Install the dependencies:
- python -m pip install --upgrade "pip<21"
- /opt/conda/bin/python -m pip install --upgrade pip
- pip install "numpy==1.19.5" "matplotlib==3.3.4" "tqdm==4.64.0"
- pip install "torch==1.10.2"
- python -c "import cityflow, numpy, matplotlib, torch, tqdm; print('ALL OK')"

Then follow the same steps as described in Windows/Ubuntu setup 

