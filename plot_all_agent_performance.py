import matplotlib.pyplot as plt

# Average rewards from evaluation 
random_avg      = -2987.00
fixed_time_avg  = -2915.00
custom_dqn_avg  = -2259.05
sb3_dqn_avg     = -1829.00
sb3_ppo_avg     = -924.30

agents = [
    "Random",
    "Fixed-Time",
    "Custom DQN",
    "SB3 DQN",
    "SB3 PPO"
]

avg_rewards = [
    random_avg,
    fixed_time_avg,
    custom_dqn_avg,
    sb3_dqn_avg,
    sb3_ppo_avg
]

ylabel = "Average reward (less negative is better)"

plt.figure(figsize=(8, 5))
bars = plt.bar(agents, avg_rewards, color=["gray", "darkgray", "steelblue", "royalblue", "seagreen"])
plt.xlabel("Agent")
plt.ylabel(ylabel)
plt.title("Traffic Signal Control: Performance Comparison (All Agents)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Add labels above bars
for bar, value in zip(bars, avg_rewards):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.0f}",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("agent_performance_all_agents.png")
# plt.show()
