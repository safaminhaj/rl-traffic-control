import matplotlib.pyplot as plt

# === 1. Fill these with your actual averages ===
custom_dqn_avg = -2259.05   # replace with your logged value
sb3_dqn_avg    = -1829.00   # replace with your logged value
sb3_ppo_avg    = -924.30 # replace with your logged value

agents = ["Custom DQN", "SB3 DQN", "SB3 PPO"]
avg_rewards = [custom_dqn_avg, sb3_dqn_avg, sb3_ppo_avg]

# Because rewards are negative (waiting time), we can also plot their absolute value
# if you want "higher = better". Comment out the next line if you prefer raw rewards.
# avg_rewards_plot = [-r for r in avg_rewards]  # convert to "effective score"
# ylabel = "Effective score (higher is better)"

avg_rewards_plot = avg_rewards
ylabel = "Average reward (less negative is better)"

plt.figure(figsize=(6, 4))
plt.bar(agents, avg_rewards_plot)
plt.xlabel("Agent")
plt.ylabel(ylabel)
plt.title("Traffic Signal Control: Agent Performance Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for i, v in enumerate(avg_rewards_plot):
    plt.text(i, v, f"{v:.0f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=9)

plt.tight_layout()
plt.savefig("agent_performance.png")
# plt.show()
