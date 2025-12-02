from cityflow_env import CityFlowSingleJunctionEnv


def main():
    env = CityFlowSingleJunctionEnv(
        config_path="cityflow_scenario/config.json",
        intersection_id="intersection_1_1",
        action_duration=10,
        max_episode_steps=300,
    )

    num_episodes = 3

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        step = 0
        phase = 0

        print(f"\n=== Fixed-time Episode {ep+1} ===")

        done = False
        while not done:
            # Cycle through phases deterministically
            action = phase
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            step += 1

            # rotate phase
            phase = (phase + 1) % env.action_space_n

            if step % 5 == 0 or done:
                print(
                    f"step={step}, action={action}, reward={reward:.2f}, "
                    f"total_reward={total_reward:.2f}, sim_time={info['sim_time']}"
                )

            state = next_state

        print(f"Fixed-time Episode {ep+1} finished, total_reward={total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
