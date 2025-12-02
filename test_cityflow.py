import cityflow
import os

"""
This script is a smoke test — its purpose is only to verify that CityFlow loads correctly, the config and roadnet files are working, and that the simulator runs for several steps without crashing.

This file does not involve RL at all.
It purely tests the simulator.
"""

# Path to the config we copied
CONFIG_PATH = os.path.join("cityflow_scenario", "config.json")


def main():
    # Create the engine
    eng = cityflow.Engine(CONFIG_PATH, thread_num=1)

    steps = 300  # run 300 simulation steps just as a smoke test

    for step in range(steps):
        eng.next_step()

        # Every 20 steps, print some basic info
        if step % 20 == 0:
            t = eng.get_current_time()
            waiting = eng.get_lane_waiting_vehicle_count()
            total_waiting = sum(waiting.values())
            print(f"step={step}, sim_time={t}, total_waiting_vehicles={total_waiting}")

    print("Simulation finished successfully.")


if __name__ == "__main__":
    main()
