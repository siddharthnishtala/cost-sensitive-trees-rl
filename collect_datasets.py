import numpy as np
import pickle5
import gym
import os

from stable_baselines3.dqn.dqn import DQN

from utils import load_config, set_seed, get_env


env_name = "FourRooms"
dataset_sizes = [500, 1000, 2000]
model_dir = os.path.join("rl-trained-agents", "dqn")
dataset_dir = os.path.join("datasets", env_name)

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

config = load_config(env_name, model_dir)

set_seed(config["seed"])

env = get_env(env_name)

model = DQN.load(os.path.join(model_dir, env_name, "model"))

episode_rewards = []
state_list, action_list, q_values_list = [], [], []

for episode_no in range(1, dataset_sizes[-1] + 1):
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:

        action, _ = model.predict(state, deterministic=True)
        q_values = model.predict_q_values(state)

        next_state, reward, done, info = env.step(int(action))

        episode_reward += reward

        if env_name == "Taxi-v3":
            decoded_state = list(env.decode(state))
        else:
            decoded_state = state

        state_list.append(decoded_state)
        action_list.append([int(action)])
        q_values_list.append(q_values)

        state = next_state

    episode_rewards.append(episode_reward)

    print("Episode", str(episode_no), "| Total Reward:", str(episode_reward))

    if episode_no in dataset_sizes:
        dataset = (state_list, action_list, q_values_list)

        with open(os.path.join(dataset_dir, str(episode_no) + ".pkl"), "wb") as handle:
            pickle5.dump(dataset, handle)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(
            str(episode_no),
            "Episodes",
            "| Mean Reward:",
            str(mean_reward),
            "\u00B1",
            str(std_reward),
        )
