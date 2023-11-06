import numpy as np
import highway_env
import pickle5
import random
import torch
import yaml
import gym
import os

from environments.wrappers import FlattenObservation
from environments.gridworld import FourRooms
from stable_baselines3 import DQN
from datetime import datetime


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_config(model_dir, config):

    with open(os.path.join(model_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(env, model_dir):

    path = os.path.join(model_dir, env, "config.yml")

    config = load_yaml(path)

    return config

def load_yaml(path):

    with open(path, "r") as f:
        file = yaml.safe_load(f)

    return file

def load_pickled_object(path):

    with open(path, "rb") as f:
        obj = pickle5.load(f)

    return obj

def create_results_directory(method, config, hyperparams, base_path=""):

    if not os.path.exists(os.path.join(base_path, "results", method)):
        os.makedirs(os.path.join(base_path, "results", method))

    dir_name = "_".join(
        [config["env"]]
        + [str(config[hyperparam]) for hyperparam in hyperparams]
        + [datetime.now().strftime("%H_%M_%S_%d_%b")]
    )
    datetime.now().strftime("%H_%M_%S_%d_%b")

    os.makedirs(os.path.join(base_path, "results", method, dir_name))

    with open(
        os.path.join(base_path, "results", method, dir_name, "config.yml"), "w"
    ) as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    config["results_dir"] = os.path.join(base_path, "results", method, dir_name)

    return config

def get_env(env_name):

    if env_name in ["LunarLander-v2", "Taxi-v3"]:
        env = gym.make(env_name)
    elif env_name == "highway-fast-v0":
        env = gym.make(env_name)
        env.configure({"observation": {"type": "TimeToCollision", "horizon": 3}})
        env = FlattenObservation(env)
    elif env_name == "FourRooms":
        env = FourRooms()
    else:
        raise ValueError("Environment not supported. Supported environments: [LunarLander-v2, Taxi-v3, FourRooms, highway-fast-v0].")

    return env

def get_env_details(env_name):

    if env_name == "LunarLander-v2":
        state_dim, num_actions = 8, 4
    elif env_name == "Taxi-v3":
        state_dim, num_actions = 4, 6
    elif env_name == "FourRooms":
        state_dim, num_actions = 4, 4
    elif env_name == "highway-fast-v0":
        state_dim, num_actions = 27, 5
    else:
        ValueError("Environment not supported. Supported environments: [CartPole-v1, MountainCar-v0, LunarLander-v2, Taxi-v3, FourRooms].")

    return state_dim, num_actions 

def load_dqn_model(env_name, env, seed=0):

    model_path = os.path.join("rl-trained-agents", "dqn", env_name, "model.zip")

    kwargs = dict(seed=seed)
    kwargs.update(dict(buffer_size=1))

    model = DQN.load(
        model_path, env=env, custom_objects={}, device="auto", **kwargs
    )

    return model

def save_results_summary(
    model_dir, episode_dataframes, episode_rewards, fidelitys, misclassification_costs, suffix=""
):

    results_dict = {
        "mean_episode_reward" + suffix: float(np.mean(episode_rewards)),
        "std_episode_reward" + suffix: float(np.std(episode_rewards)),
        "mean_fidelity" + suffix: float(np.mean(fidelitys)),
        "std_fidelity" + suffix: float(np.std(fidelitys)),
        "mean_misclassification_cost" + suffix: float(np.mean(misclassification_costs)),
        "std_misclassification_cost" + suffix: float(np.std(misclassification_costs)),
        "rewards" + suffix: [float(reward) for reward in episode_rewards],
        "fidelitys" + suffix: [float(fidelity) for fidelity in fidelitys],
    }

    print("-"*100)

    print(
        "Mean Cumulative Reward:", 
        str(round(results_dict["mean_episode_reward" + suffix], 2)), 
        "\u00B1", 
        str(round(results_dict["std_episode_reward" + suffix], 2))
    )

    print(
        "Mean Fidelity:", 
        str(round(results_dict["mean_fidelity" + suffix], 2)), 
        "\u00B1", 
        str(round(results_dict["std_fidelity" + suffix], 2))
    )

    print(
        "Mean Misclassification Cost:", 
        str(round(results_dict["mean_misclassification_cost" + suffix], 2)), 
        "\u00B1", 
        str(round(results_dict["std_misclassification_cost" + suffix], 2))
    )

    with open(os.path.join(model_dir, "results" + suffix + ".yml"), "w") as outfile:
        yaml.dump(results_dict, outfile, default_flow_style=False)

    if not os.path.exists(os.path.join(model_dir, "episodes" + suffix)):
        os.makedirs(os.path.join(model_dir, "episodes" + suffix))

    if episode_dataframes is not None:
        for ep_no in range(len(episode_dataframes)):
            episode_dataframes[ep_no].to_csv(os.path.join(model_dir, "episodes" + suffix, str(ep_no) + ".csv"), index=False)

    return results_dict
