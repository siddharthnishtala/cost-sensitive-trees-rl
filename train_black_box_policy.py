import gym
import os

from stable_baselines3.dqn.dqn import DQN

from utils import set_seed, get_env, save_config, save_results_summary


# config = {
#     "seed": 0,
#     "env": "LunarLander-v2",
#     "policy": "MlpPolicy",
#     "lr": 0.00063,
#     "buffer_size": 50000,
#     "learning_starts": 0,
#     "batch_size": 128,
#     "gamma": 0.99,
#     "train_freq": 4,
#     "gradient_steps": -1,
#     "target_update_interval": 250,
#     "exploration_fraction": 0.12,
#     "exploration_initial_eps": 1.0,
#     "exploration_final_eps": 0.1,
#     "policy_kwargs": {
#         "net_arch": [256, 256]
#     },
#     "n_timesteps": 200000,
# }

# config = {
#     "seed": 0,
#     "env": "Taxi-v3",
#     "policy": "MlpPolicy",
#     "lr": 0.0005,
#     "buffer_size": 1000000,
#     "learning_starts": 50000,
#     "batch_size": 32,
#     "gamma": 0.99,
#     "train_freq": 4,
#     "gradient_steps": 1,
#     "target_update_interval": 10000,
#     "exploration_fraction": 0.1,
#     "exploration_initial_eps": 1.0,
#     "exploration_final_eps": 0.05,
#     "policy_kwargs": {"net_arch": [64, 64]},
#     "n_timesteps": 1000000,
# }

# config = {
#     "seed": 0,
#     "env": "FourRooms",
#     "policy": "MlpPolicy",
#     "lr": 0.0001,
#     "buffer_size": 100000,
#     "learning_starts": 1000,
#     "batch_size": 64,
#     "gamma": 0.99,
#     "train_freq": 256,
#     "gradient_steps": 128,
#     "target_update_interval": 10,
#     "exploration_fraction": 0.4,
#     "exploration_initial_eps": 1.0,
#     "exploration_final_eps": 0.04,
#     "policy_kwargs": {"net_arch": [256, 256]},
#     "n_timesteps": 1000000,
# }

config = {
    "seed": 0,
    "env": "highway-fast-v0",
    "policy": "MlpPolicy",
    "lr": 0.0005,
    "buffer_size": 15000,
    "learning_starts": 200,
    "batch_size": 64,
    "gamma": 0.8,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 50,
    "exploration_fraction": 0.75,
    "exploration_initial_eps": 0.5,
    "exploration_final_eps": 0.01,
    "policy_kwargs": {"net_arch": [256, 256]},
    "n_timesteps": 150000,
}

set_seed(config["seed"])

env = get_env(config["env"])

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=config["lr"],
    buffer_size=config["buffer_size"],
    learning_starts=config["learning_starts"],
    batch_size=config["batch_size"],
    gamma=config["gamma"],
    train_freq=config["train_freq"],
    gradient_steps=config["gradient_steps"],
    target_update_interval=config["target_update_interval"],
    exploration_fraction=config["exploration_fraction"],
    exploration_initial_eps=config["exploration_initial_eps"],
    exploration_final_eps=config["exploration_final_eps"],
    policy_kwargs=config["policy_kwargs"],
    verbose=1,
    seed=config["seed"],
)

model_dir = os.path.join("rl-trained-agents", "dqn", config["env"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.learn(total_timesteps=config["n_timesteps"], progress_bar=True)

model.save(os.path.join(model_dir, "model"))

save_config(model_dir, config)

episode_rewards = []
for episode_no in range(1000):
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(int(action))
        episode_reward += reward

    episode_rewards.append(episode_reward)

    print("Episode", str(episode_no + 1), "| Total Reward:", str(episode_reward))

results_summary = save_results_summary(model_dir, None, episode_rewards, [100], [0])
