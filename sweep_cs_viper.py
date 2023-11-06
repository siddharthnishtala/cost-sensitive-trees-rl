from utils import set_seed, create_results_directory, get_env, load_dqn_model
from algorithms.cs_viper import train_cs_viper


config = {
    "algorithm": "CS-VIPER",
    "max_iters": 100,
    "n_batch_rollouts": 10,
    "max_samples": 200000,
    "train_frac": 0.8,
    "n_test_rollouts": 50,
    "n_eval_episodes": 200,
}

sweep = {
    "FourRooms": {
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "Taxi-v3": {
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "LunarLander-v2": {
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "highway-fast-v0": {
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
}

for key in sweep.keys():

    config["env"] = key

    dt_depths = sweep[key]["dt_depths"]
    seeds = sweep[key]["seeds"]

    for dt_depth in dt_depths:
        for seed in seeds:

            config["dt_depth"] = dt_depth
            config["seed"] = seed

            set_seed(seed)

            config = create_results_directory("cs_viper", config, ["dt_depth", "seed"])

            env = get_env(config["env"])

            black_box_policy = load_dqn_model(config["env"], env, seed)

            train_cs_viper(env, black_box_policy, config)
