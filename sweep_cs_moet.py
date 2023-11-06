from utils import set_seed, create_results_directory, get_env, load_dqn_model
from algorithms.cs_moet import train_cs_moet


config = {
    "algorithm": "CS-MoET",
    "moe_init_learning_rate": 0.3,
    "moe_learning_rate_decay": 0.97,
    "moe_log_frequency": None,
    "moe_stop_count": None,
    "moe_regularization_mode": 0,
    "use_adam_optimizer": True,
    "max_iters": 40,
    "n_batch_rollouts": 10,
    "max_samples": 200000,
    "train_frac": 0.8,
    "n_test_rollouts": 50,
    "n_eval_episodes": 200,
}

sweep = {
    "FourRooms": {
        "no_of_experts": [2, 3],
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "Taxi-v3": {
        "no_of_experts": [2, 3],
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "LunarLander-v2": {
        "no_of_experts": [2, 3],
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    },
    "highway-fast-v0": {
        "no_of_experts": [2, 3],
        "dt_depths": [i for i in range(1, 16)],
        "seeds": [1, 2, 3, 4, 5],
    }
}

for key in sweep.keys():

    config["env"] = key

    no_of_experts = sweep[key]["no_of_experts"]
    dt_depths = sweep[key]["dt_depths"]
    seeds = sweep[key]["seeds"]

    for no_of_expert in no_of_experts:
        for dt_depth in dt_depths:
            for seed in seeds:

                config["no_of_experts"] = no_of_expert
                config["dt_depth"] = dt_depth
                config["seed"] = seed

                set_seed(seed)

                config = create_results_directory(
                    "cs_moet",
                    config,
                    [
                        "no_of_experts",
                        "dt_depth",
                        "seed"
                    ],
                )

                env = get_env(config["env"])

                black_box_policy = load_dqn_model(config["env"], env, seed)

                train_cs_moet(env, black_box_policy, config)
