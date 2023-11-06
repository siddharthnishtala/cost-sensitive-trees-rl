import pandas as pd
import numpy as np
import torch


def evaluate(env, tree, black_box_policy, n_episodes, env_name, mean=None, std=None):

    reward_list = []
    fidelity_list = []
    misclassification_cost_list = []
    dfs = []

    for i in range(n_episodes):

        ep_reward = 0

        currentObs = env.reset()
        done = False

        states_list = []
        bbm_states_list = []
        actions_list = []

        while not done:

            if env_name == "Taxi-v3":
                state = np.array([list(env.decode(currentObs))])
            else:
                state = np.array([currentObs])

            # VIPER and MoET don't require normalization
            # SDT and CDT require normalization
            if (mean is not None) and (std is not None):
                if env_name == "Taxi-v3":
                    normalized_state = (
                        np.array([list(env.decode(currentObs))]) - mean
                    ) / std
                else:
                    normalized_state = (np.array([currentObs]) - mean) / std
                
                normalized_state = normalized_state.astype(np.float32)
                action, _, _ = tree.forward(torch.from_numpy(normalized_state))

                with torch.no_grad():
                    action = action.data.max(1)[1].numpy()[0]
            else:
                action = tree.predict(state)[0]

            states_list.append(state[0])
            actions_list.append(action)
            bbm_states_list.append(currentObs)

            nextObs, reward, done, _ = env.step(action)

            ep_reward += reward

            currentObs = nextObs

        bbm_q_values = black_box_policy.predict_q_values(np.array(bbm_states_list))

        ep_df, ep_fidelity, ep_misclassification_cost = analyze_and_compile(
            np.array(states_list), 
            np.array(actions_list),
            bbm_q_values
        )
        
        dfs.append(ep_df)
        reward_list.append(ep_reward)
        fidelity_list.append(ep_fidelity)
        misclassification_cost_list.append(ep_misclassification_cost)

    return dfs, reward_list, fidelity_list, misclassification_cost_list

def analyze_and_compile(states, actions, bbm_q_values):

    cols = ["timestep"]
    cols.extend(["state_" + str(i) for i in range(states.shape[1])])
    cols.append("action")
    cols.extend(["qvalue_" + str(i) for i in range(bbm_q_values.shape[1])])
    cols.append("best_action")
    cols.extend(["qvalue_max", "qvalue_min", "qvalue_restmax", "qvalue_restavg"])

    df = pd.DataFrame(columns=cols)

    df.loc[:, "timestep"] = np.arange(states.shape[0])

    for i in range(states.shape[1]):
        df.loc[:, "state_" + str(i)] = states[:, i]

    df.loc[:, "action"] = actions

    for i in range(bbm_q_values.shape[1]):
        df.loc[:, "qvalue_" + str(i)] = bbm_q_values[:, i]

    df.loc[:, "best_action"] = np.argmax(bbm_q_values, axis=1)

    sorted_bbm_q_values = np.sort(bbm_q_values, axis=1)

    df.loc[:, "qvalue_max"] = sorted_bbm_q_values[:, -1]
    df.loc[:, "qvalue_min"] = sorted_bbm_q_values[:, 0]
    df.loc[:, "qvalue_restmax"] = sorted_bbm_q_values[:, -2]
    df.loc[:, "qvalue_restavg"] = np.mean(sorted_bbm_q_values[:, :-1], axis=1)

    fidelity = np.mean(df["action"] == df["best_action"]) * 100
    misclassification_cost = np.mean(
        df["qvalue_max"] - bbm_q_values[np.arange(bbm_q_values.shape[0]), actions]
    )

    return df, fidelity, misclassification_cost
