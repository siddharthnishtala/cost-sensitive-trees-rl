import numpy as np
import pickle5
import os

from sklearn.tree import DecisionTreeClassifier, _tree
from stable_baselines3.dqn.dqn import DQN
from copy import deepcopy

from utils import save_results_summary
from evaluation import evaluate


def train_viper(env, black_box_policy, config):

    model = DecisionTreeClassifier(max_depth=config["dt_depth"])

    best_tree = train_dagger(
        env,
        config["env"],
        black_box_policy,
        model,
        config["max_iters"],
        config["n_batch_rollouts"],
        config["max_samples"],
        config["train_frac"],
        config["n_test_rollouts"],
    )

    max_parameters = get_max_number_of_parameters(config["dt_depth"])
    real_parameters = get_number_of_parameters(best_tree)

    print("-"*100)
    print("Maximum Possible Parameters:", max_parameters)
    print("Parameters:", real_parameters)

    with open(os.path.join(config["results_dir"], "tree.pkl"), "wb") as f:
        pickle5.dump(best_tree, f)

    print("-"*100)
    print("Evaluating VIPER")
    episode_dataframes, episode_rewards, fidelitys, misclassification_costs = evaluate(
        env, best_tree, black_box_policy, config["n_eval_episodes"], config["env"]
    )

    results = save_results_summary(
        config["results_dir"], episode_dataframes, episode_rewards, fidelitys, misclassification_costs
    )

    print("-"*100)

def train_dagger(
    env,
    env_name,
    teacher,
    student,
    max_iters,
    n_batch_rollouts,
    max_samples,
    train_frac,
    n_test_rollouts,
):

    obss, acts, qs = [], [], []
    students = []

    trace = get_rollouts(env, teacher, False, n_batch_rollouts, env_name)

    obss.extend((obs for obs, _, _ in trace))
    acts.extend((act[0] for _, act, _ in trace))
    qs.extend(get_bbm_q_values(trace, teacher, env, env_name))

    for i in range(max_iters):

        print("Iteration {}/{}".format(i+1, max_iters))

        cur_obss, cur_acts, cur_qs = sample(
            np.array(obss), np.array(acts), np.array(qs), max_samples
        )

        print("Training student with {} points".format(len(cur_obss)))
        if type(student) == DecisionTreeClassifier:
            student = learn_tree_policy(
                student, cur_obss, cur_acts, train_frac
            )
        else:
            student.train(cur_obss, cur_acts, train_frac)

        student_trace = get_rollouts(env, student, False, n_batch_rollouts, env_name)
        student_obss = [obs for obs, _, _ in student_trace]

        black_box_model_predictions = get_bbm_predictions(student_trace, teacher, env, env_name)
        black_box_model_qvalues = get_bbm_q_values(student_trace, teacher, env, env_name)

        obss.extend((obs for obs in student_obss))
        acts.extend(list(black_box_model_predictions))
        qs.extend(list(black_box_model_qvalues))

        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        print("Student reward: {}".format(cur_rew))

        if type(student) == DecisionTreeClassifier:
            students.append((deepcopy(student), cur_rew))
        else:
            students.append((student.clone(), cur_rew))

    max_student = identify_best_policy_reward_mispredictions(
        env, students, n_test_rollouts, np.array(obss), np.array(acts), env_name
    )

    return max_student

def get_rollouts(env, policy, render, n_batch_rollouts, env_name):

    rollouts = []

    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, render, env_name))

    return rollouts

def get_rollout(env, policy, render, env_name):

    obs = np.array(env.reset())
    done = False

    if env_name == "Taxi-v3" and type(policy) != DQN:
        obs = list(env.decode(obs))

    rollout = []

    while not done:
        if render:
            env.render()

        if type(policy) == DQN:
            act = policy.predict(np.array([obs]), deterministic=True)[0]
        else:
            act = policy.predict(np.array([obs]))[0]

        if isinstance(act, np.int64):
            next_obs, rew, done, info = env.step(act)
        else:
            next_obs, rew, done, info = env.step(act[0])

        if env_name == "Taxi-v3":
            if type(policy) != DQN:
                rollout.append((obs, act, rew))
            else:
                rollout.append((list(env.decode(obs)), act, rew))

            if type(policy) != DQN:
                next_obs = list(env.decode(next_obs))
        else:
            rollout.append((obs, act, rew))

        obs = np.array(next_obs)

    return rollout

def get_bbm_q_values(trace, black_box_policy, env, env_name):

    if env_name == "Taxi-v3":
        q_values = black_box_policy.predict_q_values(
            np.array(
                [
                    env.encode(obs[0], obs[1], obs[2], obs[3])
                    for obs, _, _ in trace
                ]
            )
        )
    else:
        q_values = black_box_policy.predict_q_values(
            np.array([obs for obs, _, _ in trace])
        )

    return q_values

def get_bbm_predictions(trace, black_box_policy, env, env_name):

    if env_name == "Taxi-v3":
        predictions = black_box_policy.predict(
            np.array(
                [
                    env.encode(obs[0], obs[1], obs[2], obs[3])
                    for obs, _, _ in trace
                ]
            )
        )[0]
    else:
        predictions = black_box_policy.predict(
            np.array([obs for obs, _, _ in trace])
        )[0]

    return predictions

def sample(states, actions, q_values, max_pts):

    ps = np.max(q_values, axis=1) - np.min(q_values, axis=1)
    ps = ps / np.sum(ps)

    idx = np.random.choice(len(states), size=min(max_pts, np.sum(ps > 0)), p=ps)

    return states[idx], actions[idx], q_values[idx]

def learn_tree_policy(dt, states, actions, train_frac):

    states_train, actions_train, states_test, actions_test = split_train_test(
        states, actions, train_frac
    )

    dt.fit(states_train, actions_train)

    train_accuracy = np.mean(actions_train == dt.predict(states_train))
    test_accuracy = np.mean(actions_test == dt.predict(states_test))

    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    return dt

def split_train_test(states, actions, train_frac):

    n_train = int(train_frac * len(states))

    idx = np.arange(len(states))
    np.random.shuffle(idx)

    states_train = states[idx[:n_train]]
    actions_train = actions[idx[:n_train]]

    states_test = states[idx[n_train:]]
    actions_test = actions[idx[n_train:]]

    return states_train, actions_train, states_test, actions_test

def identify_best_policy_reward_mispredictions(
    env, policies, n_test_rollouts, observations, teacher_actions, env_name
):

    results = list()

    for policy_idx in range(0, len(policies)):

        print("Evaluating Policy {0}/{1}".format(policy_idx+1, len(policies)))

        policy = policies[policy_idx][0]

        reward = test_policy(env, policy, n_test_rollouts, env_name)

        student_actions = policy.predict(observations)
        accuracy = np.mean(teacher_actions == student_actions)
        mispredictions = 1 - accuracy

        results.append((policy, reward, mispredictions))

        print(
            "Policy {0}: reward={1}; mispredictions={2}".format(
                policy_idx+1, reward, mispredictions
            )
        )

    sorted_results = sorted(results, key=lambda x: (-x[1], x[2]))

    best = sorted_results[0]
    best_policy_index = results.index(best)

    print("Choosing Policy {0} as the best.".format(best_policy_index+1))

    return sorted_results[0][0]

def test_policy(env, policy, n_test_rollouts, env_name):

    cumulative_reward = 0.0

    for i in range(n_test_rollouts):

        policy_dataset = get_rollout(env, policy, False, env_name)

        cumulative_reward += sum((reward for _, _, reward in policy_dataset))

    return cumulative_reward / n_test_rollouts

def get_max_number_of_parameters(tree_depth):

    return 3 * (2**tree_depth) - 2

def get_number_of_parameters(tree):
    tree_ = tree.tree_

    def recurse(node, depth):
        params = 0
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            params += 2
            params += recurse(tree_.children_left[node], depth + 1)
            params += recurse(tree_.children_right[node], depth + 1)
        else:
            params += 1

        return params

    total_params = recurse(0, 1)

    return total_params
