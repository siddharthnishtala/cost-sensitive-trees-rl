import numpy as np
import pickle5
import os

from sklearn.tree import ExampleDependentCostSenstiveDecisionTreeClassifier, _tree
from stable_baselines3.dqn.dqn import DQN
from copy import deepcopy

from utils import save_results_summary
from evaluation import evaluate

from .viper import (
    identify_best_policy_reward_mispredictions,
    get_max_number_of_parameters,
    get_number_of_parameters, 
    get_bbm_predictions,  
    get_bbm_q_values, 
    get_rollouts, 
    sample, 
)

def train_cs_viper(env, black_box_policy, config):

    model = ExampleDependentCostSenstiveDecisionTreeClassifier(max_depth=config["dt_depth"])

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
    print("Evaluating CS-VIPER")
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
        if type(student) == ExampleDependentCostSenstiveDecisionTreeClassifier:
            student = learn_tree_policy(
                student, cur_obss, cur_acts, cur_qs, train_frac
            )
        else:
            student.train(cur_obss, cur_qs, train_frac)

        student_trace = get_rollouts(env, student, False, n_batch_rollouts, env_name)
        student_obss = [obs for obs, _, _ in student_trace]

        black_box_model_predictions = get_bbm_predictions(student_trace, teacher, env, env_name)
        black_box_model_qvalues = get_bbm_q_values(student_trace, teacher, env, env_name)

        obss.extend((obs for obs in student_obss))
        acts.extend(list(black_box_model_predictions))
        qs.extend(list(black_box_model_qvalues))

        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        print("Student reward: {}".format(cur_rew))

        if type(student) == ExampleDependentCostSenstiveDecisionTreeClassifier:
            students.append((deepcopy(student), cur_rew))
        else:
            students.append((student.clone(), cur_rew))

    max_student = identify_best_policy_reward_mispredictions(
        env, students, n_test_rollouts, np.array(obss), np.array(acts), env_name
    )

    return max_student

def learn_tree_policy(dt, states, actions, q_values, train_frac):

    (
        states_train, actions_train, q_values_train, 
        states_test, actions_test, q_values_test
    ) = split_train_test(states, actions, q_values, train_frac)

    states_train = states_train.astype(np.float32)
    q_values_train = q_values_train.astype(np.float32)

    cost_mat_train = build_cost_matrix(q_values_train)

    dt.fit(states_train, actions_train, cost_mat_train)

    train_accuracy = np.mean(actions_train == dt.predict(states_train))
    test_accuracy = np.mean(actions_test == dt.predict(states_test))

    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    return dt

def split_train_test(states, actions, q_values, train_frac):

    n_train = int(train_frac * len(states))

    idx = np.arange(len(states))
    np.random.shuffle(idx)

    states_train = states[idx[:n_train]]
    actions_train = actions[idx[:n_train]]
    q_values_train = q_values[idx[:n_train]]

    states_test = states[idx[n_train:]]
    actions_test = actions[idx[n_train:]]
    q_values_test = q_values[idx[n_train:]]

    return states_train, actions_train, q_values_train, states_test, actions_test, q_values_test

def build_cost_matrix(q_values):

    n_classes = q_values.shape[1]

    best_actions = np.argmax(q_values, axis=1)

    cost_mat = np.zeros((q_values.shape[0], n_classes**2))
    for class_gt in range(n_classes):
    
        relevant_points = best_actions == class_gt
    
        for class_p in range(n_classes):
    
            cost_mat[relevant_points, class_gt * n_classes + class_p] = (
                np.max(q_values[relevant_points], axis=1)
                - q_values[relevant_points, class_p]
            )

    return cost_mat
