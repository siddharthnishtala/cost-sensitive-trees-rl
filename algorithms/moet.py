import numpy as np
import pickle5
import copy
import math
import os

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from utils import get_env_details, save_results_summary
from evaluation import evaluate

from .viper import train_dagger, split_train_test


ADAM_EPSILON = 1e-7
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

def train_moet(env, black_box_policy, config):

    state_dim, num_actions = get_env_details(config["env"])

    model = MOEPolicy(
        experts_no=config["no_of_experts"],
        dts_depth=config["dt_depth"],
        num_classes=num_actions,
        hard_prediction=False,
        max_epoch=config["max_iters"],
        init_learning_rate=config["moe_init_learning_rate"],
        learning_rate_decay=config["moe_learning_rate_decay"],
        log_frequency=config["moe_log_frequency"],
        stop_count=config["moe_stop_count"],
        regularization_mode=config["moe_regularization_mode"],
        use_adam_optimizer=config["use_adam_optimizer"]
    )

    best_model = train_dagger(
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

    max_parameters = get_max_number_of_parameters(state_dim, config["no_of_experts"], config["dt_depth"], num_actions)
    real_parameters = get_number_of_parameters(state_dim, best_model, num_actions)

    print("-"*100)
    print("Maximum Possible Parameters:", max_parameters)
    print("Parameters:", real_parameters)

    with open(os.path.join(config["results_dir"], "model.pkl"), "wb") as f:
        pickle5.dump(best_model, f)

    print("-"*100)
    print("Evaluating MoET")
    episode_dataframes, episode_rewards, fidelitys, misclassification_costs = evaluate(
        env, best_model, black_box_policy, config["n_eval_episodes"], config["env"]
    )

    results = save_results_summary(
        config["results_dir"], episode_dataframes, episode_rewards, fidelitys, misclassification_costs
    )

    best_model.hard_prediction = True

    print("-"*100)
    print("Evaluating MoET_h")
    episode_dataframes, episode_rewards, fidelitys, misclassification_costs = evaluate(
        env, best_model, black_box_policy, config["n_eval_episodes"], config["env"]
    )

    results_d = save_results_summary(
        config["results_dir"], episode_dataframes, episode_rewards, fidelitys, misclassification_costs, "_d"
    )

    print("-"*100)

def get_max_number_of_parameters(state_dim, no_of_experts, tree_depth, num_actions):

    return no_of_experts * state_dim + no_of_experts * ((num_actions + 2) * (2**tree_depth) - 2) 

def get_number_of_parameters(state_dim, student, num_actions):

    dtc_list = student.moe.dtc_list
    total_params = 0
    for tree in dtc_list:
        total_params += get_number_of_parameters_tree(tree, num_actions)

    total_params += len(dtc_list) * state_dim

    return total_params

def get_number_of_parameters_tree(tree, num_actions):
    tree_ = tree.tree_

    def recurse(node, depth):
        params = 0
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            params += 2
            params += recurse(tree_.children_left[node], depth + 1)
            params += recurse(tree_.children_right[node], depth + 1)
        else:
            params += num_actions

        return params

    total_params = recurse(0, 1)

    return total_params

class MOEPolicy:

    hard_prediction = False
    num_classes = 2
    max_epoch = 80
    init_learning_rate = 2
    learning_rate_decay = 0.95
    log_frequency = 10
    use_adam_optimizer = False

    def __init__(
        self,
        experts_no,
        dts_depth,
        num_classes,
        hard_prediction,
        max_epoch,
        init_learning_rate,
        learning_rate_decay,
        log_frequency,
        stop_count,
        regularization_mode,
        use_adam_optimizer
    ):

        self.experts_no = experts_no
        self.dts_depth = dts_depth
        self.num_classes = num_classes
        self.hard_prediction = hard_prediction
        self.max_epoch = max_epoch
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.log_frequency = log_frequency
        self.stop_count = stop_count
        self.regularization_mode = regularization_mode
        self.use_adam_optimizer = use_adam_optimizer

    def get_node_count(self):

        sum = 0

        for tree in self.moe.dtc_list:
            sum += tree.tree_.node_count

        sum += int(math.ceil(math.log(self.experts_no, 2)))

        return sum

    def get_depth(self):

        max_dt_depth = 0

        for tree in self.moe.dtc_list:
            max_dt_depth = max(tree.tree_.max_depth, max_dt_depth)

        experts_num = len(self.moe.dtc_list)

        return int(math.ceil(math.log(experts_num, 2))) + max_dt_depth

    def _fit(self, obss, acts, obss_test, acts_test):
        
        self.moe = MOETClassifierNew(
            experts_no=self.experts_no,
            no_class=self.num_classes,
            use_adam=self.use_adam_optimizer
        )

        self.moe.train(
            obss,
            acts,
            obss_test,
            acts_test,
            max_depth=self.dts_depth,
            max_epoch=self.max_epoch,
            init_learning_rate=self.init_learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            log_frequency=self.log_frequency,
            stop_count=self.stop_count,
            regularization_mode=self.regularization_mode,
        )

    def train(self, obss, acts, train_frac):

        obss_train, acts_train, obss_test, acts_test = split_train_test(
            obss, acts, train_frac
        )
        self._fit(obss_train, acts_train, obss_test, acts_test)

        train_accuracy = np.mean(acts_train == self.predict(obss_train))
        test_accuracy = np.mean(acts_test == self.predict(obss_test))

        print("Train accuracy: {}".format(train_accuracy))
        print("Test accuracy: {}".format(test_accuracy))

    def predict(self, obss):

        if not self.hard_prediction:
            return self.moe.predict(obss)
        else:
            return self.moe.predict_hard(obss)

    def clone(self):

        clone = MOEPolicy(
            experts_no=self.experts_no,
            dts_depth=self.dts_depth,
            num_classes=self.num_classes,
            hard_prediction=self.hard_prediction,
            max_epoch=self.max_epoch,
            init_learning_rate=self.init_learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            log_frequency=self.log_frequency,
            stop_count=self.stop_count,
            regularization_mode=self.regularization_mode,
            use_adam_optimizer=self.use_adam_optimizer
        )
        clone.moe = copy.copy(self.moe)

        return clone

class MOETBase(object):

    def __init__(self, experts_no, default_type=np.float32):

        self.experts_no = experts_no
        self.default_type = default_type

    def softmax(self, x_normalized, tetag, experts_no):

        x = np.tile(x_normalized, experts_no) * tetag
        x = (
            x.reshape(-1, x_normalized.shape[0], x_normalized.shape[1])
            .sum(axis=2)
            .reshape(x_normalized.shape[0], -1)
        )
        e_x = np.exp(
            x
            - np.array(
                [
                    np.max(x, axis=1),
                ]
                * x.shape[1]
            ).transpose()
        )
        out = (e_x.transpose() / e_x.sum(axis=1)).transpose()

        return out

    def h_fun(self, gating, pdf):

        h = ((gating * pdf).T / np.sum(gating * pdf, axis=1)).T

        return h

    def ds_dtetag(self, x_normalized, tetag, experts_no):

        N = x_normalized.shape[0]
        U = tetag.shape[0]
        E = experts_no

        dsdtetag = np.zeros([N, U, E], dtype=self.default_type)
        no_x = int(U / E)

        for j in range(E):
            dsdtetag[:, j * no_x : (j + 1) * no_x, j] = x_normalized[:, :]

        return dsdtetag

    def e_fun(self, hf, gating, dsdtetag):

        # N - #examples, E - #experts
        # hf dims: (N, E)
        # Gating dims: (N, E);
        # dsdtetag dims: (N, U, E)
        N = dsdtetag.shape[0]
        U = dsdtetag.shape[1]
        E = dsdtetag.shape[2]

        # dims: (N, E)
        tmp1 = hf - gating
        # dims: (N * E)
        tmp1 = np.reshape(tmp1, (N * E))
        # dims: (N * E, 1)
        tmp1 = tmp1[:, np.newaxis]
        # dims: (N, E, U)
        tmp2 = np.swapaxes(dsdtetag, 1, 2)
        # dims: (N * E, U)
        tmp2 = np.reshape(tmp2, (N * E, U))

        tmp2 *= tmp1
        e = np.sum(tmp2, axis=0)

        return e

    def R_fun(self, gating, dsdtetag, experts_no):

        N = dsdtetag.shape[0]
        U = dsdtetag.shape[1]
        E = experts_no

        max_batch_size = 10000
        num_batches = np.ceil(float(N) / max_batch_size).astype(int)

        R = np.zeros([U, U], dtype=self.default_type)
        for batch_idx in range(num_batches):
            start_index = batch_idx * max_batch_size
            end_index = start_index + max_batch_size
            gating_sliced = gating[start_index:end_index, :]
            dsdtetag_sliced = dsdtetag[start_index:end_index, :, :]

            n = gating_sliced.shape[0]

            # dims: (N, E)
            tmp1 = gating_sliced * (1 - gating_sliced)
            # dims: (N * E)
            tmp1 = np.reshape(tmp1, (n * E))
            # dims: (N * E, 1, 1)
            tmp1 = tmp1[:, np.newaxis, np.newaxis]
            # dims: (N, E, U)
            tmp2 = np.swapaxes(dsdtetag_sliced, 1, 2)
            # dims: (N * E, U)
            tmp2 = np.reshape(tmp2, (n * E, U))
            # dims: (N * E, U, U)
            tmp2 = np.matmul(tmp2[:, :, np.newaxis], tmp2[:, np.newaxis, :])

            tmp2 *= tmp1
            R += np.sum(tmp2, axis=0)

        return R
    
class MOETClassifier(MOETBase):

    default_type = np.float32
    use_adam = False

    def __init__(self, experts_no, no_class, default_type=np.float32, use_adam=False):

        super(MOETClassifier, self).__init__(
            experts_no=experts_no,
            default_type=default_type,
        )
        self.no_class = no_class
        self.tetag = None
        self.scaler = None
        self.dtc_list = None
        self.use_adam = use_adam

    def _preprocess_train_data(self, x):

        x = x.astype(self.default_type)
        self.scaler = StandardScaler()
        self.scaler.fit(x)

    def _normalize_x(self, x):

        x = self.scaler.transform(x)

        return np.append(x, np.ones([x.shape[0], 1], dtype=self.default_type), axis=1)

    def _dt_proba(self, expert_id, x_normalized):

        dt = self.dtc_list[expert_id]
        check = np.where(np.isin(np.arange(self.no_class), dt.classes_) == False)[0]
        dt_probs = dt.predict_proba(x_normalized)
        if check.size != 0:
            for i in check:
                dt_probs = np.insert(dt_probs, [i], 0, axis=1)

        return dt_probs

    def _expert_proba(self, x_normalized):

        N = x_normalized.shape[0]
        E = self.experts_no
        C = self.no_class

        probs = np.zeros([N, C, E], dtype=self.default_type)
        for expert_id in range(self.experts_no):
            expert_probs = self._dt_proba(expert_id, x_normalized)
            probs[:, :, expert_id] = expert_probs

        return probs

    def predict_expert_proba(self, x):

        x_normalized = self._normalize_x(x)

        return self.softmax(x_normalized, self.tetag, self.experts_no)

    def predict_expert(self, x):

        gating = self.predict_expert_proba(x)

        return np.argmax(gating, axis=1)

    def predict_with_expert(self, x, experts):

        x_normalized = self._normalize_x(x)

        N = x.shape[0]
        E = self.experts_no
        C = self.no_class

        # dims: (N, E).
        gating = np.zeros([N, E])
        gating[np.arange(N), experts] = 1

        # dims: (N, C, E).
        probs = self._expert_proba(x_normalized)
        probs *= np.repeat(gating[:, np.newaxis, :], C, axis=1)

        result = np.argmax(np.sum(probs, axis=2), axis=1)

        return np.round(result).astype(int)

    def predict(self, x):

        x_normalized = self._normalize_x(x)

        N = x.shape[0]
        E = self.experts_no
        C = self.no_class

        # dims: (N, E).
        gating = self.predict_expert_proba(x)

        # dims: (N, C, E).
        probs = self._expert_proba(x_normalized)
        probs *= np.repeat(gating[:, np.newaxis, :], C, axis=1)

        result = np.argmax(np.sum(probs, axis=2), axis=1)

        return np.round(result).astype(int)

    def predict_hard(self, x):

        return self.predict_with_expert(x, self.predict_expert(x))

    def _fit_epoch(
        self,
        x_normalized,
        y,
        max_depth,
        learn_rate,
        regularization_mode,
        is_first_epoch,
        train_gating_multiple_times=False,
    ):

        self.dtc_list = [None for i in range(self.experts_no)]
        gating = self.softmax(x_normalized, self.tetag, self.experts_no)

        if regularization_mode == 0:
            weights = gating
        elif regularization_mode == 1:
            indexes = np.random.choice(
                x_normalized.shape[0], size=int(0.2 * x_normalized.shape[0])
            )
            weights = gating
            weights[indexes, :] = np.ones(self.experts_no)
        elif regularization_mode == 2:
            indexes = np.random.choice(
                x_normalized.shape[0], size=int(0.8 * x_normalized.shape[0])
            )
            x_normalized = x_normalized[indexes, :]
            y = y[indexes]
            gating = gating[indexes, :]
            weights = gating
        else:
            raise Exception(
                "Unrecognized regularization mode: {}".format(regularization_mode)
            )

        pdf = np.zeros(
            [x_normalized.shape[0], self.experts_no], dtype=self.default_type
        )

        for j in range(self.experts_no):

            if max_depth == 0:
                self.dtc_list[j] = DecisionTreeClassifier(
                    max_depth=1, min_samples_split=len(x_normalized) + 1
                )
            else:
                self.dtc_list[j] = DecisionTreeClassifier(max_depth=max_depth)
            self.dtc_list[j].fit(x_normalized, y, sample_weight=weights[:, j].T)
            dt_probs = self._dt_proba(j, x_normalized)
            pdf[:, j] = dt_probs[np.arange(len(y)), y.reshape(-1)].astype(
                self.default_type
            )

        h = self.h_fun(gating, pdf)
        dsdtetag = self.ds_dtetag(x_normalized, self.tetag, self.experts_no)
        e = self.e_fun(h, gating, dsdtetag)
        R = self.R_fun(gating, dsdtetag, self.experts_no)

        if np.linalg.cond(R) < 1e7:
            self.tetag += learn_rate * np.linalg.inv(R).dot(e)
        else:
            self.tetag += learn_rate * e

        return R

    def fit(
        self,
        x,
        y,
        max_depth,
        max_epoch=100,
        init_learning_rate=2,
        learning_rate_decay=0.95,
        regularization_mode=0,
    ):

        self._preprocess_train_data(x)
        x_normalized = self._normalize_x(x)

        self.tetag = np.random.rand(x_normalized.shape[1] * self.experts_no).astype(
            self.default_type
        )
        learn_rate = [
            init_learning_rate * (learning_rate_decay ** max(float(i), 0.0))
            for i in range(max_epoch)
        ]

        for epoch_id in range(max_epoch):
            self._fit_epoch(
                x_normalized,
                y,
                max_depth,
                learn_rate[epoch_id],
                regularization_mode,
                epoch_id == 0,
            )

    def train(
        self,
        x,
        y,
        x_test,
        y_test,
        max_depth,
        max_epoch=100,
        init_learning_rate=1.0,
        learning_rate_decay=0.98,
        log_frequency=None,
        stop_count=None,
        regularization_mode=0,
        return_best_epoch=True,
        gradually_increase_max_depth=True,
        train_gating_multiple_times=False,
    ):

        self._preprocess_train_data(x)
        x_normalized = self._normalize_x(x)

        self.tetag = np.random.rand(x_normalized.shape[1] * self.experts_no).astype(
            self.default_type
        )
        learn_rate = [
            init_learning_rate * (learning_rate_decay ** max(float(i), 0.0))
            for i in range(max_epoch)
        ]

        best_tetag = None
        best_dtc_list = None
        best_test_perf = -1.0
        test_perf = -1.0
        no_improvement_count = 0

        is_first_epoch = True

        if gradually_increase_max_depth:
            self._fit_epoch(
                x_normalized,
                y,
                max_depth=0,
                learn_rate=learn_rate[0],
                regularization_mode=regularization_mode,
                is_first_epoch=is_first_epoch,
                train_gating_multiple_times=train_gating_multiple_times,
            )
            is_first_epoch = False

        for epoch_id in range(max_epoch):

            current_max_depth = max_depth

            R = self._fit_epoch(
                x_normalized,
                y,
                current_max_depth,
                learn_rate[epoch_id],
                regularization_mode,
                is_first_epoch=is_first_epoch,
                train_gating_multiple_times=train_gating_multiple_times,
            )
            is_first_epoch = False

            old_test_perf = test_perf
            test_perf = accuracy_score(y_test, self.predict(x_test))

            if return_best_epoch and test_perf > best_test_perf:
                best_test_perf = test_perf
                best_tetag = self.tetag.copy()
                best_dtc_list = copy.deepcopy(self.dtc_list)

            if stop_count is not None:
                if test_perf <= old_test_perf:
                    no_improvement_count += 1
                    if no_improvement_count == stop_count:
                        print(
                            "Early stopping training after {} epochs.".format(
                                epoch_id + 1
                            )
                        )
                        break
                else:
                    no_improvement_count = 0

            if (
                log_frequency is not None
                and log_frequency > 0
                and (epoch_id == 0 or (epoch_id + 1) % log_frequency == 0)
            ):
                print("Epoch {} stats:".format(epoch_id + 1))

                train_score_soft = f1_score(y, self.predict(x), average="weighted")
                train_score_hard = f1_score(y, self.predict(x), average="weighted")

                print(
                    "Train (f1 score): soft={:1.2f}, hard={:1.2f}".format(
                        train_score_soft, train_score_hard
                    )
                )

                test_score_soft = f1_score(
                    y_test, self.predict(x_test), average="weighted"
                )
                test_score_hard = f1_score(
                    y_test, self.predict_hard(x_test), average="weighted"
                )
                print(
                    "Test (f1 score): soft={:1.2f}, hard={:1.2f}".format(
                        test_score_soft, test_score_hard
                    )
                )

        if return_best_epoch:
            self.tetag = best_tetag
            self.dtc_list = best_dtc_list

class MOETClassifierNew(MOETClassifier):

    weights = None

    def initialize_weights(self, regularization_mode, gating):

        self.weights = np.random.rand(gating.shape[0], gating.shape[1])

        if regularization_mode == 0:
            pass
        elif regularization_mode == 1:
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.2 * x_normalized.shape[0]))
            self.weights[indexes, :] = np.ones(self.experts_no)
        elif regularization_mode == 2:
            indexes = np.random.choice(x_normalized.shape[0],
                                       size=int(0.8 * x_normalized.shape[0]))
            x_normalized = x_normalized[indexes, :]
            y = y[indexes]
            self.weights = self.weights[indexes, :]
        else:
            raise Exception('Unrecognized regularization mode: {}'.format(regularization_mode))

    def train_experts(self, x_normalized, y, pdf, max_depth):

        for j in range(self.experts_no):
            if max_depth == 0:
                self.dtc_list[j] = DecisionTreeClassifier(max_depth=1,
                                       min_samples_split=len(x_normalized) + 1)
            else:
                self.dtc_list[j] = DecisionTreeClassifier(max_depth=max_depth)
            self.dtc_list[j].fit(x_normalized,
                                 y,
                                 sample_weight=self.weights[:, j].T)
            dt_probs = self._dt_proba(j, x_normalized)
            pdf[:, j] = dt_probs[np.arange(len(y)), y].astype(self.default_type)

    def train_gating(self, x_normalized, gating, pdf, learn_rate):

        h = self.h_fun(gating, pdf)
        self.weights = h
        dsdtetag = self.ds_dtetag(x_normalized, self.tetag, self.experts_no)
        e = self.e_fun(h, gating, dsdtetag)

        if not self.use_adam:
            R = self.R_fun(gating, dsdtetag, self.experts_no)

            if np.linalg.cond(R) < 1e7:
                self.tetag += learn_rate * np.linalg.inv(R).dot(e)
            else:
                self.tetag += learn_rate * e

            return R
        else:
            t = self.iter_no
            self.adam_m = ADAM_BETA_1 * self.adam_m + (1 - ADAM_BETA_1) * e
            self.adam_v = ADAM_BETA_2 * self.adam_v + (1 - ADAM_BETA_2) * np.power(e, 2)
            m_hat = self.adam_m / (1 - np.power(ADAM_BETA_1, t))
            v_hat = self.adam_v / (1 - np.power(ADAM_BETA_2, t))
            self.tetag += learn_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            self.iter_no += 1
            
    def _fit_epoch(self,
                   x_normalized,
                   y,
                   max_depth,
                   learn_rate,
                   regularization_mode,
                   is_first_epoch,
                   train_gating_multiple_times):

        self.dtc_list = [None for i in range(self.experts_no)]
        gating = self.softmax(x_normalized, self.tetag, self.experts_no)

        if is_first_epoch:
            self.iter_no = 1
            self.adam_m = 0
            self.adam_v = 0
            self.initialize_weights(regularization_mode, gating)

        pdf = np.zeros([x_normalized.shape[0], self.experts_no],
                       dtype=self.default_type)
        
        self.train_experts(x_normalized, y, pdf, max_depth)

        if train_gating_multiple_times:
            for i in range(5):
                self.train_gating(x_normalized, gating, pdf, learn_rate)
        else:
            self.train_gating(x_normalized, gating, pdf, learn_rate)