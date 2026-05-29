"""
Each node selects a different number of neighbors, which means that the reinforcement strategy is node-specific.
"""
import copy

import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class BernoulliBandit:
    """
    Bernoulli multi-armed bandit, where K denotes the number of arms.
    """
    def __init__(self, gnn_env):
        set_seed()
        self.num_nodes = gnn_env.adj_ori.shape[0]
        self.action_size = gnn_env.action_space
        self.policy = gnn_env.policy

    # Reward calculation method
    def step(self, action, gnn_env, past_res):
        # After the player selects arm k, calculate the current reward based on the action.
        # (1) First, convert k to tensor type.
        action = torch.tensor(action)
        # (2) Start feeding it into the GNN for calculation.
        reward, cur_res, clu_res = gnn_env.evaluate(action, past_res)
        return reward, cur_res, clu_res

class Solver:
    """
    Basic framework of the multi-armed bandit algorithm.
    """
    def __init__(self, bandit, init_action, policy="", action_size=""):
        set_seed()
        self.bandit = bandit
        bandit.policy = policy
        self.bandit.action_size = action_size
        self.policy = bandit.policy
        self.counts = np.zeros((self.bandit.num_nodes, self.bandit.action_size))  # Number of attempts for each arm (number of aggregated neighbors) of each node

        if self.policy == "breadth":
            row_indices = init_action[:, np.newaxis]
            column_indices = np.arange(self.bandit.action_size)
            self.counts = np.where(column_indices < row_indices, np.zeros((self.bandit.num_nodes, self.bandit.action_size)), 1000)
        else:
            self.counts = np.zeros((self.bandit.num_nodes, self.bandit.action_size))
        self.regret = 0.
        self.actions = []
        self.regrets = []

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps, gnn_env, past_res, clu_res):
        best_clu_res = clu_res
        best_clu_obj = [0]
        consecutive_step = 0
        accumulated_reward_list = []
        accumulated_reward = 0
        for i in range(num_steps):
            temp_model = copy.deepcopy(gnn_env)
            action, reward, cur_res, clu_res = self.run_one_step(gnn_env, past_res)
            #print("step: {} action: {} reward: {} clu_res: {}".format(i, action, reward, clu_res))
            print("step: {} reward: {} clu_res: {}".format(i, reward, clu_res))
            self.counts[np.arange(self.bandit.num_nodes), action] += 1
            self.actions.append(action)

            if consecutive_step > 50:   #50
                break

            if clu_res[0] >= best_clu_res[0]:
                best_clu_res = clu_res
                best_model = temp_model
                best_action = action
                consecutive_step = 0
            else:
                consecutive_step += 1
            past_res = cur_res
            accumulated_reward += reward
            accumulated_reward_list.append(accumulated_reward)
        return best_clu_res, best_action, reward, best_model, cur_res

# UCB
class UCB(Solver):
    """
    UCB algorithm, inheriting from the Solver class.
    """
    def __init__(self, bandit, init_action, policy="", action_size="", coef=0.5):
        super(UCB, self).__init__(bandit, init_action, policy, action_size)
        set_seed()
        self.policy = bandit.policy
        self.total_count = 0
        if self.policy == "breadth":
            row_indices = init_action[:, np.newaxis]
            column_indices = np.arange(self.bandit.action_size)
            self.estimates = np.where(column_indices < row_indices, np.random.rand(self.bandit.num_nodes, self.bandit.action_size), 0.0)
        else:
            self.estimates = np.random.uniform(low=0.4, high=0.6, size=(self.bandit.num_nodes, self.bandit.action_size))
            self.estimates[:, 0] = 0.9
        self.coef = coef  # Weight used to control exploration

    def run_one_step(self, gnn_env, past_res):
        self.total_count += 1
        uncertainty = np.sqrt(np.log(self.total_count)/(2*self.counts+1))  # Calculate the uncertainty measure
        ucb = self.estimates + self.coef * uncertainty  # # Calculate the upper confidence bound
        action = np.argmax(ucb, axis=1)  # Select the action with the largest upper confidence bound
        reward, cur_res, clu_res = self.bandit.step(action, gnn_env, past_res)
        reward = np.mean(reward)
        self.estimates[np.arange(self.bandit.num_nodes), action] += 1. / (self.counts[np.arange(self.bandit.num_nodes), action] + 1) * (reward - self.estimates[np.arange(self.bandit.num_nodes), action])
        return action, reward, cur_res, clu_res

