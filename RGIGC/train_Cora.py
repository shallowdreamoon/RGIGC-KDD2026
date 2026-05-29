import numpy as np
import gym
import os
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from config import all_seed, set_seed, agent_config, get_args
#from multi_armed_bandit import BernoulliBandit, EpsilonGreedy
from multi_armed_bandit_v1 import BernoulliBandit, UCB
from gnn_env import CustomGNNEnv, data_prepare
from torch_geometric.datasets import Planetoid
import os.path as osp
from utils import data_preprocessing, normalize_adj, sparse_to_tuple
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from evaluation import eva
from utils import save_result, get_action
import time
import copy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

def all_policy_train(cfg, gnn_env, breadth_action, deepth_action, breadth_action_size, deepth_action_size, breadth_model_path, deepth_model_path, breadth_action_path, deepth_action_path, clu_eva_res_path, save = False):
    best_res = [0]
    best_eva_res = [0]
    bandit_arm = BernoulliBandit(gnn_env)
    breadth_epsilon_greedy_solver = UCB(bandit_arm, breadth_action.numpy(), policy="breadth", action_size=breadth_action_size, coef=0.2)  #0.2
    deepth_epsilon_greedy_solver = UCB(bandit_arm, deepth_action.numpy(), policy="deepth", action_size=deepth_action_size, coef=0.2)      #0.2

    breadth_best_action = breadth_action
    deepth_best_action = deepth_action
    breadth_cur_res = best_res
    deepth_cur_res = best_res
    best_gnn_model = gnn_env
    deepth_gnn_model = copy.deepcopy(gnn_env)
    consecutive_step = 0
    breadth_accumulated_reward = 0
    deepth_accumulated_reward = 0
    breadth_best_res = best_eva_res
    deepth_best_res = best_eva_res
    for epoch in range(1):
        #1.neighbor_agent
        print("Neighbor-Agent is optimizing...")
        best_gnn_model.policy = "breadth"
        best_gnn_model.deepth_action = deepth_action
        breadth_best_res, breadth_best_action, breadth_reward, breadth_best_model, breadth_cur_res = breadth_epsilon_greedy_solver.run(100, best_gnn_model, breadth_cur_res, breadth_best_res)
        breadth_accumulated_reward += breadth_reward
        gnn_env.temp = 20
        if breadth_best_res[0] > best_eva_res[0]:
            best_eva_res = breadth_best_res
            consecutive_step = 0
        else:
            consecutive_step += 1
        #2.layer_agent
        print("Layer-Agent is optimizing...")
        deepth_gnn_model.policy = "deepth"
        deepth_gnn_model.breadth_action = breadth_action
        #print("{} policy is optimizing...".format(deepth_gnn_model.policy))
        deepth_best_res, deepth_best_action, deepth_reward, deepth_best_model, deepth_cur_res = deepth_epsilon_greedy_solver.run(100, deepth_gnn_model, deepth_cur_res, deepth_best_res)
        deepth_accumulated_reward += deepth_reward
        if deepth_best_res[0] > best_eva_res[0]:
            best_eva_res = deepth_best_res
            consecutive_step = 0
        else:
            consecutive_step += 1
        if consecutive_step > 80:  #
            break
    if save:
        print("start saving model...")
        torch.save(breadth_best_model, breadth_model_path)
        torch.save(deepth_best_model, deepth_model_path)
        breadth_best_res.append("breadth")
        deepth_best_res.append("deepth")
        save_result(clu_eva_res_path, breadth_best_res)
        save_result(clu_eva_res_path, deepth_best_res)
        save_result(breadth_action_path, breadth_best_action, mode="w")
        save_result(deepth_action_path, deepth_best_action, mode="w")
    print("best_clu_eva_res of breadth_model: {}".format(breadth_best_res))
    print("best_clu_eva_res of deepth_model: {}".format(deepth_best_res))
    return best_eva_res, breadth_best_action, deepth_best_action

# Updates without any reinforcement strategy
def gnn_train(gnn_env, best_res, best_action, best_model_path = "", clu_eva_res_path="", save = False):
    #temp_gnn_env = copy.deepcopy(gnn_env)
    consecutive_step = 0
    for i in range(100):
        temp_model = copy.deepcopy(gnn_env)
        reward, cur_res, clu_res = gnn_env.evaluate(best_action, best_res[0])
        print("epoch: {} res: {}".format(i, clu_res))
        if clu_res[0] > best_res[0]:
            best_res = clu_res
            best_model = temp_model
            consecutive_step = 0
        else:
            consecutive_step += 1
        if consecutive_step > 50:
            break
    if save:
        print("start saving model...")
        torch.save(best_model, best_model_path)
        best_res.append("ALL")
        save_result(clu_eva_res_path, best_res)
    return best_res


def get_data(dataset):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
    dataset_all = Planetoid(path, dataset)
    features, adj_ori, adj_norm, adj_lab, label = data_prepare(dataset_all[0], name="cora")
    label_indices = np.arange(features.shape[0])
    att_sim = cosine_similarity(features.numpy())
    att_sim = torch.from_numpy(att_sim)
    att_sim = adj_ori * att_sim
    return features, adj_ori, adj_norm, adj_lab, label, label_indices, att_sim

def task(cfg, dataset, dh, do, policy = None):
    features, adj_ori, adj_norm, adj_lab, label, label_indices, att_sim = get_data(dataset)

    # 3.get_paras
    # (1) public_paras
    num_nodes = features.shape[0]
    state_size = features.shape[1]
    init_state = features
    recent_step = 5
    num_features = features.shape[1]
    hidden_size = dh
    embedding_size = do
    alpha = 0.2
    # (2) neighbor_agent_paras
    breadth_init_action = adj_lab.sum(dim=1).int()
    breadth_action_size = breadth_init_action.max()
    # (3) layer_agent_paras
    deepth_action_size = 3
    deepth_init_action = torch.randint(low=0, high=1, size=(features.shape[0],))  #low=0, high=1

    # 4. Create an instance of the custom environment
    premodel_path = r".\pretrain_res\{}\premodel_cora_{}_{}.pkl".format(dataset, dh, do)
    gnn_env = CustomGNNEnv(state_size, breadth_action_size, init_state, recent_step,
                           num_features, hidden_size, embedding_size, alpha, cfg['gnn_lr'],
                           adj_ori, adj_norm, adj_lab, features, att_sim, label, policy=None,
                           breadth_action=breadth_init_action, deepth_action=deepth_init_action,
                           premodel_path=premodel_path, label_indices=label_indices)

    # 5. Initialize model parameters (cluster centers)
    agent_config(gnn_env, cfg)
    with torch.no_grad():
        _, z = gnn_env.reinforced_gnn_model.gat(features, adj_norm, breadth_init_action)
    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=label.max() + 1, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    gnn_env.reinforced_gnn_model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_)
    gnn_env1 = copy.deepcopy(gnn_env)
    res = eva(label, y_pred)
    print("Epoch: {} acc: {} nmi: {} f1: {} ari: {}".format(-1, res[0], res[1], res[2], res[3]))



    best_model_path = r".\results\{}\best_model\best_model_GNN.pt".format(dataset)
    breadth_model_path = r".\results\{}\best_model\best_model_BreadthWithGNN.pt".format(dataset)
    breadth_action_path = r".\results\{}\best_model\best_action_BreadthWithGNN.txt".format(dataset)
    deepth_model_path = r".\results\{}\best_model\best_model_DeepthWithGNN.pt".format(dataset)
    deepth_action_path = r".\results\{}\best_model\best_action_DeepthWithGNN.txt".format(dataset)
    all_model_path = r".\results\{}\best_model\best_model_Whole1.pt".format(dataset)
    clu_eva_res_path = r".\results\{}\best_model\clu_eva_res.txt".format(dataset)

    if policy == "Only_GNN":
        print("start training gnn model...\n")
        gnn_env.policy = None
        temp_action = torch.tensor(breadth_init_action)
        best_res = gnn_train(gnn_env, [0], temp_action, best_model_path, clu_eva_res_path, save=False)
        print(best_res)

    elif policy == "Breadth_And_Deepth":
        # Simultaneously optimize the breadth and depth models
        print("start training whole model...\n")
        best_eva_res, breadth_action, deepth_action = all_policy_train(cfg, gnn_env, breadth_init_action,
                                                                       deepth_init_action, breadth_action_size,
                                                                       deepth_action_size, breadth_model_path,
                                                                       deepth_model_path, breadth_action_path,
                                                                       deepth_action_path, clu_eva_res_path, save=False)
        print("\nbest_res of whole model: {}".format(best_eva_res))
        print("\nstart training gnn model...\n")
        gnn_env.policy = "ALL"
        best_action = [torch.tensor(breadth_action), torch.tensor(deepth_action)]
        best_res = gnn_train(gnn_env, [0], best_action, all_model_path, clu_eva_res_path, save=False)
        print(best_res)

    else:
        # Load the optimal model according to the saved optimal model parameters and directly output the saved model results
        gnn_env = torch.load(all_model_path)
        gnn_env.policy = None
        breadth_action = get_action(breadth_action_path)
        deepth_action = get_action(deepth_action_path)
        best_action = [breadth_action, deepth_action]
        reward, cur_res, clu_res = gnn_env.evaluate(best_action, [0])
        print(clu_res)
    return None

if __name__ == '__main__':
    sta = time.time()
    set_seed(seed=10)
    #1.get_paras
    cfg = get_args()
    dataset = "cora"
    dh = 256
    do = 16
    task(cfg, dataset, dh, do, policy="Breadth_And_Deepth")  #Only_GNN, Breadth_And_Deepth

    print("All Time Cost: {}s".format(time.time()-sta))










