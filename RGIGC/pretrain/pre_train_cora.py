import os
import copy
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from evaluation import eva
import argparse
import random
from gnn_model import GAT
from utils import data_prepare, save_result
import os.path as osp
from torch_geometric.datasets import Planetoid
import time



class Pre_Train(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, premodel_path):
        super(Pre_Train, self).__init__()
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(premodel_path, map_location='cpu'))

    def forward(self, inputs, adj):
        A_pred, z = self.gat(inputs, adj)
        return A_pred, z

def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

def trainer(args, dataset):
    # data process
    # 1.get dataset
    path = r"..\\data\\cora"
    dataset_all = Planetoid(path, dataset)
    features, adj_ori, adj_norm, adj_lab, label = data_prepare(dataset_all[0])

    args.num_features = features.shape[1]
    #args.n_classes = label.max()+1
    args.n_classes = label.max()+1-1

    model = GAT(num_features=args.num_features, hidden_size=args.hidden_size,
                         embedding_size=args.embedding_size, alpha=args.alpha).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_res = [0]
    consecutive_epoch = 0
    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(features, adj_norm)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_lab.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(features, adj_norm)
            # get kmeans and pretrain cluster result
            kmeans = KMeans(n_clusters=args.n_classes, n_init=20)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            res = eva(label, y_pred, epoch)
            print("Epoch: {} loss: {:.4f} acc: {} nmi: {} f1: {} ari: {}".format(epoch, loss, res[0], res[1], res[2], res[3]))
            if res[0] >= best_res[0]:
                best_res = res
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                consecutive_epoch = 0
            else:
                consecutive_epoch += 1

            if consecutive_epoch > args.patience:
                #bestmodel_path = os.path.join(args.premodel_path, "premodel_{}_{}s.pkl".format(dataset, best_epoch))
                bestmodel_path = os.path.join(args.premodel_path, "premodel_{}_{}_{}.pkl".format(dataset, args.hidden_size, args.embedding_size))
                torch.save(best_model, bestmodel_path)
                bestres_path = os.path.join(args.premodel_path, "premodel_{}_{}s_bestres.txt".format(dataset, best_epoch))
                best_resparas = copy.deepcopy(best_res)
                best_resparas.append(args.patience)
                best_resparas.append(args.lr)
                best_resparas.append(args.hidden_size)
                best_resparas.append(args.embedding_size)
                best_resparas.append(args.seed)
                save_result(bestres_path, best_resparas)
                print("\nbest_res: {} best_epoch: {}".format(best_res, best_epoch))
                break
    bestmodel_path = os.path.join(args.premodel_path, "premodel_{}_{}_{}.pkl".format(dataset, args.hidden_size, args.embedding_size))
    torch.save(best_model, bestmodel_path)
    bestres_path = os.path.join(args.premodel_path, "premodel_{}_{}s_bestres.txt".format(dataset, best_epoch))
    best_resparas = copy.deepcopy(best_res)
    best_resparas.append(args.patience)
    best_resparas.append(args.lr)
    best_resparas.append(args.hidden_size)
    best_resparas.append(args.embedding_size)
    best_resparas.append(args.seed)
    save_result(bestres_path, best_resparas)
    print("\nbest_res: {} best_epoch: {}".format(best_res, best_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--patience', type=int, default=10)       #10
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)        #0.003
    parser.add_argument('--n_classes', default=5, type=int)    #7
    parser.add_argument('--num_features', default=2708, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)            #0
    parser.add_argument('--premodel_path', type=str, default='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if args.cuda else "cpu")

    premodel_path = r"..\pretrain_res\cora"
    args.premodel_path = premodel_path

    sta = time.time()
    #running
    set_seed(args.seed)
    print("Start training...\n")
    start = time.time()
    trainer(args, args.dataset)
    time_cost = time.time()-start
    print("\nTime cost: {}s.".format(time_cost))

    print("ALL time cost: {}s".format(time.time()-sta))










