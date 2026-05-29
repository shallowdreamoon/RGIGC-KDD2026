import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Reinforced_GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(Reinforced_GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, action, breadth_policy=None, deepth_policy=None, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)

        if breadth_policy == True:
            # The following operation selects the top-k neighbors for aggregation.
            # Use torch.topk to find the largest k values and their indices in each row.
            top_values, top_indices = torch.topk(attention, action.max()+1, dim=1, largest=True, sorted=True)
            thresholds = top_values[torch.arange(top_values.shape[0]), action.long()]
            thresholds_expand = thresholds.unsqueeze(1).expand_as(attention)
            attention = torch.where(attention<thresholds_expand, torch.tensor(0.), attention)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class Reinforced_GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, max_layer=5):
        super(Reinforced_GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv_hid = []
        self.conv1 = Reinforced_GATLayer(num_features, hidden_size, alpha)
        for i in range(1, max_layer):
            self.conv_hid.append(Reinforced_GATLayer(hidden_size, hidden_size, alpha))
        self.conv2 = Reinforced_GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, action, breadth_policy=None, deepth_policy=None):
        breadth_action = action[0]
        deepth_action = action[1]
        #print(deepth_action.max())

        if breadth_policy == True or deepth_policy == True:
            buffers = {i: [] for i in range(deepth_action.max() + 1)}
            for idx, act in enumerate(deepth_action):
                buffers[act.item()].append(idx)

            h = self.conv1(x, adj, breadth_action, breadth_policy, deepth_policy)
            if deepth_action.max() == 0:
                h = self.conv2(h, adj, breadth_action, breadth_policy, deepth_policy)
            else:
                temp_h = h.clone()
                temp_save = []
                for a in range(1, deepth_action.max() + 1):  # 对于action=0的不再进行卷积；相当于所有的节点再进行action.max()层的卷积
                    temp_h = self.conv_hid[a - 1](temp_h, adj, breadth_action, breadth_policy, deepth_policy)
                    temp_save.append([buffers[a], temp_h.clone()])
                for i in range(len(temp_save)):
                    idx, te_h = temp_save[i][0], temp_save[i][1]
                    h[idx] = te_h[idx]
                h = self.conv2(h, adj, breadth_action, breadth_policy, deepth_policy)
        else:
            h = self.conv1(x, adj, action, breadth_policy, deepth_policy)
            h = self.conv2(h, adj, action, breadth_policy, deepth_policy)

        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z


    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred