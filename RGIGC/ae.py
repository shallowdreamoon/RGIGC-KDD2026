from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

class AE(nn.Module):

    def __init__(self, num_features, hidden_size, embedding_size, alpha=0.2):
        super(AE, self).__init__()
        self.enc_1 = Linear(num_features, hidden_size)
        self.enc_2 = Linear(hidden_size, embedding_size)

        self.dec_1 = Linear(embedding_size, hidden_size)
        self.dec_2 = Linear(hidden_size, num_features)

        #self.act = nn.LeakyReLU(alpha, inplace=True)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        enc_h1 = self.act(self.enc_1(x))
        ae_z = self.enc_2(enc_h1)

        dec_h1 = self.act(self.dec_1(ae_z))
        ae_x_bar = self.dec_2(dec_h1)
        return ae_z, ae_x_bar, enc_h1
