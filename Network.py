import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Network(nn.Module):
    def __init__(self, field_num, l):
        super(Network, self).__init__()
        self.field_num = field_num
        self.emb_size = 10
        self.embedding = []
        for i in range(field_num):
            self.embedding.append(nn.Embedding(l[i], self.emb_size) )

        self.layer_sizes = [self.field_num, 100, 100, 50 ]
        self.convolution = []
        for i in range(len(self.layer_sizes) ):
            self.convolution.append([])

        for i in range(len(self.layer_sizes) ):
            if i >= len(self.layer_sizes)-1:
                continue

            for j in range(self.layer_sizes[i+1] ):
                tmp = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (self.layer_sizes[i], self.field_num) )
                self.convolution[i].append(tmp)

        self.out = nn.Sequential(
            nn.Linear(100 + 100 + 50, 1),
            nn.Sigmoid()
        )

    def forward(self, xi, xv):
        matrix_0 = torch.randn(self.field_num, self.emb_size)
        for i in range(len(xi) ):
            matrix_0[i] = self.embedding[i](torch.tensor(xi[i] ) ) * xv[i]

        matrix_0t = matrix_0.t()
        hidden_layers = []; hidden_layers.append(matrix_0 )
        pooling = []

        for i in range(len(self.layer_sizes) ):
            matrix_h = hidden_layers[-1]; m = len(matrix_h )
            matrix_z = torch.randn(self.field_num, m, self.field_num )
            for j in range(self.emb_size ):
                matrix_z[j] = matrix_h[:,j].reshape(-1, 1).mm(matrix_0t[j].reshape(1, -1) )

            if i >= len(self.layer_sizes)-1:
                continue

            next_matrixh = torch.randn(self.layer_sizes[i + 1], len(matrix_z) )
            for j in range(self.layer_sizes[i+1] ):
                for k in range(len(matrix_z) ):
                    next_matrixh[j][k] = self.convolution[i][j](matrix_z[k].reshape(1, 1, m, -1) )

                pooling.append(next_matrixh[j].sum() )

            hidden_layers.append(next_matrixh)

        return self.out(torch.tensor(pooling ) )