import pdb
from Network import *
import torch
import torch.nn as nn
from torch.autograd import Variable

class xDeepFM(object):
    def __init__(self, field_num, l):
        self.mn = "CIN.pth"
        self.net = Network(field_num, l)
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate )
        self.loss_func = nn.BCELoss()
        self.field_num = field_num
        self.d = l
        self.load()

    def load(self):
        import os
        if os.path.exists(self.mn):
            print("Load model parameters.")
            self.net.load_state_dict(torch.load(self.mn) )

    def save(self):
        print("Save model parameters.")
        torch.save(self.net.state_dict(), self.mn )

    def train(self, features, labels ):
        self.net.train()
        batch_num = features[0][0][0].size()[0]
        y_predict = self.predict(features, batch_num ); #pdb.set_trace()
        y_true = torch.randn(batch_num )
        for i in range(batch_num ):
            y_true[i] = labels[i]

        loss = self.loss_func(y_predict, y_true ); #pdb.set_trace()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, features, labels ):
        self.net.eval()
        batch_num = features[0][0][0].size()[0]
        y_predict = self.predict(features, batch_num)
        y_true = torch.randn(batch_num)
        for i in range(batch_num ):
            y_true[i] = labels[i]

        return self.loss_func(y_predict, y_true)

    def predict(self, features, batch_num):
        pre = torch.randn(batch_num )
        for i in range(batch_num ):
            xi = []; xv = []
            for j in range(self.field_num ):
                xi.append(features[0][j][0].numpy()[i] )
                xv.append(features[1][j][0].numpy()[i] )

            pre[i] = self.net(xi, xv)

        return pre