import torch as t
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from .embedding import Embedding
from .conv import Conv
from torch.autograd import Variable
from util.batch_loader import Batch
from util.parameter import Parameter


class Ranker(nn.Module):

    def __init__(self,params):
        super(Ranker,self).__init__()

        self.params=params
        self.embedding=Embedding(params)
        self.conv=Conv(params)
        self.fc=nn.Linear(self.params.fc_input_size,1)

    def forward(self,input):
        '''
        input [batch_size,seq_len]
        output [batch_size]
        '''
        (batch_size,_)=input.size()
        embedding=self.embedding(input)
        conv=self.conv(embedding)
        output=self.fc(conv)
        return output.view(batch_size,-1)


    def trainer(self,optimizer,loss_f):
        def train(batch,use_cuda=True):
            X,y=batch
            input=Variable(t.from_numpy(X),requires_grad=False).long()
            label=Variable(t.from_numpy(y),requires_grad=False).float()
            if use_cuda:
                #sorry
                input=input.cuda()
                label=label.cuda()
            y=self.forward(input)
            loss=loss_f(y,label)
            return loss
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # return loss.data
        return train

    def batchClassify(self,input):
        '''

        input [batch, seq_len]

        return [batch]

        '''

        return self.forward(input).detach()

    def validater(self,loss_f):
        def validate(batch,use_cuda=True):
            X,y=batch
            input=Variable(t.from_numpy(X),requires_grad=False).long()
            label=Variable(t.from_numpy(y),requires_grad=False).float()
            if use_cuda:
                #sorry
                input=input.cuda()
                label=label.cuda()
            y=self.forward(input)
            loss=loss_f(y,label)
            return loss.data         
        return validate   

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

            


    


