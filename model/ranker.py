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
        self.gru=nn.GRU(
            input_size=params.embedding_size,
            hidden_size=params.gru_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        # self.conv=Conv(params)
        self.fc=nn.Linear(4*self.params.gru_hidden,self.params.hidden)
        self.dropout_linear = nn.Dropout(p=params.dropout)
        self.out=nn.Linear(self.params.hidden,1)

    def forward(self,input,dropout=0.0):
        '''
        input [batch_size,seq_len]
        output [batch_size]
        '''
        (batch_size,_)=input.size()
        embedding=self.embedding(input)
        # conv=self.conv(embedding)
        # conv=F.dropout(conv,dropout)
        _,hidden=self.gru(embedding)
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        output=self.fc(hidden.view(-1,4*self.params.gru_hidden))
        output=F.tanh(output)
        output=self.out(output)
        output=F.sigmoid(output)
        return output.view(batch_size,-1)


    def trainer(self,optimizer,loss_f):
        def train(batch,use_cuda=True,dropout=0.5):
            X,y=batch
            input=Variable(t.from_numpy(X),requires_grad=False).long()
            label=Variable(t.from_numpy(y),requires_grad=False).float()
            if use_cuda:
                #sorry
                input=input.cuda()
                label=label.cuda()
            y=self.forward(input,dropout)
            loss=loss_f(y,label)
            return loss
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # return loss.data
        return train

    def batchClassify(self,input,dropout=0.0,use_cuda=True):
        '''

        input [batch, seq_len]

        return [batch]

        '''
        input=Variable(input,volatile=True)
        return self.forward(input,dropout).data

    def validater(self,loss_f):
        def validate(batch,use_cuda=True):
            X,y=batch
            input=Variable(t.from_numpy(X),volatile=True).long()
            label=Variable(t.from_numpy(y),volatile=True).float()
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

            


    


