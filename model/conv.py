import torch as t
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

class Conv(nn.Module):
    def __init__(self,params):
        super(Conv,self).__init__()

        self.params=params
        self.kernels= [Parameter(t.Tensor(oc,params.embedding_size,kw)) for (kw,oc) in self.params.kernels]
        self._add_to_parameters(self.kernels,'kernels')

    def forward(self,x):
        [batch_size,seq_len,embedding_size]=x.size()

        assert embedding_size==self.params.embedding_size, "embedding size not match"

        x=x.transpose(1,2).contiguous()
        xs=[F.tanh(F.conv1d(x,kernel)).max(2)[0] for kernel in self.kernels] # not squeeze in 0.3
        output=t.cat(xs,1)

        return output

    def _add_to_parameters(self,parameters,name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)