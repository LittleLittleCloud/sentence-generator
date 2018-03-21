<<<<<<< HEAD
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,size,num_layers,f):
        super(Highway,self).__init__()
        self.num_layers=num_layers

        self.nonlinear=[nn.Linear(size,size) for _ in range(num_layers)]
        for i,module in enumerate(self.nonlinear):
            self._add_to_parameters(module.parameters(),'nonlinear_{}'.format(i))
        
        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), 'linear_{}'.format(i))

        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), 'gate_{}'.format(i))

        self.f=f

    def forward(self,x):
        '''
            x: [batch_size,dim]
            return: [batch_size,dim]
        
        '''
        for layer in range(self.num_layers):
            gate=F.sigmoid(self.gate[layer](x))
            non_linear=self.f(self.nonlinear[layer](x))
            linear=self.linear[layer](x)

            x=gate*non_linear+(1-gate)*linear

        return x

    def _add_to_parameters(self,params,name):
        for i,param in enumerate(params):
=======
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,size,num_layers,f):
        super(Highway,self).__init__()
        self.num_layers=num_layers

        self.nonlinear=[nn.Linear(size,size) for _ in range(num_layers)]
        for i,module in enumerate(self.nonlinear):
            self._add_to_parameters(module.parameters(),'nonlinear_{}'.format(i))
        
        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), 'linear_{}'.format(i))

        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), 'gate_{}'.format(i))

        self.f=f

    def forward(self,x):
        '''
            x: [batch_size,dim]
            return: [batch_size,dim]
        
        '''
        for layer in range(self.num_layers):
            gate=F.sigmoid(self.gate[layer](x))
            non_linear=self.f(self.nonlinear[layer](x))
            linear=self.linear[layer](x)

            x=gate*non_linear+(1-gate)*linear

        return x

    def _add_to_parameters(self,params,name):
        for i,param in enumerate(params):
>>>>>>> freegpu/master
            self.register_parameter(name='{}-{}'.format(name,i),param=param)