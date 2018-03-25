import torch as t
import torch.nn as nn
import torch.nn.functional as F 
from .encoder import Encoder
from .decoder import Decoder
from .embedding import Embedding
from torch.autograd import Variable
import numpy as np
import math
class RVAE(nn.Module):
    def __init__(self,params):
        super(RVAE,self).__init__()
        self.kl_weight=0
        self.encoder=Encoder(params)
        self.decoder=Decoder(params)
        self.logvar=nn.Linear(params.encode_rnn_size*2,params.latent_variable_size)
        self.mu=nn.Linear(params.encode_rnn_size*2,params.latent_variable_size)
        self.params=params
        self.embedding=Embedding(params)
        self.latent=nn.Linear(params.latent_variable_size,params.decode_rnn_size*2)
        self.i=Variable(t.FloatTensor(1),requires_grad=False)

    def forward(self, encode_input,decode_input,drop_rate,init_state=None,z=None):

        '''
        encode_input: [batch_size,seq_len]
        decode_input: [batch_size,seq_len+2]
        z: [batch_size,latent_variable_size]
        output: [batch_size,seq_len,vocab_size]
        '''

        use_cuda=self.embedding.word_embed.weight.is_cuda
        if z is None:
            encode_input=self.embedding(encode_input)
            final_state=self.encoder(encode_input) 
            logvar=self.logvar(final_state)
            mu=self.mu(final_state)
            std=t.exp(0.5*logvar)
            z=Variable(std.data.new(std.size()).normal_())
            if use_cuda:
                z=z.cuda()
            z=z*std+mu
            KLD=(-0.5*t.sum(1+logvar-t.pow(mu,2)-t.exp(logvar),1))
            KLD=KLD.mean().squeeze()
            
            
        else:
            KLD=None
        [batch_size,latent_variable_size]=z.size()
        if init_state is None:
            init_state=F.relu(self.latent(z)).view(-1,1,batch_size,self.params.decode_rnn_size)
            print(init_state.size())
        decode_input=self.embedding(decode_input)
        decode_final_state=self.decoder(decode_input,z,drop_rate,(init_state[0],init_state[1]),concat=True)
        return decode_final_state[0],decode_final_state[1],KLD

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer):
        kl_weight=lambda i: (math.tanh((i - 30000)/10000) + 1)/2
        def train(batch,dropout,use_cuda):
            encode_input,decode_input,target=batch

            encode_input=Variable(t.from_numpy(encode_input)).long()
            decode_input=Variable(t.from_numpy(decode_input)).long()
            target=Variable(t.from_numpy(target)).long().view(-1)


            if use_cuda:
                #sorry
                encode_input=encode_input.cuda()
                decode_input=decode_input.cuda()
                target=target.cuda()


            result,_,kld=self(encode_input,decode_input,dropout,init_state=None,z=None)
            result=result.view(-1,self.params.vocab_size)
            ce=F.cross_entropy(result,target)
            i=self.i.data.cpu().numpy()[0]           
            loss=79*ce+kl_weight(i)*kld
            self.i+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return ce.data,kld.data,kl_weight(i)

        return train

    def sample(self,seq_len,batch,use_cuda):
        encode_input,decode_input=batch
        encode_input=Variable(t.from_numpy(encode_input)).long()
        decode_input=Variable(t.from_numpy(decode_input)).long()

        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
            decode_input=decode_input.cuda()
        
        encode_input=self.embedding(encode_input)

        final_state=self.encoder(encode_input)
        logvar=self.logvar(final_state)
        mu=self.mu(final_state)
        std=t.exp(0.5*logvar)
        z=Variable(std.data.new(std.size()).normal_())
        if use_cuda:
            z=z.cuda()
        z=z*std+mu

        res=[]
        [batch_size,_]=z.size()        
        init_state=F.relu(self.latent(z)).view(-1,1,batch_size,self.params.decode_rnn_size)
        print(init_state.size())
        for i in range(seq_len):
            logits,init_state,_=self(None,decode_input,0.0,init_state=init_state,z=z)
            logits=logits.view(-1,self.params.vocab_size)
            prediction=F.softmax(logits)

            word=np.random.choice(np.arange(self.params.vocab_size),p=prediction.data.cpu().numpy()[-1])

            #the end token
            if word==1:
                break
            res+=[word]
            decode_input=np.array([[word]]).reshape(1,-1)
            decode_input=Variable(t.from_numpy(decode_input).long())
            if use_cuda:
                decode_input=decode_input.cuda()
        return res
            



