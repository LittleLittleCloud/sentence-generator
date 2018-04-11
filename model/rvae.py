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
        # self.latent=nn.Linear(params.latent_variable_size,params.decode_rnn_size*2)
        self.latent=nn.Linear(params.latent_variable_size,params.decode_rnn_size)
        
        self.i=Variable(t.FloatTensor(1),requires_grad=False)
        self.use_cuda=params.use_cuda
        self.kl_weight=lambda i: (math.tanh((i - 30000)/10000) + 1)/2
        

    def forward(self, encode_input,z=None,init_state=None):

        '''
        encode_input: [batch_size,seq_len]
        decode_input: [batch_size,seq_len+1]
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
            KLD=KLD.mean()
            
            
            
        else:
            KLD=None
        [batch_size,latent_variable_size]=z.size()
        if init_state is None:
            init_state=F.relu(self.latent(z)).view(-1,batch_size,self.params.decode_rnn_size)
        return init_state,KLD,z

        # decode_final_state=self.decoder(decode_input,z,drop_rate,(init_state[0],init_state[1]),concat=True)
        # return decode_final_state[0],decode_final_state[1],KLD

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]


    def REC_LOSS(self,batch,dropout,use_cuda,use_teacher=True):
        encode_input,decode_input,target=batch
        encode_input=Variable(t.from_numpy(encode_input)).long()
        decode_input=Variable(t.from_numpy(decode_input)).long()
        target=Variable(t.from_numpy(target)).long()
        
        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
            decode_input=decode_input.cuda()
            target=target.cuda()
        
        hidden,kld,z=self.forward(encode_input)
        decode_input=self.embedding(decode_input) #[batch,seq_len,embedding_size]
        decode_input=decode_input.permute(1,0,2) #[seq_len,batch,embedding_size]
        target=target.permute(1,0)
        [seq_len,batch,embedding_size]=decode_input.size()
        input=decode_input[0].contiguous().view(batch,1,embedding_size)
        rec_loss=0
        for i in range(1,seq_len):
            hidden.detach_()
            
            out,hidden=self.decoder.forward(input,z,0.1,hidden,True)
            # out.detach_()
            if use_teacher:
                input=decode_input[i].contiguous().view(batch,1,embedding_size)
            else:
                input=Variable(out.data.topk(1)[1])

                input=self.embedding(input)

            rec_loss+=F.cross_entropy(out,target[i-1])
        i=self.i.data.cpu().numpy()[0]           
        self.i+=1
        return rec_loss/seq_len,kld,self.kl_weight(i)


    def PG_LOSS(self,batch,dropout,use_cuda,rewards,use_teacher=True):
        '''

            rewards: [batch]
            see http://karpathy.github.io/2016/05/31/rl/ for detail

        '''

        encode_input,decode_input,target=batch
        encode_input=Variable(t.from_numpy(encode_input)).long()
        decode_input=Variable(t.from_numpy(decode_input)).long()
        target=Variable(t.from_numpy(target)).long()
        
        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
            decode_input=decode_input.cuda()
            target=target.cuda()
        
        hidden,_,z=self.forward(encode_input)
        decode_input=self.embedding(decode_input) #[batch,seq_len,embedding_size]

        decode_input=decode_input.permute(1,0,2) #[seq_len,batch,embedding_size]
        target=target.permute(1,0)             #[seq_len,batch]
        [seq_len,batch,embedding_size]=decode_input.size()
        input=decode_input[0].contiguous().view(batch,1,embedding_size)
        pg_loss=0
        rewards=Variable(rewards)
        for i in range(1,seq_len):
            out,hidden=self.decoder.forward(input,z,0.1,hidden,True)
            # out.detach_()
            # hidden.detach_()
            if use_teacher:
                input=decode_input[i].contiguous().view(batch,1,embedding_size)
            else:
                input=Variable(out.data.topk(1)[1])
                input=self.embedding(input)
            for j in range(batch):
                pg_loss+=-out[j][target.data[i-1][j]]*rewards[j]
        return pg_loss/batch


    def trainer(self, optimizer):
        kl_weight=lambda i: (math.tanh((i - 30000)/10000) + 1)/2
        def train_rec(batch,dropout,use_cuda):
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
            self.i+=1
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            return ce,kld,kl_weight(i)
            
        def train_pg(batch,dropout,use_cuda,rewards):

            '''

                rewards: [batch]
                see http://karpathy.github.io/2016/05/31/rl/ for detail

            '''
            encode_input,decode_input,target=batch

            encode_input=Variable(t.from_numpy(encode_input)).long()
            decode_input=Variable(t.from_numpy(decode_input)).long()
            target=Variable(t.from_numpy(target)).long() #[batch seq_len]


            if use_cuda:
                #sorry
                encode_input=encode_input.cuda()
                decode_input=decode_input.cuda()
                target=target.cuda()

            result,_,_=self(encode_input,decode_input,dropout,init_state=None,z=None)
            [batch,seq_len,vocab_size]=result.size()            
            result=result.view(-1,self.params.vocab_size)
            result=F.log_softmax(result)
            result=result.view(batch,seq_len,vocab_size) #[batch seq_len vocab_size]
            
            loss=0
            for i in range(batch):
                for j in range(seq_len):
                    loss+=-result[i,j,target.data[i,j]]*rewards[i]  
            
            loss=loss/batch
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            return loss

        return train_rec, train_pg


    def sample(self,encode_input,use_cuda):
        [batch,seq_len]=encode_input.shape
        encode_input=Variable(t.from_numpy(encode_input),volatile=True).long()
        decode_input=np.array([0]*batch).reshape(batch,-1)
        decode_input=Variable(t.from_numpy(decode_input),volatile=True).long()

        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
            decode_input=decode_input.cuda()
        
        encode_input=self.embedding(encode_input)
        decode_input=self.embedding(decode_input)
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
        hidden=F.relu(self.latent(z)).view(-1,batch_size,self.params.decode_rnn_size)
        
        for i in range(seq_len):
            out, hidden=self.decoder.forward(decode_input,z,0,hidden)
            out.detach_()
            hidden.detach_()
            words=out.data.topk(1)[1]
            #the end token
            res+=[words]
            decode_input=words.view(batch,-1)
            decode_input=Variable(decode_input,volatile=True)    
            decode_input=self.embedding(decode_input)
            if use_cuda:
                decode_input=decode_input.cuda()
        return t.cat(res,1)
    
    def random_sample(self,seq_len,use_cuda):
        seed=Variable(t.rand(self.params.latent_variable_size),volatile=True)

        seed=seed.view(1,-1)
        res=[]
        decode_input=np.array([[0]])
        decode_input=Variable(t.from_numpy(decode_input),volatile=True).long()
        

        if use_cuda:
            seed=seed.cuda()
            decode_input=decode_input.cuda()

        decode_input=self.embedding(decode_input)
        
        hidden=F.relu(self.latent(seed)).view(-1,1,self.params.decode_rnn_size)
        for i in range(seq_len):
            out, hidden=self.decoder.forward(decode_input,seed,0,hidden)
            word=np.random.choice(np.arange(self.params.vocab_size),p=out.data.cpu().numpy()[-1])
            out.detach_()
            hidden.detach_()
            #the end token
            if word==1:
                break
            res+=[word]
            decode_input=np.array([[word]])
            decode_input=Variable(t.from_numpy(decode_input),volatile=True).long()
            if use_cuda:
                decode_input=decode_input.cuda()
            decode_input=self.embedding(decode_input)
        return res

    def random_sample_n(self,n,use_cuda):
        '''

            n: sample n
            return results [n,seq_len]

        '''
        results=[]
        for _ in range(n):
            results+=[self.random_sample(50,use_cuda)]
        return results



