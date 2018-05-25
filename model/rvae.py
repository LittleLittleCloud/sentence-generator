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
        self.encoder=Encoder(params)
        self.decoder=Decoder(params)
        self.logvar=nn.Linear(params.encode_rnn_size*2,params.latent_variable_size)
        self.mu=nn.Linear(params.encode_rnn_size*2,params.latent_variable_size)
        self.params=params
        self.embedding=Embedding(params)
        self.ranker=nn.Linear(params.latent_variable_size,1)
        # self.latent=nn.Linear(params.latent_variable_size,params.decode_rnn_size*2)
        self.latent=nn.Linear(params.encode_rnn_size*2,params.decode_rnn_size)
        self.i=1
        self.use_cuda=params.use_cuda
        self.kl_weight=lambda i: (math.tanh((i - 3000)/1000) + 1)/2
        

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
            final_hidden_state,final_cell_state=self.encoder(encode_input) 
            [batch,hidden_size]=final_hidden_state.size()
            # final_hidden_state=final_hidden_state.view(-1,hidden_size)
            # final_cell_state=final_cell_state.view(-1,hidden_size)
            logvar=self.logvar(final_hidden_state)  #make sure logvar in 
            mu=self.mu(final_hidden_state)
            std=t.exp(0.5*logvar)
            e=Variable(t.rand(batch,self.params.latent_variable_size).normal_())
            if use_cuda:
                e=e.cuda()
            z=mu+std*e
            # print(t.sum((std*z)**2,1))
            # z=mu#+z*std

            KLD=-0.5*t.sum(1+logvar-t.pow(mu,2)-t.exp(logvar),1)
            KLD=KLD.mean()
            
            
            
        else:
            KLD=None
        # init_state=t.cat([final_hidden_state,final_cell_state],0)
        # print(init_state.size())
        # init_state=self.latent(init_state).view(-1,1,batch,self.params.decode_rnn_size)
        # if init_state is None:
        #     init_state=F.relu(self.latent(z)).view(-1,1,batch,self.params.decode_rnn_size)
        return None,KLD,z,mu

        # decode_final_state=self.decoder(decode_input,z,drop_rate,(init_state[0],init_state[1]),concat=True)
        # return decode_final_state[0],decode_final_state[1],KLD

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]


    def REC_LOSS(self,batch,dropout,use_cuda,use_teacher=True):
        encode_input,decode_input,target,real_seq_len=batch
        encode_input=Variable(t.from_numpy(encode_input)).long()
        decode_input=Variable(t.from_numpy(decode_input)).long()
        target=Variable(t.from_numpy(target)).long()
        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
            decode_input=decode_input.cuda()
            target=target.cuda()

        hidden,kld,z,_=self.forward(encode_input)
        decode_input=self.embedding(decode_input) #[batch,seq_len,embedding_size]
        [batch,seq_len,embedding_size]=decode_input.size()
        
        if use_teacher:
            out,_=self.decoder.forward(decode_input,z,dropout,hidden,True)
        else:
            res=[]
            input=decode_input[:,0,:].contiguous().view(batch,1,embedding_size)
            for i in range(seq_len):
                out,hidden=self.decoder.forward(input,z,0,hidden,True)
                res+=[out.data]

                input=t.multinomial(F.softmax(out,dim=1),1)
                input=self.embedding(input)
            out=Variable(t.cat(res,0),requires_grad=True)
        rec_loss=F.cross_entropy(out,target.contiguous().view(-1))
        # print((out.view(-1).topk(1)[1]==target.view(-1)).data.cpu().numpy())
        i=self.i           
        self.i+=1
        return rec_loss,kld,self.kl_weight(i)

    def RANKER_MSE_LOSS(self,batch,use_cuda):
        X,y=batch
        input=Variable(t.from_numpy(X),volatile=True).long()
        label=Variable(t.from_numpy(y),volatile=True).float()
        if use_cuda:
            #sorry
            input=input.cuda()
            label=label.cuda()
        _,_,_,z=self.forward(input) #z:[batch,latent]
        z=Variable(z.data)
        target=F.sigmoid(self.ranker(z)) #target: [batch]
        loss=F.mse_loss(target,label)
        return loss
    
    def ranker_validator(self,batch,use_cuda):
        X,y=batch
        input=Variable(t.from_numpy(X),volatile=True).long()
        label=Variable(t.from_numpy(y),volatile=True).float()
        if use_cuda:
            #sorry
            input=input.cuda()
            label=label.cuda()
        _,_,_,z=self.forward(input) #z:[batch,latent]
        target=F.sigmoid(self.ranker(z)) #target: [batch]
        loss=F.mse_loss(target,label)
        return loss.data           

    def batchClassify(self,encode_input,use_cuda=True):
        input=Variable(encode_input,volatile=True).long()
        if use_cuda:
            #sorry
            input=input.cuda()
        _,_,_,z=self.forward(input) #z:[batch,latent]
        target=F.sigmoid(self.ranker(z)) #target: [batch]
        return target.data


    def REENCODE_LOSS(self,encode_input,use_cuda):
        '''

            the distance between latten(x) and latten(encode(decode(latten(x)))) 

        '''
        encode_input,_,_,_=encode_input

        sample_input=self.sample(encode_input,use_cuda)
        
        encode_input=Variable(t.from_numpy(encode_input)).long()
        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()

        _,_,z_origin,_=self.forward(encode_input)
        _,_,z_new,_=self.forward(Variable(sample_input))


        return t.sum((z_origin-z_new)**2,1).mean()


    def PG_LOSS(self,batch,dropout,use_cuda,dis,use_teacher=True,rollout=1):
        '''

            dis: discriminator that provide batchclassify
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
        
        hidden,_,z,_=self.forward(encode_input)
        decode_input=self.embedding(decode_input) #[batch,seq_len,embedding_size]

        # decode_input=decode_input.permute(1,0,2) #[seq_len,batch,embedding_size]
        # target=target.permute(1,0)             #[seq_len,batch]
        [batch,_,embedding_size]=decode_input.size()
        [_,seq_len]=target.size()
        input=decode_input[:,0,:].contiguous().view(batch,1,embedding_size)
        pg_loss=0
        rewards=t.from_numpy(np.zeros((batch,seq_len-1))).float().cuda()
        start=Variable(t.from_numpy(np.zeros((batch,1))).long().cuda())
        for i in range(1,seq_len):
            out,hidden=self.decoder.forward(input,z,0.1,hidden,True)
            # hidden.detach_()


            if use_teacher:
                input=decode_input[:,i,:].contiguous().view(batch,1,embedding_size)
            else:
                input=t.multinomial(F.softmax(out), 1)
                input=self.embedding(input)

            # do a rollout
            for _ in range(rollout):
                if i==1:
                    sample=start.clone()
                else:
                    sample=t.cat((start,target[:,:i-1]),1)
                #sorry for that
                hidden_=(hidden[0].clone(),hidden[1].clone())

                input_=input.clone()
                res=[sample.data]
                for j in range(i,seq_len):
                    out_,hidden_=self.decoder.forward(input_,z,0.1,hidden_,True)
                    input_=t.multinomial(F.softmax(out_),1)
                    res+=[input_.data]
                    input_=self.embedding(input_)
                sample=t.cat(res,1)
                reward=dis.batchClassify(sample[:,1:])
                # print(reward)
                rewards[:,i-1]+=(reward/rollout).view(-1)
                
            for j in range(batch):
                pg_loss+=F.log_softmax(out,1)[j][target.data[j][i-1]]*rewards[j,i-1]
        return -pg_loss/batch

    def SAMPLE_PG_LOSS(self,batch,max_len,use_cuda,dis,output=True):
        encode_input=Variable(t.from_numpy(batch)).long()
        [batch,_]=encode_input.size()
        if use_cuda:
            encode_input=encode_input.cuda()
        _,_,seed,_=self.forward(encode_input)
        # seed=Variable(t.rand(batch,self.params.latent_variable_size))
        seed=seed.detach()
        res=[]
        decode_input=np.zeros((batch,1))
        decode_input=Variable(t.from_numpy(decode_input)).long()
        

        if use_cuda:
            seed=seed.cuda()
            decode_input=decode_input.cuda()

        decode_input=self.embedding(decode_input)
        loss=[0]*batch
        # hidden=F.relu(self.latent(seed)).view(-1,1,1,self.params.decode_rnn_size)
        hidden=None
        for i in range(max_len):
            out, hidden=self.decoder.forward(decode_input,seed,0,hidden)
            word=t.multinomial(F.softmax(out,1), 1)

            res+=[word.data]
            decode_input=self.embedding(word)
            for j in range(batch):
                loss[j]-=F.log_softmax(out)[j][word[j]]
        
        res=t.cat(res,1)
        rewards=dis.batchClassify(res).cpu().numpy().reshape(-1).tolist()
        if output:
            print("rewards: ",rewards)
        # rewards=np.log(rewards).tolist()
        loss=[x*rewards[i] for i,x in enumerate(loss)]
        
        loss=sum(loss)/batch
        return loss


    def sample(self,encode_input,use_cuda,beam_search_k=1):
        [batch,seq_len]=encode_input.shape
        encode_input=Variable(t.from_numpy(encode_input),volatile=True).long()
        if use_cuda:
            #sorry
            encode_input=encode_input.cuda()
        
        res=np.zeros((batch,seq_len),dtype=np.int32)
        _,_,z,_=self.forward(encode_input)
        z=z.data.cpu().numpy()
        for b in range(batch):
            _res=self.base_sample(seq_len,use_cuda,z[b],beam_search_k)
            res[b]=_res
        return res
    
    def base_sample(self,seq_len,use_cuda,z=None,beam_search_k=1):
        if z is None:
            seed=Variable(t.rand(self.params.latent_variable_size),volatile=True)
        else:
            assert z.shape==(self.params.latent_variable_size,)
            seed=Variable(t.from_numpy(z),volatile=True).float()
        seed=seed.view(1,-1)
        res=[[0]*seq_len]*beam_search_k #[beam_search_k, seq_len]    
        decode_inputs=np.array([[0]]*beam_search_k) #[beam_search_k,1]
        decode_inputs=Variable(t.from_numpy(decode_inputs),volatile=True).long()
        

        if use_cuda:
            seed=seed.cuda()
            decode_inputs=decode_inputs.cuda()

        decode_inputs=self.embedding(decode_inputs)
        decode_inputs=decode_inputs.view(beam_search_k,1,1,self.params.word_embed_size)
        probs=np.zeros(beam_search_k).reshape(beam_search_k,-1)
        # hidden=F.relu(self.latent(seed)).view(-1,1,1,self.params.decode_rnn_size)
        hiddens=[None]*beam_search_k    
        for i in range(seq_len):
            _probs=np.repeat(probs,beam_search_k,1) #[beam_search_k,beam_search_k]
            _res=np.repeat(res,beam_search_k,0).reshape(beam_search_k,beam_search_k,-1)
            _words=np.zeros((beam_search_k,beam_search_k))
            for j,decode_input in enumerate(decode_inputs):
                out, hiddens[j]=self.decoder.forward(decode_input,seed,0,hiddens[j])
                out=F.softmax(out,1)
                word=t.multinomial(out, beam_search_k,True).data.cpu().numpy()[0]
                out=out.data.cpu().numpy()[0]
                _probs[j]=_probs[j]-np.log(out[word])
                # c=np.concatenate((_res[j],word.reshape(beam_search_k,-1)),axis=1)
                _res[j,:,i]=word
                _words[j]=word
                # res[j]+=[word[0]]
            _probs=_probs.reshape(-1)
            _res=_res.reshape(beam_search_k*beam_search_k,-1)
            _words=_words.reshape(-1)
            index=np.argsort(_probs)[:beam_search_k]

            new_hidden=[]
            for j in range(beam_search_k):
                new_hidden+=[hiddens[int(index[j]/beam_search_k)]]
            hiddens=new_hidden

            probs=_probs[index].reshape(beam_search_k,-1)
            res=_res[index]
            word=_words[index]
            decode_inputs=np.array([word])
            decode_inputs=Variable(t.from_numpy(decode_inputs),volatile=True).long()
            if use_cuda:
                decode_inputs=decode_inputs.cuda()
            decode_inputs=self.embedding(decode_inputs)
            decode_inputs=decode_inputs.view(beam_search_k,1,1,self.params.word_embed_size)
            
        index=np.argmin(probs)
        res=res[index]
        return res

    def random_sample(self,n,length,use_cuda,beam_search_k=1):
        '''

            n: sample n
            return results [n,seq_len]

        '''
        res=np.zeros((n,length),dtype=np.int32)
        for b in range(n):
            res[b]=self.base_sample(length,use_cuda,None,beam_search_k)
        return res



