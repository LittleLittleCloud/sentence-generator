from model.rvae import RVAE
from util.preprocess import Preprocess
from util.parameter import Parameter
from gensim.models import KeyedVectors
from torch.optim import Adam
import numpy as np
import torch as t
from util.batch_loader import Batch as Batch
from util.batch_loader2 import Batch as Batch2
from util.parameter2 import Parameter as Parameter2
import torch.nn as nn
from model.ranker import Ranker
import os
import pandas as pd
from torch.autograd import Variable

DATA_PATH='./data/event_score.csv'
WORD2VEC='../sentence generator/embedding.bin'
PRETRAIN_DIS_PATH='PRETRAIN_DIS_PATH'
PRETRAIN_GEN_PATH='PRETRAIN_GEN_PATH'
embedding_model=KeyedVectors.load_word2vec_format(WORD2VEC)
data=pd.read_csv(DATA_PATH,encoding='utf-8').dropna().values[:,[1,2]]
batch_loader2=Batch2(data,0.9,embedding_model.wv.index2word)
dis_params=Parameter2(batch_loader2.vocab_size,'embedding.npy',100)

use_cuda=t.cuda.is_available()

#step 1 
#pre-train generater

with open('train','r',encoding='utf-8') as f:
    data=f.readlines()
preprocess=Preprocess(embedding_model)
input=preprocess.to_sequence(data)
batch_loader=Batch(input,0.7)
np.save('index2word',preprocess.index_to_word)
params=Parameter(word_embed_size=300,encode_rnn_size=600,latent_variable_size=1000,\
            decode_rnn_size=600,vocab_size=preprocess.vocab_size,embedding_path='embedding.npy',use_cuda=use_cuda)

generator=RVAE(params)

if use_cuda:
    generator=generator.cuda()
    
gen_optimizer=Adam(generator.learnable_parameters(), 1e-3)

test_batch=batch_loader.test_next_batch(1)

if os.path.isfile(PRETRAIN_GEN_PATH):
    print('find PRETRAINED_GEN_FILE')
    generator.load_state_dict(t.load(PRETRAIN_GEN_PATH))
#useless
for i,batch in enumerate(batch_loader.train_next_batch(5)):
    break
    ce,kld,coef=generator.REC_LOSS(batch,0.2,use_cuda)
    loss=79*ce+coef*kld
    gen_optimizer.zero_grad()
    loss.backward()
    gen_optimizer.step()
    # loss.detach_()
    if i%10==0:
        print('ten step: ce:{}, kld:{} '.format(ce,kld))
    del loss,ce,kld



#step2
#pre-train discriminator
params=Parameter2(vocab_size=preprocess.vocab_size,embedding_path='embedding.npy',\
                    embedding_size=100,ranker_hidden_size=32,dropout=0.2)
discriminator=Ranker(params)
if use_cuda:
    discriminator=discriminator.cuda()
# dis_optimizer=Adam([p for p in generator.ranker.parameters()])
dis_optimizer=Adam(discriminator.learnable_parameters())
train_step=discriminator.trainer(dis_optimizer,nn.MSELoss())
# validate=generator.ranker_validator
validate=discriminator.validater(nn.MSELoss())
loss_lst=[]



print('pre-train dis begin')

if os.path.isfile(PRETRAIN_DIS_PATH):
    print('find PRETRAINED_DIS_FILE')
    discriminator.load_state_dict(t.load(PRETRAIN_DIS_PATH))

# for _ in range(1):
#     test_batch=batch_loader2.test_next_batch(10)
#     for i,batch in enumerate(batch_loader2.train_next_batch((10))):
#         loss=train_step(batch,t.cuda.is_available())
#         dis_optimizer.zero_grad()
#         loss.backward()
#         dis_optimizer.step()
#         loss_lst+=[loss.data]
#         if i%100==0:
#             test=next(test_batch)
#             test_loss=validate(test,t.cuda.is_available())
#             print("train: ",sum(loss_lst[-100:])/100)
#             print("test: ",test_loss)
#             t.save(discriminator.state_dict(),PRETRAIN_DIS_PATH)
            

print('pre-train dis finish')
    


#step 3
# train gan
#first train the generator
#then train the discriminator

print('train gan')
seq_data=batch_loader2.train_data
BATCH_SIZE=2
gen_loss=[]
dis_loss=[]
test_input=batch_loader2.test_next_batch(5,raw=True)
for _ in range(10):
    for round,i in enumerate(range(0,len(seq_data),BATCH_SIZE)):

        batch=seq_data[i:i+BATCH_SIZE]
        # target=t.from_numpy(np.array(batch_loader2.train_label[i:i+BATCH_SIZE])).float()
        # if use_cuda:
            # target=target.cuda()
        print('train generator')
        encode_input,_,_=batch_loader.to_input(batch)
        # res=generator.sample(encode_input,use_cuda)
        # rewards=discriminator.batchClassify(res,use_cuda).view(-1) #[b]
        # print(rewards)
        
        # [b,s]=res.size()
        # res=res.cpu().numpy()
        # encode_input=res.copy()
        # decode_input=np.concatenate((np.array([0]*b).reshape(b,-1),encode_input),1)
        # target=np.concatenate((encode_input,np.array([0]*b).reshape(b,-1)),1)
        
        # res=res.view(-1,v)
        # res=res.topk(1)[1]
        # res=res.view(b,s)
        # encode_input,decode_input,_=batch_loader.to_input(batch)

        # loss=generator.PG_LOSS((encode_input,decode_input,target),0,use_cuda,discriminator,True)
        for _ in range(100):
            loss=generator.SAMPLE_PG_LOSS(encode_input,100,True,discriminator,True)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()
            gen_loss+=[loss.data]

        print('train discriminator')
        for _round in range(1):
            #sample positive and negative samples
            pos=batch_loader2.gen_positive_sample(10)
            neg=generator.random_sample_n(10,100,use_cuda)
            print(' '.join([preprocess.index_to_word[i] for i in neg[0][:10]]))

            data=pos[0]+neg
            target=pos[1]+[0]*len(neg)
            index=np.arange(len(data))
            np.random.shuffle(index)
            for i in range(0,len(data),10):
                X=[data[_i] for _i in index[i:i+10]]
                y=[target[_i] for _i in index[i:i+10]]
                max_len=max(len(x) for x in X)

                for i,line in enumerate(X):
                    to_add=max_len-len(line)
                    X[i]=line+[preprocess.word_to_index['_']]*to_add
                
                X=np.array(X)
                y=np.array(y)

                #train
                loss=train_step([X,y],use_cuda=use_cuda)
                dis_optimizer.zero_grad()
                loss.backward()
                dis_optimizer.step()
                dis_loss+=[loss.cpu().data.numpy()[0]]


        if round%10==0:
            print('---PG LOSS---')
            print(sum(gen_loss[-10:])/10)
            print('-------------')

            print('---dis LOSS---')
            print(sum(dis_loss[-10:])/10)
            print('-------------')

            #sample
            try:
                input=next(test_input)
            except:
                test_input=batch_loader2.test_next_batch(5,raw=True)
                input=next(test_input)
            encode_input,_,_=batch_loader.to_input(input[0])

            res=generator.sample(encode_input,use_cuda)
            # res=res.view(b,-1)
            y=discriminator.batchClassify(res,use_cuda=True).view(-1).cpu().numpy()
            loss=((y-input[1])**2).mean()
            print('---SAMPLE LOSS---\n {}'.format(loss))
            print('-------------')
            print(' '.join([preprocess.index_to_word[i] for i in res[0][:10]]))
            
            
            # save model
            t.save(discriminator.state_dict(),PRETRAIN_DIS_PATH)
            t.save(generator.state_dict(),PRETRAIN_GEN_PATH)
            # print('sample result: ')
            # res=res.cpu().numpy()
            # print(res)
            # for sentence in res:
            #     sentence=[preprocess.index_to_word[i] for i in sentence]
            #     print(' '.join(sentence))
        



        



    











