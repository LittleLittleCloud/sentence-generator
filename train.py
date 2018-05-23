from model.rvae import RVAE
from util.batch_loader import Batch
from util.preprocess import Preprocess
from util.parameter import Parameter
from gensim.models import KeyedVectors
from torch.optim import Adam
import numpy as np
import torch as t
import os
embedding_model=KeyedVectors.load_word2vec_format('embedding.bin')

#load data
data=0
with open('train','r',encoding='utf-8') as f:
    data=f.readlines()


preprocess=Preprocess(embedding_model)
input=preprocess.to_sequence(data)
# embedding=preprocess.embedding()
# np.save('embedding',embedding)

batch_loader=Batch(input,0.7)

params=Parameter(word_embed_size=300,encode_rnn_size=600,latent_variable_size=1000,\
            decode_rnn_size=600,vocab_size=preprocess.vocab_size,embedding_path='embedding.npy',use_cuda=True)
model=RVAE(params)
model=model.cuda()
if os.path.isfile("PRETRAIN_GEN_PATH"):
    model.load_state_dict(t.load("PRETRAIN_GEN_PATH"))
optimizer=Adam(model.learnable_parameters(), 1e-5)

use_cuda=t.cuda.is_available()
ce_list=[]
kld_list=[]
coef_list=[]
re_list=[]
test_batch=batch_loader.test_next_batch(28)
for _ in range(50):
    for i,batch in enumerate(batch_loader.train_next_batch(28)):
        if i%101==0:
            # sample=batch[0][0,:].reshape(1,-1)
            sentence=model.random_sample(50,use_cuda)
            sentence=[preprocess.index_to_word[i] for i in sentence]
            # s=[preprocess.index_to_word[i] for i in sample[0]]
            # print('origin',' '.join(s))
            print('sample',' '.join(sentence))
        use_teacher=np.random.rand()>0.1
        ce,kld,coef=model.REC_LOSS(batch,0.1,use_cuda,use_teacher)
        loss=87*ce+kld#+coef*reencode_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_list+=[ce.data]
        kld_list+=[kld.data]
        if i%100==0:

            print('100 step: ce:{}, kld:{}, kld_coef:{} '\
            .format(sum(ce_list[-100:])/100,sum(kld_list[-100:])/100,coef))
            t.save(model.state_dict(),"PRETRAIN_GEN_PATH")



np.save('ce_list',np.array(ce_list))
np.save('kld_list',np.array(kld_list))
np.save('coef_list',np.array(coef_list))