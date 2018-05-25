from model.rvae import RVAE
from util.batch_loader import Batch
from util.preprocess import Preprocess
from util.parameter import Parameter
from gensim.models import KeyedVectors
from torch.optim import Adam
import numpy as np
import torch as t
import os
import argparse

parser=argparse.ArgumentParser(description='word2vec')
parser.add_argument('--batch-size',type=int,default=28,metavar='BS')
parser.add_argument('--ce-coef',type=float,default=150.0)
args=parser.parse_args()
batch_size=args.batch_size
ce_coef=args.ce_coef
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

params=Parameter(word_embed_size=300,encode_rnn_size=1000,latent_variable_size=1200,\
            decode_rnn_size=600,vocab_size=preprocess.vocab_size,embedding_path='embedding.npy',use_cuda=True)
model=RVAE(params)
model=model.cuda()
if os.path.isfile("PRETRAIN_GEN_PATH0"):
    model.load_state_dict(t.load("PRETRAIN_GEN_PATH0"))
optimizer=Adam(model.learnable_parameters(), 1e-5)

use_cuda=t.cuda.is_available()
ce_list=[]
kld_list=[]
coef_list=[]
re_list=[]
test_batch=batch_loader.test_next_batch(batch_size)
for _ in range(50):
    for i,batch in enumerate(batch_loader.train_next_batch(batch_size)):
        if i%101==0:
            sample=batch[0][0,:].reshape(1,-1)
            sentence=model.sample(sample,use_cuda)[0]
            sentence=[preprocess.index_to_word[i] for i in sentence]
            # s=[preprocess.index_to_word[i] for i in sample[0]]
            # print('origin',' '.join(s))
            print('sample',' '.join(sentence))
        use_teacher=np.random.rand()>-0.3
        ce,kld,coef=model.REC_LOSS(batch,0.3,use_cuda,use_teacher)
        loss=ce_coef*ce+coef*kld#+coef*reencode_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_list+=[ce.data]
        kld_list+=[kld.data]
        if i%100==0:

            print('100 step: ce:{}, kld:{}, kld_coef:{} '\
            .format(sum(ce_list[-100:])/100,sum(kld_list[-100:])/100,coef))
            t.save(model.state_dict(),"PRETRAIN_GEN_PATH0")



np.save('ce_list',np.array(ce_list))
np.save('kld_list',np.array(kld_list))
np.save('coef_list',np.array(coef_list))