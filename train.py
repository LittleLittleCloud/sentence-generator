from model.rvae import RVAE
from util.batch_loader import Batch
from util.preprocess import Preprocess
from util.parameter import Parameter
from gensim.models import KeyedVectors
from torch.optim import Adam
import numpy as np
import torch as t

embedding_model=KeyedVectors.load_word2vec_format('embedding.bin')

#load data
data=0
with open('train.txt','r') as f:
    data=f.readlines()


preprocess=Preprocess(embedding_model)
input=preprocess.to_sequence(data)
# embedding=preprocess.embedding()
# np.save('embedding',embedding)

batch_loader=Batch(input,0.7)

params=Parameter(word_embed_size=300,encode_rnn_size=600,latent_variable_size=1400,\
            decode_rnn_size=600,vocab_size=preprocess.vocab_size,embedding_path='embedding.npy')
model=RVAE(params)
model=model.cuda()
optimizer=Adam(model.learnable_parameters(), 1e-3)
train_step=model.trainer(optimizer)

use_cuda=t.cuda.is_available()
ce_list=[]
kld_list=[]
coef_list=[]
for i,batch in enumerate(batch_loader.train_next_batch(2)):
    ce,kld,coef=train_step(i,batch,300,0.2,use_cuda)
    if i%10==0:
        print('ten step: ce:{}, kld:{} '.format(ce,kld))
    if i%20==0:
        seed = np.random.normal(size=[1, params.latent_variable_size])
        sentence=model.sample(50,seed,use_cuda)
        sentence=[preprocess.index_to_word(i) for i in sentence]
        print(' '.join(sentence))


np.save('ce_list',np.array(ce_list))
np.save('kld_list',np.array(kld_list))
np.save('coef_list',np.array(coef_list))