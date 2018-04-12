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
with open('train','r',encoding='utf-8') as f:
    data=f.readlines()


preprocess=Preprocess(embedding_model)
input=preprocess.to_sequence(data)
# embedding=preprocess.embedding()
# np.save('embedding',embedding)

batch_loader=Batch(input,0.7)

params=Parameter(word_embed_size=300,encode_rnn_size=100,latent_variable_size=100,\
            decode_rnn_size=100,vocab_size=preprocess.vocab_size,embedding_path='embedding.npy',use_cuda=True)
model=RVAE(params)
model=model.cuda()
optimizer=Adam(model.learnable_parameters(), 1e-4)

use_cuda=t.cuda.is_available()
ce_list=[]
kld_list=[]
coef_list=[]
test_batch=batch_loader.test_next_batch(1)

for i,batch in enumerate(batch_loader.train_next_batch(5)):
    if i%101==0:
        sample=next(test_batch)
        sentence=model.sample(sample[0],use_cuda).cpu().numpy()[0]
        sentence=[preprocess.index_to_word[i] for i in sentence]
        s=[preprocess.index_to_word[i] for i in sample[0][0]]
        print('origin',' '.join(s))
        print('sample',' '.join(sentence))
        continue
    ce,kld,coef=model.REC_LOSS(batch,0.1,use_cuda)
    loss=77*ce+coef*kld
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ce_list+=[ce.data]
    kld_list+=[kld.data]
    if i%100==0:
        print('100 step: ce:{}, kld:{} '.format(sum(ce_list[-100:])/100,sum(kld_list[-100:])/100))



np.save('ce_list',np.array(ce_list))
np.save('kld_list',np.array(kld_list))
np.save('coef_list',np.array(coef_list))