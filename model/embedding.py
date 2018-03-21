import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Embedding(nn.Module):
    def __init__(self,params):
        super(Embedding,self).__init__()

        self.params=params
        word_embed=np.load(self.params.embedding_path)

        assert word_embed.shape==(self.params.vocab_size,self.params.word_embed_size)
        self.word_embed=nn.Embedding(self.params.vocab_size,self.params.word_embed_size)
        self.word_embed.weight=nn.Parameter(torch.from_numpy(word_embed).float(),requires_grad=False)


    def forward(self, input):
        '''
            input: [batch_size,seq_len]
            output: [batch_size,seq_len,embed_size]
        '''

        output=self.word_embed(input)

        return output