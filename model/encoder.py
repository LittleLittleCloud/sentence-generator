import torch
import torch.nn as nn
from .highway import Highway
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,params):
        super(Encoder,self).__init__()
        self.highway=Highway(params.word_embed_size,1,F.relu)
        self.lstm=nn.LSTM(  
                            input_size=params.word_embed_size,
                            hidden_size=params.encode_rnn_size,
                            bidirectional = True,
                            batch_first=True
                        )
        self.params=params
    

    def forward(self,input):

        [batch_size,seq_len,embed_size]=input.size()

        input=input.view(-1,embed_size)
        input=self.highway(input)
        input=input.view(batch_size,seq_len,embed_size)
        _,(final_hidden_state,final_cell_state)=self.lstm(input)
        h1,h2=final_hidden_state[0],final_hidden_state[1]
        c1,c2=final_cell_state[0],final_cell_state[1]
        final_hidden_state=torch.cat([h1,h2],dim=1)
        final_cell_state=torch.cat([c1,c2],dim=1)
        return final_hidden_state,final_cell_state