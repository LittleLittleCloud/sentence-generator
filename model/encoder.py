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
        _,(final_state,_)=self.lstm(input)
        final_state = final_state.view(self.params.encode_num_layer, 2, batch_size, self.params.encode_rnn_size)
        final_state = final_state[-1]
        h_1,h_2=final_state[0],final_state[1]
        final_state=torch.cat([h_1,h_2],1)
        return final_state