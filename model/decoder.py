import torch
import torch.nn as nn
import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(self,params):
        super(Decoder,self).__init__()
        self.params=params

        self.lstm=nn.LSTM(  input_size=params.latent_variable_size+params.word_embed_size,
                            hidden_size=params.decode_rnn_size,
                            num_layers=params.decode_num_layer,
                            batch_first=True
        )
        # self.gru=nn.GRU( 
        #                     input_size=params.latent_variable_size+params.word_embed_size,
        #                     hidden_size=params.decode_rnn_size,
        #                     num_layers=params.decode_num_layer,
        #                     batch_first=True
        # )

        self.fc=nn.Linear(params.decode_rnn_size,params.vocab_size)

    

    def forward(self,input,z,drop_prob=0.0,init_state=None,concat=True):

        [batch_size,seq_len,embeding_size]=input.size()
        decoder_input = F.dropout(input, drop_prob)
        if concat:
            # cat z to the end of each word
            z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
            decoder_input = torch.cat([decoder_input, z], 2)
        if init_state is not None:
            init_state=(init_state[0],init_state[1])
        out,hidden=self.lstm(decoder_input)
        out=out.contiguous().view(-1,self.params.decode_rnn_size)
        out=self.fc(out)
        out=F.log_softmax(out,1)
        return out,hidden
        # input=decoder_input[0].view(batch_size,1,embeding_size)
        # for i in range(seq_len):
        #     out,h=self.gru(input,h)
        #     if teacher:
        #         input=decoder_input[i].view(batch_size,1,embeding_size)

        # rnn_out,final_state=self.lstm(decoder_input,init_state)
        # rnn_out = rnn_out.contiguous().view(-1, self.params.decode_rnn_size)
        # result = self.fc(rnn_out)
        # result = result.view(batch_size, seq_len, self.params.vocab_size)
        
        # return result, final_state
        
