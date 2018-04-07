import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, decoder_hidden_size,encoder_hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = decoder_hidden_size
        self.attn = nn.Linear(decoder_hidden_size+encoder_hidden_size*2, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (B,layers*directions,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,T,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        H = hidden.repeat(1,max_len,1) #[B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        cat_mat=torch.cat([hidden, encoder_outputs], 2)
        [b,seq_len,s]=cat_mat.size()
        cat_mat=cat_mat.view(-1,s)
        energy = F.tanh(self.attn(cat_mat))  # [B*T*2H]->[B*T*H]
        energy=energy.view(b,seq_len,self.hidden_size)
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.size()[0],1)
        v=v.unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]


class AttnDecoder(nn.Module):
    def __init__(self,params):
        super(AttnDecoder, self).__init__()
        # Define parameters
        self.params=params

        # self.hidden_size = hidden_size
        # self.embed_size = embed_size
        # self.output_size = output_size
        # self.n_layers = n_layers
        # self.dropout_p = dropout_p
        # Define layers
        # self.embedding = nn.Embedding(output_size, embed_size)
        # self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', params.decode_rnn_size,params.encode_rnn_size)
        self.gru = nn.GRU(params.latent_variable_size+params.word_embed_size, params.decode_rnn_size, params.decode_num_layer)
        self.attn_combine = nn.Linear(params.latent_variable_size+params.word_embed_size+params.encode_rnn_size*2, \
                            params.latent_variable_size+params.word_embed_size )
        self.out = nn.Linear(params.decode_rnn_size, params.vocab_size)

    def forward(self, input,z,drop_prob=0.0, init_state=None,concat=True,c=None):
        [batch_size,seq_len,embeding_size]=input.size()
        decoder_input = F.dropout(input, drop_prob)
        if concat:
            # cat z to the end of each word
            z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
            decoder_input = torch.cat([decoder_input, z], 2)
        outputs=[]
        hidden=init_state

        for i in range(seq_len):
            word_input=decoder_input[:,i,:].view(batch_size,1,-1)
            output,hidden=self.one_step(word_input,init_state,c)
            outputs+=[output.unsqueeze(1)]
        
        result=torch.cat(outputs,1).contiguous().view(batch_size,seq_len,self.params.vocab_size)
        return result,hidden



    def one_step(self,word_input, last_hidden, encoder_outputs):

        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)  # (B,1,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_input, context), 2)
        [b,_,s]=rnn_input.size()
        rnn_input = self.attn_combine(rnn_input.view(b,s)) # use it in case your size of rnn_input is different
        rnn_input=rnn_input.unsqueeze(1) # [B*1*h]
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)  # (B,1,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden

