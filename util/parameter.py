class Parameter:
    def __init__(self,word_embed_size,encode_rnn_size,\
        latent_variable_size,decode_rnn_size,vocab_size,embedding_path,use_cuda):
        self.word_embed_size=word_embed_size
        self.encode_rnn_size=encode_rnn_size
        self.latent_variable_size=latent_variable_size
        self.decode_rnn_size=decode_rnn_size
        self.vocab_size=vocab_size
        self.decode_num_layer=1
        self.encode_num_layer=1
        self.embedding_path=embedding_path
        self.use_cuda=use_cuda