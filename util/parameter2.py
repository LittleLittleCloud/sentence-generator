class Parameter:
    def __init__(self,vocab_size,embedding_path,embedding_size):
        self.vocab_size=vocab_size
        self.embedding_path=embedding_path
        self.embedding_size=embedding_size
        self.kernels=[(1,3),(2,5),(3,15),(4,30),(5,120)] #(kernel size, out channel)
        self.fc_input_size=sum([oc for (_,oc) in self.kernels])
        # sorry for the compatibality
        self.word_embed_size=embedding_size

        