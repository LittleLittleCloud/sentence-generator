import numpy as np 
import collections
from six.moves import cPickle

class Preprocess:
    '''
        input: raw sentences
        output: vocab, max_len,[num*seq_len]
    '''
    def __init__(self,embedding_model,max_len=50):
        self.index_to_word=embedding_model.wv.index2word
        self.start_token='>'
        self.end_token='|'
        self.pad_token='_'
        self.missing_token='$'
        self.index_to_word=['>','|','_','$']+self.index_to_word
        self.vocab_size=len(self.index_to_word)
        
        self.word_to_index={w:i for i,w in enumerate(self.index_to_word)}
        self.max_len=max_len
        self.embedding_model=embedding_model
    
    def to_sequence(self,raw_sentences):
        tokens=[sentence.split() for sentence in raw_sentences ]
        # tokens=list(filter(lambda x:len(x)<=self.max_len,tokens))
        # max_seq_len=self.max_len
        # print(max_seq_len)
        # for token in tokens:
        #     token=['_']*(max_seq_len-len(token))+token
        # output=np.zeros((len(tokens),max_seq_len))
        for i,sentence in enumerate(tokens):
            for j,word in enumerate(sentence):
                tokens[i][j]=self.word_to_index.get(word,self.word_to_index[self.missing_token])
        return tokens

    def embedding(self):
        '''
        output: [vocab_size,embedding_size]
        '''
        #sorry
        output=np.zeros((self.vocab_size,300))
        for word,i in self.word_to_index.items():
            if word=='>':
                output[i]=np.zeros(300)
                continue
            if word=='|':
                output[i]=np.zeros(300)-1
                continue
            if word=='_':
                output[i]=np.zeros(300)-50
                continue
            if word=='$':
                output[i]=np.zeros(300)+100
                continue
            output[i]=self.embedding_model[word]
        return output

    def save(self,prefix):
        with open(prefix+'idx_to_word','wb') as f:
            cPickle.dump(self.idx_to_word)
        with open(prefix+'word_to_idx','wb') as f:
            cPickle.dump(self.word_to_idx)

if __name__=='__main__':
    preprocess=Preprocess("I:/meetup/python/sentence generator/train.txt")