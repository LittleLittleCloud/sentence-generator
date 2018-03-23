import numpy as np 
import collections
from six.moves import cPickle
import re
class Preprocess:
    '''
        input: raw sentences
        output: vocab, max_len,[num*seq_len]
    '''
    def __init__(self,embedding_model=None,max_len=50):
        self.index_to_word=embedding_model.wv.index2word
        self.start_token='>'
        self.end_token='|'
        self.pad_token='_'
        self.unknown_token='<ukn>'
        self.digit_token='#'
        self.index_to_word=['>','|','_','<ukn>','#']+self.index_to_word
        self.vocab_size=len(self.index_to_word)
        
        self.word_to_index={w:i for i,w in enumerate(self.index_to_word)}
        self.max_len=max_len
        self.embedding_model=embedding_model
    
    def to_sequence(self,raw_sentences):
        tokens=[sentence.split() for sentence in raw_sentences ]

        for i,sentence in enumerate(tokens):
            for j,word in enumerate(sentence):
                tokens[i][j]=self.word_to_index.get(word,self.word_to_index[self.unknown_token])
        return tokens

    def wash_data(self,raw_data,split='\n',save=None):
        raw_data=raw_data.lower()
        raw_sentence=raw_data.split(split)

        raw_sentence=[sentence for sentence in raw_sentence if len(sentence.split(' '))>7]   
        
        raw_token=[item for i in raw_sentence for item in re.split(r'[ ]+',i)]
        raw_token_counts=collections.Counter(raw_token)
        ukn_set=set()
        for (token,val) in raw_token_counts.items():
            if val<5:
                ukn_set|=set(token)

        for i,sentence in enumerate(raw_sentence):
            tokens=re.split(r'[ ]+',sentence)
            for j,token in enumerate(tokens):
                if re.match(r'^\d*$',token):
                    tokens[j]='#'
                elif token in ukn_set:
                    tokens[j]=self.unknown_token
            raw_sentence[i]=' '.join(tokens)
        
        
        if save:
            with open(save,'w') as f:
                f.write('\n'.join(raw_sentence))
        return raw_sentence
                    

    def embedding(self):
        '''
        output: [vocab_size,embedding_size]
        '''
        #sorry

        start_embedding=np.random.rand(300)
        end_embedding=np.random.rand(300)
        pad_embedding=np.random.rand(300)
        output=np.zeros((self.vocab_size,300))
        for word,i in self.word_to_index.items():
            if word=='>':
                output[i]=start_embedding.copy()
                continue
            if word=='|':
                output[i]=end_embedding.copy()
                continue
            if word=='_':
                output[i]=pad_embedding.copy()
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