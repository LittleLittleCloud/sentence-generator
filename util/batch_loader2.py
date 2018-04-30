import numpy as np 
import operator

class Batch:
    def __init__(self, data, train_test_split,index2word,max_len=1000):
        '''
        data: [num,2]
        '''
        assert train_test_split<1 and train_test_split>0

        self.unknown_token='<ukn>'
        self.num_token='#'
        self.pad_token='_'
        self.new_line='\r\n'
        self.index_to_word=index2word
        self.index_to_word=['>','\r\n','_','<ukn>','#']+self.index_to_word
        self.vocab_size=len(self.index_to_word)
        self.word_to_index={w:i for i,w in enumerate(self.index_to_word)}
        self.data,self.label=self.preprocess(data,max_len)
        self.index=np.arange(len(self.data))
        train_len=int(len(self.data)*train_test_split)
        self.train_index=self.index[:train_len]
        self.test_index=self.index[train_len:]
        self.train_data=[self.data[i] for i in self.train_index]
        self.train_label=[self.label[i] for i in self.train_index]
        self.train_test_split=train_test_split
        print('data size ',len(self.data))

    
    def preprocess(self,raw_data,max_len):
        '''
        raw_data: [num,2] [str;float]
        '''
        sentences=raw_data[:,0]
        sentences_len=np.array([len(x.split()) for x in sentences])
        mask=sentences_len<max_len
        raw_data=raw_data[mask]
        sentences=raw_data[:,0]
        label=raw_data[:,1]
        
        # replace nums to #
        import re
        regex=re.compile(r'\d+')
        sentences=[regex.sub(self.num_token,sentence) for sentence in sentences]
        tokens=[sentence.split() for sentence in sentences]

        for i,sentence in enumerate(tokens):

            for j,word in enumerate(sentence):
                tokens[i][j]=self.word_to_index.get(word,self.word_to_index[self.unknown_token])
        return tokens,label

    def train_next_batch(self,batch_size,shuffle=True):

        assert batch_size<=len(self.train_index)
        data=self.data
        label=self.label
        if shuffle:
            np.random.shuffle(self.train_index)
        cur=0
        while cur<len(self.train_index):
            start=cur
            end=min(cur+batch_size,len(self.train_index))
            cur=end
            X=[data[i] for i in self.train_index[start:end]]
            y=[label[i] for i in self.train_index[start:end]]
            max_len=max(len(x) for x in X)

            #sorry
            for i,line in enumerate(X):
                to_add=max_len-len(line)
                X[i]=line+[self.word_to_index[self.pad_token]]*to_add
            yield np.array(X),np.array(y)

    
    def test_next_batch(self,batch_size,raw=False):
        '''

            if raw==False, align the output length return np.array
                
        '''
        assert batch_size<len(self.test_index)
        data=self.data
        label=self.label
        np.random.shuffle(self.test_index)
        cur=0
        while cur<len(self.train_index):
            start=cur
            end=min(cur+batch_size,len(self.test_index))
            cur=end
            X=[data[i] for i in self.test_index[start:end]]
            y=[label[i] for i in self.test_index[start:end]]
            max_len=max(len(x) for x in X)
            if not raw:
            #sorry
                for i,line in enumerate(X):
                    to_add=max_len-len(line)
                    X[i]=line+[self.word_to_index[self.pad_token]]*to_add
                yield np.array(X),np.array(y)
            else:
                yield X,np.array(y)

    def gen_positive_sample(self,n):
        sample=np.random.choice(len(self.train_index),n)
        index=self.train_index[sample]
        X=[self.data[i] for i in index]
        y=[self.label[i] for i in index]
        return X,y



if __name__=='__main__':
    batch=Batch()