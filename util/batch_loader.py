import numpy as np 


class Batch:
    def __init__(self, data, train_test_split):

        assert train_test_split<1 and train_test_split>0
        self.data=data
        self.train_test_split=train_test_split
        self.index=np.arange(len(data))
        np.random.shuffle(self.index)
        train_len=int(len(data)*train_test_split)

        self.train_index=self.index[:train_len]
        self.test_index=self.index[train_len:]
    
    def train_next_batch(self,batch_size):
        assert batch_size<len(self.train_index)
        data=self.data
        np.random.shuffle(self.train_index)
        cur=0
        while cur<len(self.train_index):
            start=cur
            end=min(cur+batch_size,len(self.train_index))
            cur=end
            encode_input=[data[i] for i in self.train_index[start:end]]
            decode_input=[[0]+data[i] for i in self.train_index[start:end]]
            target=[data[i]+[1] for i in self.train_index[start:end]]
            max_len=max(len(x) for x in encode_input)

            #sorry again
            for i,line in enumerate(encode_input):
                to_add=max_len-len(line)
                encode_input[i]=[2]*to_add+line[::-1]
            encode_input=np.array(encode_input)

            for i,line in enumerate(decode_input):
                to_add=max_len+1-len(line)
                decode_input[i]=line+[2]*to_add
            decode_input=np.array(decode_input)

            for i,line in enumerate(target):
                to_add=max_len+1-len(line)
                target[i]=line+[2]*to_add
            target=np.array(target)

            assert decode_input.shape==target.shape

            assert decode_input.shape==(encode_input.shape[0],encode_input.shape[1]+1)
            yield encode_input,decode_input,target

    
    def test_next_batch(self,batch_size):
        assert batch_size<len(self.test_index)

        np.random.shuffle(self.test_index)
        cur=0
        while cur<len(self.test_index):
            start=cur
            end=cur+batch_size
            cur=end
            yield self.data[self.test_index[start:end]]




if __name__=='__main__':
    batch=Batch()