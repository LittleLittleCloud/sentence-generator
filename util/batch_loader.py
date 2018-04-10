import numpy as np 


class Batch:
    def __init__(self, data, train_test_split):

        assert train_test_split<1 and train_test_split>0
        self.data=data
        self.train_test_split=train_test_split
        self.index=np.arange(len(data))
        train_len=int(len(data)*train_test_split)

        self.train_index=self.index[:train_len]
        self.test_index=self.index[train_len:]
    
    def train_next_batch(self,batch_size,shuffle=True,index=False,data=None):
        assert batch_size<=len(self.train_index)
        if data:
            data=data
            train_index=np.arange(len(data))
        else:
            data=self.data
            train_index=self.train_index
        if shuffle:
            np.random.shuffle(self.train_index)
        cur=0
        while cur<len(train_index):
            start=cur
            end=min(cur+batch_size,len(train_index))
            cur=end
            encode_input=[data[i] for i in train_index[start:end]]
            decode_input=[[0]+data[i] for i in train_index[start:end]]
            target=[data[i]+[1] for i in train_index[start:end]]
            max_len=max(len(x) for x in encode_input)

            #sorry again
            for i,line in enumerate(encode_input):
                to_add=max_len-len(line)
                encode_input[i]=line+[2]*to_add
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
            if index:
                yield encode_input,decode_input,target,(start,end)
            else:
                yield encode_input,decode_input,target

    
    def test_next_batch(self,batch_size):
        assert batch_size<len(self.test_index)
        data=self.data
        np.random.shuffle(self.test_index)
        cur=0
        while cur<len(self.test_index):
            start=cur
            end=min(cur+batch_size,len(self.test_index))
            cur=end
            encode_input=[data[i] for i in self.test_index[start:end]]
            decode_input=[[0] for i in self.test_index[start:end]]
            max_len=max(len(x) for x in encode_input)

            #sorry again
            for i,line in enumerate(encode_input):
                to_add=max_len-len(line)
                encode_input[i]=line+[2]*to_add
            encode_input=np.array(encode_input)
            decode_input=np.array(decode_input)

            assert decode_input.shape==(encode_input.shape[0],1)
            yield encode_input,decode_input

    def to_input(self,data):
        '''
            data: [batch] should be a list of sequence
        '''
        batch_size=len(data)
        max_len=max(len(x) for x in data)
        encode_input=[data[i] for i in range(batch_size)]
        decode_input=[[0]+data[i] for i in range(batch_size)]
        target=[data[i]+[1] for i in range(batch_size)]

        #sorry again
        for i,line in enumerate(encode_input):
            to_add=max_len-len(line)
            encode_input[i]=line+[2]*to_add
        encode_input=np.array(encode_input)

        for i,line in enumerate(decode_input):
            to_add=max_len+1-len(line)
            decode_input[i]=line+[2]*to_add
        decode_input=np.array(decode_input)

        for i,line in enumerate(target):
            to_add=max_len+1-len(line)
            target[i]=line+[2]*to_add
        target=np.array(target)

        return encode_input,decode_input,target


if __name__=='__main__':
    batch=Batch()