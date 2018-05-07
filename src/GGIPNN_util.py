import numpy as np

def load_embedding_vectors(vocabulary: dict, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        if word in vocabulary:
            idx = vocabulary[word]
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def oneHot(geneLabels):
    datasets = dict()
    datasets['target_names'] = ['0', '1']
    target = []
    for label in geneLabels:
        target.append(datasets['target_names'].index(label))
    datasets['target'] = target
    labels = []
    for i in range(len(geneLabels)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return y

def myFitDict(dataList: list, length: int):
    '''
    generate gene dictionary, map gene name to index
    '''
    dataDict = dict()
    index = 0
    for line in dataList:
        eles = line.strip().split(" ")
        if len(eles) == length:
            for ele in eles:
                if ele in dataDict:
                    continue
                else:
                    dataDict[ele] = index
                    index += 1
    return dataDict

def myFit(dataList: list, length: int, dataDict :dict):
    '''
    map gene names to index by using the gene dictionary
    '''
    x = np.ones((len(dataList), length), dtype=int)
    i=0

    for line in dataList:
        eles = line.strip().split(" ")
        if len(eles) == length:
            j = 0
            for ele in eles:
                x[i,j] = dataDict[ele]
                j = 1
        i += 1
    return  x