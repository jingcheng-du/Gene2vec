import gensim
import itertools
import random

msigdb_file = "msigdb.v6.1.symbols.gmt" #update with your msigdb file address
pathwayList = []
with open(msigdb_file, 'r') as readFile:
    for line in readFile:
        tmpList = line.split("\t")
        n = len(tmpList)
        if n > 52:
            continue
        pathwayList.append(line)
readFile.close()

def targetFunc (emb_w2v_file):
    geneEmbedList = []
    with open(emb_w2v_file, 'r') as file:
        for line in file:
            if len(line.split(" ")) == 2: # first line
                continue
            geneEmbedList.append((line.split(" ")[0]))
    file.close()

    model = gensim.models.KeyedVectors.load_word2vec_format(emb_w2v_file)
    paths_array = [] # Numerator in target function

    for pathway in pathwayList:
        geneList = list()
        path_arr = []
        tmpList = pathway.split("\t")
        n = len(tmpList)
        for i in range(2, n):
            if tmpList[i] in geneEmbedList:
                geneList.append(tmpList[i])
        genePairs = list(itertools.combinations(geneList, 2))
        for pair in genePairs:
            sim = model.wv.similarity(pair[0], pair[1])
            path_arr.append(sim)
        paths_array.append(sum(path_arr)/len(path_arr))
        tmpList.clear()
        geneList.clear()

    randArray = [] # Denominator in target function
    random.seed(35)
    random.shuffle(geneEmbedList)
    genePairs = list(itertools.combinations(geneEmbedList[:1000], 2))
    for pair in genePairs:
        sim = model.wv.similarity(pair[0], pair[1])
        randArray.append(sim)
    genePairs.clear()
    print("------------")
    print(emb_w2v_file)
    path_mean = sum(paths_array) / len(paths_array)
    rand_mean = sum(randArray) / len(randArray)
    print(path_mean,end="")
    print("\t",rand_mean)
    print(path_mean/rand_mean)
    print("------------")
    return path_mean/ rand_mean

emb_w2v_file = "../pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
targetFunc(emb_w2v_file)