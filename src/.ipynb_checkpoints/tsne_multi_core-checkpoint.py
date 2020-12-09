import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
import random
from multiprocessing import Pool

def load_embedding(filename):
    geneList = list()
    vectorList = list()
    f = open(filename)
    for line in f:
        values = line.split()
        gene = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        geneList.append(gene)
        vectorList.append(vector)
    f.close()
    return np.asarray(vectorList), np.asarray(geneList)

embeddings_file = "../pre_trained_emb/gene2vec_dim_200_iter_9.txt"
wv, vocabulary = load_embedding(embeddings_file)

indexes = list(range(len(wv)))
random.shuffle(indexes)

topN = len(wv)

rdWV = wv[indexes][:topN,:]
rdVB = vocabulary[indexes][:topN]
print("PCA!")
pca = PCA(n_components=50)
pca.fit(rdWV)
pca_rdWV=pca.transform(rdWV)
print("PCA done!")
print("tsne!")

with open("../TSNE_label_gene2vec.txt", 'w') as out:
    for str in rdVB:
        out.write(str + "\n")
out.close()

def TSNEWoker (iter):
    print(iter+" begin")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=int(iter), learning_rate=200, n_jobs=32)
    np.set_printoptions(suppress=True)
    # save tsne data
    Y = tsne.fit_transform(pca_rdWV)
    np.savetxt("../TSNE_data_gene2vec.txt_"+iter+".txt",Y)
    print(iter+" tsne done!")
def mp_handler():
    p = Pool(6)
    p.map(TSNEWoker, ["100","5000","10000","20000","50000","100000"]) #generate tsne of different iteration in parallel

if __name__ == '__main__':
    # TSNEWoker("5000")
    mp_handler()