import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from ge.classify import read_node_label, Classifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import math

from ge import DeepWalk
from ge import Node2Vec
def get_network(filename="../ml-1m/ratings.csv"):
    rating_df = pd.read_csv(filename,encoding='ISO-8859-1',error_bad_lines=False,engine='python')
    data = list(zip(rating_df['movieID'],rating_df['rating']))
    train_user, vali_user, train_data, vali_data = train_test_split(rating_df['userID'],data,test_size=0.2)
    train_user_ = ["u%i"%i for i in train_user]
    vali_user_ = ["u%i"%i for i in vali_user]
    train_movie = [i[0] for i in train_data]
    vali_movie = [i[0] for i in vali_data]
    train_movie_ = ['m%i'%i[0] for i in train_data]
    vali_movie_ = ["m%i"%i[0] for i in vali_data]
    train_rating = [i[1] for i in train_data]
    vali_rating = [i[1] for i in vali_data]
    train_rating_ = [i[1] for i in train_data]
    vali_rating_ = [i[1] for i in vali_data]
    train_df = pd.DataFrame(data={'userID':train_user_,'movieID':train_movie_,'rating':train_rating_})
    vali_df = pd.DataFrame(data={'userID':vali_user_,'movieID':vali_movie_,'rating':vali_rating_})
    order = ['userID','movieID','rating']
    train_df = train_df[order]
    vali_df = vali_df[order]
    train_mat = csr_matrix((train_rating,(train_user, train_movie)))
    vali_mat = csr_matrix((vali_rating, (vali_user, vali_movie)))
    return train_df, vali_df, train_mat, vali_mat

def appendMovie():
    f = open("networkentities.txt",encoding='utf-16')
    network  =open("network.txt",mode='a+',encoding='utf-16')
    lines = f.readlines()
    for line in tqdm(lines):
        alist = line.strip().split('@')
        if len(alist)>=2:
            network.write(alist[0] + '@' + alist[1] + '\n')


def get_similarity(embedding, a, b):
    if a in embedding and b in embedding:
        ar = embedding[a]
        br = embedding[b]
        res = (ar*br).sum()
        ari = np.square(ar).sum()
        bri = np.square(br).sum()
        return (res)/(math.sqrt(ari*bri))
    return 0.0
def get_dissimilarity(embedding, a, b):
    if a in embedding and b in embedding:
        ar = embedding[a]
        br = embedding[b]
        dr = br-ar
        dis = math.sqrt(np.square(dr).sum())
        return 1.0/(1+dis)
    return 0.0
def topN(embdding,train_mat, user,itemcount,n=10):
    rating_list = train_mat[user].nonzero()[1]
    simi_list = []
    for i in range(1,itemcount+1):
        if i in rating_list:
            continue
        simi = get_similarity(embdding, 'u'+str(user), 'm'+str(i))
        simi_list.append([simi, i])
    simi_list.sort(reverse=True)
    return simi_list[:n]
def evaluate(embeddings, train_data, vali_data, usercount, itemcount,n=10):
    count = 0.0
    precision_list = []
    recall_list = []
    for user in tqdm(range(1,usercount+1)):
        rating_list = vali_data[user].nonzero()[1]
        similist = np.array(topN(embeddings, train_data, user, itemcount, n))
        similist = np.array(similist[:, 1],dtype=int).tolist()
        inter = float(len(set(similist)&set(rating_list)))
        count += inter
        if len(similist)==0:
            precision_list.append(0.0)
        else:
            precision_list.append(inter/len(similist))
        if len(rating_list)==0:
            recall_list.append(1.0)
        else:
            recall_list.append(inter/len(rating_list))
    precision = count/(usercount*n)
    T = vali_data.data.shape[0]
    recall = count/T
    return precision, recall,precision_list,recall_list
def buildGraph():
    f = open('network.txt',encoding='utf-16')
    lines = f.readlines()
    G = nx.Graph()
    for line in tqdm(lines):
        alist = line.strip().split("@")
        if len(alist)==3:
            G.add_weighted_edges_from([(alist[0],alist[1],float(alist[2]))])
        elif len(alist)==2:
            G.add_weighted_edges_from([(alist[0],alist[1],5.0)])
    return G


if __name__ == '__main__':
    train_df, vali_df, train_mat, vali_mat = get_network("ratings.csv")
    train_df.to_csv("network.txt",sep='@',mode='w',index=None,header=None,encoding='utf-16')
    appendMovie()
    G = buildGraph()
    #G = nx.read_edgelist("network.txt", encoding='utf-16',delimiter='@',
                        # create_using=nx.Graph(), nodetype=None, data=[('weight', float)])
    print("G done!")
    model = Node2Vec(G, p=0.25,q=4,walk_length=10, num_walks=80, workers=1)
    print("begin to train")
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    print("embeddings got, begin to evaluate")
    for N in range(5,21,5):
        precision, recall, precision_list, recall_list = evaluate(embeddings, train_mat, vali_mat, 6040, 3952)
        print("top{}:".format(N))
        print("macroPrecision:{}%".format(precision * 100))
        print("macroRecall:{}%".format(recall * 100))
        print("f1 score:{}".format((2 * precision * recall) / (precision + recall)))
        microPrecision = np.average(precision_list)
        microRecall = np.average(recall_list)
        microF1 = (2 * microPrecision * microRecall) / (microPrecision + microRecall)
        print("microPrecision:{}%".format(microPrecision * 100))
        print("mircroRecall:{}%".format(microRecall * 100))
        np.savetxt("../results/precision_list_node2vec_d_know_"+str(N)+".txt", precision_list)
        np.savetxt("../results/recall_list_node2vec_d_know_"+str(N)+".txt", recall_list)



