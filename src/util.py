import pickle
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm
import copy
import random

import statistics as stats
from scipy.special import entr


'''
Load a dataset
path : path of a dataset
maxlen : maximum length of a session
'''
def data_split_userseq(path, maxlen=200):
    with open(path, 'rb') as f:
        user_num, item_num, UI, UT, interval, cnt = pickle.load(f)
    
    train_UI = copy.deepcopy(UI)
    train_UT = copy.deepcopy(UT)

    for u in train_UI:
        train_UI[u] = train_UI[u][:-2]
        train_UT[u] = train_UT[u][:-2]
    
    test_u = [u for u in UT]
    test_t = [UT[u][-1] for u in test_u]

    test_index = np.argsort(test_t)
    
    testset = []

    for i in test_index:
        u  = test_u[i]
        sess = np.zeros(maxlen)
        for i in range(1, min(maxlen, len(UI[u])-1) + 1):
            sess[-i] = UI[u][-(i+1)]
        exclude = UI[u][:-1]
        testset.append((sess, UI[u][-1], exclude))

    vld_u = [u for u in UT]
    vld_t = [UT[u][-2] for u in test_u]

    vld_index = np.argsort(vld_t)
    
    vldset = []

    for i in vld_index:
        u  = vld_u[i]
        sess = np.zeros(maxlen)
        for i in range(1, min(maxlen, len(UI[u])-2) + 1):
            sess[-i] = UI[u][-(i+2)]
        exclude = UI[u][:-2]
        vldset.append((sess, UI[u][-2], exclude))
    
    item_cnt = np.zeros(item_num+1)

    for u in UI:
        for i in UI[u][:-1]:
            item_cnt[i] += 1

    item_prob = item_cnt / item_cnt.sum()
    
    return train_UI, train_UT, user_num, item_num, testset, vldset, item_prob, interval, cnt

'''
Generate a training instance
UI : list of interacted items for each user
UT : list of interacted time for each user
user_num : number of users
item_num : number of items
item_prob : a poopularity distribution of all items
interval : size of the time window to count item frequencies
cnt : cumalative frequencies of all items
alpha : debiasing hyperparameter
maxlen : maximum length of a session
batch_size : size of a mini-batch
result_queue : 
'''
def sample_function(UI, UT, user_num, item_num, item_prob, interval, cnt, alpha, maxlen, batch_size, result_queue):
    uni = np.ones(len(item_prob))
    uni /= uni.sum()
    pool = list(range(item_num + 1))

    def sample():
        u = np.random.randint(1, user_num+1)
        while len(UI[u]) <= 1: u = np.random.randint(1, user_num + 1)

        x = np.zeros([maxlen], dtype=np.int32)
        y = np.zeros([maxlen], dtype=np.int32)        
        neg = np.zeros([maxlen], dtype=np.int32)

        idx = maxlen - 1
        nxt = UI[u][-1]
        until = UT[u][-1]

        pos = set(UI[u])
        for i, t in zip(reversed(UI[u][:-1]), reversed(UT[u][:-1])):
            x[idx] = i
            y[idx] = nxt
            if nxt!=0:
                pop = cnt[until//interval + 1] - cnt[UT[u][0]//interval]
                prob = pop/pop[0]
                prob[0] = 0

                dt = np.random.randint(100)
                if dt < alpha*100:
                    n = np.random.choice(pool, p=prob)
                else:
                    n = np.random.randint(1, item_num+1)

                while n in pos:
                    dt = np.random.randint(100)
                    if dt < alpha*100:
                        n = np.random.choice(pool, p=prob)
                    else:
                        n = np.random.randint(1, item_num+1)
                
                neg[idx] = n
            nxt=i
            until=t
            idx-=1
            if idx == -1:
                break
        
        return (u, x, y, neg)
    
    np.random.seed(42)
    random.seed(42)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        
        result_queue.put(zip(*one_batch))

# mini-batch sampler
class WarpSampler(object):
    def __init__(self, UI, UT, user_num, item_num, item_prob, interval, cnt, alpha = 1.0, maxlen=200, batch_size=128, n_workers=1):
        self.result_queue = Queue(maxsize = n_workers * 20)
        self.processors = []
        print('initializing workers...')
        for i in tqdm(range(n_workers)):
            self.processors.append(
                Process(
                    target = sample_function, 
                    args=(UI, UT, user_num, item_num, item_prob, interval, cnt, alpha, maxlen, batch_size, self.result_queue)
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

'''
Calculate HR, nDCG, Entropy, and Gini index of the given recommendation lists
reclist : recommendation lists to be evaluated
answer : ground truth item for each user
user_num : number of all users
item_num : number of all items
k : size of a recommendation list
'''
def metric(reclist, answer, user_num, item_num, k=None):
    if k is None:
        k = len(reclist[0])
    
    hit = 0
    ndcg = 0
    for i in range(len(reclist)):
        if answer[i] in reclist[i][:k]:
            hit += 1
            rank = reclist[i][:k].index(answer[i])
            ndcg += 1 / np.log2(rank + 2)
    hit /= len(reclist)
    ndcg /= len(reclist)
    
    itemcnt = np.zeros(item_num + 1)
    for r in reclist:
        for i in r[:k]:
            itemcnt[i] += 1

    ent = entr(itemcnt/sum(itemcnt)).sum()

    cnt = itemcnt[1:]
    cnt.sort()
    height, area = 0, 0
    for c in cnt:
        height += c
        area += height-c/2
    fair_area = height*item_num/2
    giny = (fair_area-area)/fair_area

    return hit, ndcg, ent, giny

'''
Calculate HR, nDCG, Entropy, and Gini index of each divided chunk of the given recommendation lists
reclist : recommendation lists to be evaluated
answer : ground truth item for each user
user_num : number of all users
item_num : number of all items
div : number of chunks to split the recommendation lists
k : size of a recommendation list
'''
def kfold_metric(reclist, answer, user_num, item_num, div=5, k=10):
    n = len(reclist)
    hits = []
    ndcgs = []
    ents = []
    gins = []
    for i in range(div):
        hit, ndcg, ent, gin = metric(reclist[i*n//div:(i+1)*n//div], answer[i*n//div:(i+1)*n//div],user_num, item_num, k=k)
        hits.append(hit)
        ndcgs.append(ndcg)
        ents.append(ent)
        gins.append(gin)
    return hits, ndcgs, ents, gins

'''
Calculate div-fold harmonic HR, div-fold harmonic nDCG, Entropy, and Gini index of the given recommendation lists
reclist : recommendation lists to be evaluated
answer : ground truth item for each user
user_num : number of all users
item_num : number of all items
div : number of chunks to split the recommendation lists
k : size of a recommendation list
'''
def kfold_report_tuple(reclist, answer, user_num, item_num, div=5, k=10):
    hits, ndcgs, ents, gins = kfold_metric(reclist, answer, user_num, item_num, div, k=k)
    hr, ndcg, ent, gin = metric(reclist, answer, user_num, item_num, k=k)
    h = stats.harmonic_mean(hits)
    n = stats.harmonic_mean(ndcgs)
    return h, n, ent, gin
