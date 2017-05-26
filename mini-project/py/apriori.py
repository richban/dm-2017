
# coding: utf-8

# In[1]:

import numpy as np
import csv
PATH = 'data/apriori.csv'


# In[2]:

def loadDataSet(filename):
    """load data and split into training & test sets"""
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset


# In[3]:

def createC1(dataset):
    """Create the first l-itemset C1 of size one."""
    c1 = []
    for t in dataset:
        for i in t:
            if not [i] in c1:
                c1.append([i])
    return map(frozenset, c1)


# In[4]:

def scanD(dataset, candidates, min_sup):
    """Return candidates l-itemsets satisfying minimum support."""
    csc = {}
    for t in dataset:
        for c in candidates:
            if c.issubset(t):
                csc.setdefault(c, 0)
                csc[c] += 1
    
    items_len = float(len(dataset))
    freq = []
    support_data = {}
    for key in csc:
        can_support = csc[key] / items_len
        if can_support >= min_sup:
            freq.insert(0, key)
        support_data[key] = can_support
    return freq, support_data


# In[5]:

def joinSet(itemSet, length):
    """Join a set with itself."""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


# In[6]:

def apriori(dataset, min_sup=0.2):
    """Generation of frequent itemsets"""
    C1 = list(createC1(dataset))
    D = list(map(set, dataset))
    L1, support_data = scanD(D, C1, min_sup)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        candidates = joinSet(L[k-2], k)
        Lk, suppK = scanD(D, candidates, min_sup)
        support_data.update(suppK)
        L.append(Lk)
        k += 1
    
    return L, support_data


# In[7]:

def confidence(itemset, H, support_data, rules, min_conf=0.7):
    """Evaluate the rule generated"""
    prunedSet = []
    for h in H:
        conf = support_data[itemset] / support_data[itemset - h]
        if conf >= min_conf:
            print(str(itemset-h) + "---->" + str(h) + " confidence: " + str(conf))
            rules.append((itemset-h, h, conf))
            prunedSet.append(h)
    return prunedSet


# In[8]:

def rules_(itemset, H, support_data, rules, min_conf=0.7):
    """Generate a set of candidate rules."""
    m = len(H[0])
    if(len(itemset) > (m+1)):
        hmp = joinSet(H, m+1)
        hmp = confidence(itemset, hmp, support_data, rules, min_conf)
        if len(hmp) > 1:
            rules(itemset, hmp, support_data, rules, min_conf)


# In[24]:

def generateRules(L, support_data, min_conf=0.7):
    """Generate the association rules."""
    rules = []
    for i in range(1, len(L)):
        for itemset in L[i]:
            H1 = [frozenset([item]) for item in itemset]
            print("FrequentSet " + str(itemset) + "H1 " + str(H1) + str(i))
            if(i > 1):
                rules_(itemset, H1, support_data, rules, min_conf)
            else:
                confidence(itemset, H1, support_data, rules, min_conf)
    return rules


# In[25]:

dataset = loadDataSet(PATH)


# In[26]:

L, support = apriori(dataset)


# In[30]:

L


# In[27]:

generateRules(L, support)


# In[ ]:



