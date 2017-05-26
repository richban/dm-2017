
# coding: utf-8

# In[1]:

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure
get_ipython().magic('matplotlib inline')


# In[2]:

def load_data():
    data = [l.strip() for l in open('data/n_minmax.csv') if l.strip()]
    features = [list(map(float, x.split(',')[:-1])) for x in data]
    labels = [x.split(',')[-1] for x in data]
    return features, labels


# In[3]:

class Cluster(object):
    """Object that represent
    the Clusters & Centroids and Data Points
    in the n dimensional space"""
    
    def __init__(self, vectors):
        """The vectors of the cluster"""
        
        self.vectors = vectors
        self.centroid = self.calcCentroid()
        
    """def __repr__(self):
        
        return str((self.vectors, self.centroid))"""
    
    def calcCentroid(self):
        """Calculate the center point"""
        size = len(self.vectors)
        # zip all features together
        zipped = zip(*self.vectors)
        # Calculate the mean for each feature/column
        centroid = [math.fsum(column)/size for column in zipped]
        
        return centroid
    
    def updateCentroid(self, vectors):
        """Returns how much the centroids moved
        from his previous place and update the new centroid"""
        previous_centroid = self.centroid
        self.vectors = vectors
        self.centroid = self.calcCentroid()
        shift = distance(previous_centroid, self.centroid)
        
        return shift


# In[4]:

def distance(x, y):
    """Euclidan distance between two vectors"""
    dist = [pow((x-y), 2) for x, y in zip(x,y)]
    dist = math.sqrt(sum(dist))
    
    return dist


# In[5]:

def kmeans(vectors, k, converaged):
    
    # Select random k vectors as initial centroids"""
    init_vectors = random.sample(vectors, k)
    
    # Create k clusters using init_vectors and used as init centroids
    clusters = [Cluster([v]) for v in init_vectors]
    
    # How many interations until centroids stabilize - converage
    iteration = 0
    
    while True:
        
        # Vector (list) which holds all the cluster points in the given cluster
        cluster_points = [[] for c in clusters]
        n_cluster = len(clusters)
        
        # Start the iteration
        iteration += 1
        
        # For every feature from the vectors (dataset)
        for v in vectors:
            # Distance between the first feature and the centroid of the 1st cluster
            distance_ = distance(v, clusters[0].centroid)
            cluster_no = 0
            
            for i in range(1, n_cluster):
                dist = distance(v, clusters[i].centroid)
                # Find the smallest distance from each remaining centroids
                if dist < distance_:
                    distance_ = dist
                    cluster_no = i
            
            cluster_points[cluster_no].append(v)
        
        # Initial (biggest) shift
        shift = 0.0
        
        for i in range(n_cluster):
            # Calculate how much the centroids moved in the interation
            shift_ = clusters[i].updateCentroid(cluster_points[i]) 
            # Largest Move from all the clusters
            shift = max(shift, shift_)
            
        if shift_ < converaged:
            print("Convereged after " + repr(iteration) + " iterations")
            break
        
    
    return clusters


# In[31]:

features, labels = load_data()


# In[32]:

converaged = 0.2e-5


# In[39]:

k = 6


# In[40]:

clusters = kmeans(features, k, converaged)


# In[43]:

def plot(X, clusters, cluster_num):
    """Visualizing the Cluster Centroids
    and Data Points in 3D"""
    centroids = []
    X = np.asarray(X)
    for i in range(0,cluster_num):
        centroids.append(clusters[i].centroid)
    
    np_centroids = np.asarray(centroids)
    
    color = ["g", "r", "b", "c", "m", "b"]
    
    fig = figure()
    ax = fig.gca(projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=color)
    
    ax.scatter(np_centroids[:, 0],np_centroids[:, 1],
               np_centroids[:, 2], marker = "x", s=150,
               linewidths = 5, zorder = 100, c=color
              )


# In[44]:

plot(features, clusters, k)


# In[22]:

def elbow(features, K, converaged):
    variance = []
    for k in range(1, K):
        clusters = kmeans(features, k, converaged)
        dist = []
        for c in clusters:
            for v in c.vectors:
                d = pow(distance(v, c.centroid), 2)
                dist.append(d)
        variance.append(sum(dist)/k)
        
    return variance


# In[23]:

var = elbow(features, 60, converaged)


# In[28]:

def plot_variance(variance):
    plt.xlabel('K')
    plt.ylabel('Average within-cluster variation')
    plt.title('Cluster Quality - Finding the right K')
    plt.plot(var)
    plt.show()


# In[29]:

plot_variance(var)


# In[ ]:



