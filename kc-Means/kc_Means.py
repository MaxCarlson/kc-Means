import csv
import copy
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Base:
    def __init__(self, data, clusters, r):
        self.r = r
        random.seed(1)
        self.data = data
        self.clusters = clusters
        self.minx = np.min(data[0])
        self.miny = np.min(data[1])
        self.maxx = np.max(data[0])
        self.maxy = np.max(data[1])

    def randomPoints(self):
        return [(random.uniform(self.minx, self.maxx), 
            random.uniform(self.miny, self.maxy)) for _ in range(self.r)]


class K_Means(Base):
    def __init__(self, data, k, r):
        super().__init__(data, k, r)

        best = np.inf
        bestCentroids = None
        for i in range(r):
            centroids, sses = self.run()
            if sses < best:
                best = sses
                bestCentroids = centroids

    def run(self):

        centroids = self.randomPoints()
        prevBest = None
        bestCentroids = np.ones((len(self.data), 2)) * np.inf

        i = 0
        while True:
            prevBest = bestCentroids
            bestCentroids = np.ones((len(self.data), 2)) * np.inf

            # Assignment
            for i, c in enumerate(centroids): 
                norm = np.linalg.norm((self.data - c), 2, 1)
                bestCentroids[:,1] = np.where(norm < bestCentroids[:,1], 
                                              norm, bestCentroids[:,1])
                bestCentroids[:,0] = np.where(norm == bestCentroids[:,1], 
                                              i, bestCentroids[:,0])

                a=4

            if (prevBest == bestCentroids).all():
                break

            # Update 
            notDone = False
            for i, c in enumerate(centroids): 

                # Get data values belonging to centroid i
                m = np.where(bestCentroids[:,0] == i)
                x = self.data[m,:]

                if notDone and prevWheres[i] != x:
                    notDone = True
                
                # Update the centroid
                centroids[i] = 1/len(bestCentroids[:,0]) * np.sum(x, 1)
                
                vv = 0
            i += 1

        return centroids, np.sum(bestCentroids[:,1])


            




                




class C_Means(Base):
    def __init__(self, data, c, r):
        super().__init__(data, c, r)







data = np.genfromtxt('545_cluster_dataset.txt')

k = K_Means(data, 3, 2)