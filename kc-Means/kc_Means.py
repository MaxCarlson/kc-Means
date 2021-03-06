import csv
import copy
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import plotly.graph_objects as go
import matplotlib.tri as tri
from scipy.interpolate import griddata
from matplotlib import cm

class Base:
    def __init__(self, data, clusters, r):
        self.r = r
        #random.seed(2)
        self.data = data
        self.clusters = clusters
        self.minx = np.min(data[:,0])
        self.miny = np.min(data[:,1])
        self.maxx = np.max(data[:,0])
        self.maxy = np.max(data[:,1])

    def randomPoints(self):
        return [(random.uniform(self.minx, self.maxx), 
            random.uniform(self.miny, self.maxy)) for _ in range(self.clusters)]


class K_Means(Base):
    def __init__(self, data, k, r):
        super().__init__(data, k, r)

        best = np.inf
        bestCentroids = None
        for i in range(r):
            centroids, sses, closestsCentroids = self.run()
            if sses < best:
                best = sses
                bestCentroids = np.array(centroids)[:,0,:]


        for i in range(self.clusters):
           plt.plot(self.data[closestsCentroids==i, 0], 
                    self.data[closestsCentroids==i, 1], 'o', 
                    zorder=0, color='rbgkmcy'[i % len('rbgkmcy')])
        print('BestSSE:', best)
        plt.title('SSE: %.2f' % sses + ' k='+str(k) + ' r='+str(r))
        plt.show()

        fig, ax = plt.subplots()
        vor = Voronoi(bestCentroids)
        fig = voronoi_plot_2d(vor, ax=ax)
        ax.scatter(self.data[:,0], self.data[:,1])
        ax.xaxis.zoom(-12)
        ax.yaxis.zoom(-15)
        plt.show()

    def run(self):

        centroids = self.randomPoints()
        prevBest = None
        bestCentroids = np.ones((len(self.data), 2)) * np.inf

        it = 0
        while True:
            prevBest = bestCentroids
            bestCentroids = np.ones((len(self.data), 2)) * np.inf

            # Assignment
            for i, c in enumerate(centroids): 

                # Calculate the 2 norm of all points to this centroid
                norm = np.linalg.norm((self.data - c), 2, 1)

                # Find the rows in which the 2norm is less than the previous best
                brows = np.where(norm < bestCentroids[:,1])

                # Replace the norm, then replace that best centroid index
                bestCentroids[brows, 1] = norm[brows] 
                bestCentroids[brows, 0] = i

                a=4

            if (prevBest == bestCentroids).all():
                break

            # Update 
            for i, c in enumerate(centroids): 

                # Get data values belonging to centroid i
                m = np.where(bestCentroids[:,0] == i)
                x = self.data[m,:]

                # Update the centroid
                centroids[i] = 1/(np.shape(x)[1] if np.shape(x)[1] else 1) \
                   * np.sum(x, 1)
                
                vv = 0
            it += 1

        return centroids, np.sum(bestCentroids[:,1]), bestCentroids[:,0]



def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))
            
def plot_countour(i, x,y,z, npts, mins, maxs):
    cmaps = [cm.Reds, cm.Blues, cm.Greens, cm.Greys, cm.PuRd, cm.GnBu, cm.YlOrBr]

    # define grid.
    xi = np.linspace(mins[0], maxs[0], 100)
    yi = np.linspace(mins[1], maxs[1], 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.05, 0.2, 0.5, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cmaps[i  % len(cmaps)], 
                      levels=levels, zorder=1, alpha=0.7)


class C_Means(Base):


    def __init__(self, data, c, r, m=1.00001):
        super().__init__(data, c, r)

        bestLoss = np.inf
        bestGrades = None
        for i in range(r):
            grades, centroids, loss = self.run(m)
            if loss < bestLoss:
                bestGrades = grades
                bestLoss = loss

        print('BestSSE:', bestLoss)
        bestGrade = np.argmax(bestGrades, 1)

        for i in range(self.clusters):
           plt.plot(self.data[bestGrade==i, 0], self.data[bestGrade==i, 1], 'o', 
                    zorder=0, color='rbgkmcy'[i % len('rbgkmcy')])

        X = self.data[:,0]
        Y = self.data[:,1]

        for i in range(self.clusters):
            w = np.where(bestGrade == i)
            b = self.data[w]
            cov = np.cov(np.transpose(b))
            u = np.mean(b, axis=0)
        
            npts = 100
            z = gauss(X, Y, cov, u)
            plot_countour(i, X, Y, z, npts, (self.minx, self.miny), 
                          (self.maxx, self.maxy))
        plt.title('SSE: %.2f' % bestLoss + ' c='+str(c) + ' r='+str(r) + ' m='+str(m))
        plt.show()

    def run(self, m):
        centroids = self.randomPoints()
        prevBest = None
        memberGrades = np.random.rand(len(self.data), self.clusters)
        
        while True:
            # Compute centroids
            pow = np.power(memberGrades, m)
            num = np.dot(np.transpose(pow), self.data)
            den = np.sum(pow, 0)
            centroids = num / np.transpose(np.stack([den]*2))

            cur = np.argmax(memberGrades, 1)
            if (prevBest == cur).all():
                break
            prevBest = cur

            # Compute membership grades
            n = np.zeros(np.shape(memberGrades))
            for j in range(self.clusters):
                n[:,j] = np.linalg.norm(self.data - centroids[j], 2, axis=1)

            s = np.zeros(np.shape(memberGrades))
            for k in range(self.clusters):
                d = np.linalg.norm(self.data - centroids[k], 2, axis=1)
                for j in range(self.clusters):
                    s[:,j] += n[:,j] / d
            memberGrades = 1 / np.power(s, 2/(m-1))
            a = 5

        bestGrades = np.argmax(memberGrades, axis=1)
        loss = np.sum(np.linalg.norm(self.data - centroids[bestGrades], axis=1))

        return memberGrades, centroids, loss

        




data = np.genfromtxt('545_cluster_dataset.txt')

k = K_Means(data, 10, 5)
#c = C_Means(data, 3, 10, 10)