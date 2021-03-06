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
            centroids, sses = self.run()
            if sses < best:
                best = sses
                bestCentroids = np.array(centroids)[:,0,:]

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

        return centroids, np.sum(bestCentroids[:,1])


            



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

        bestGrade = np.argmax(bestGrades, 1)
        #bestGrade = np.average(bestGrades, 1)

        npts = 100
        fig, ax1 = plt.subplots(nrows=1)
        xi = np.linspace(self.minx, self.maxx, npts)
        yi = np.linspace(self.miny, self.maxy, npts)

        triang = tri.Triangulation(self.data[:,0], self.data[:,1])
        interpolator = tri.LinearTriInterpolator(triang, bestGrade)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        ax1.contour(xi, yi, zi, levels=self.clusters, linewidths=0.5, colors='k')
        cntr1 = ax1.contourf(xi, yi, zi, levels=self.clusters, cmap="plasma")

        fig.colorbar(cntr1, ax=ax1)
        ax1.plot(self.data[:,0], self.data[:,1], 'ko', ms=3)
        ax1.set(xlim=(self.minx, self.maxx), ylim=(self.miny, self.maxy))
        ax1.set_title('grid and contour (%d points, %d grid points)' %
                      (len(self.data), npts*npts))

        #ax2.tricontour(self.data[:,0], self.data[:,1], bestGrade, 
        #               levels=14, linewidths=0.5, colors='k')
        #cntr2 = ax2.tricontourf(self.data[:,0], self.data[:,1], bestGrade,
        #                       levels=14, cmap="inferno")
        #
        #fig.colorbar(cntr2, ax=ax2)
        #ax2.plot(self.data[:,0], self.data[:,1], 'ko', ms=3)
        #ax2.set(xlim=(-2, 2), ylim=(-2, 2))
        #ax2.set_title('tricontour (%d points)' % len(self.data))

        #ax1.xaxis.zoom(-12)
        #ax1.yaxis.zoom(-15)

        plt.subplots_adjust(hspace=0.5)
        #plt.show()


        #fig = go.Figure(data=go.Contour(z=bestGrade, colorscale='Electric'))
        #fig.show()
        #plt.contour(self.data[:,0], self.data[:,1], bestGrade)

        fig, ax = plt.subplots()
        for i in range(self.clusters):
            ax.plot(self.data[bestGrade==i, 0], self.data[bestGrade==i, 1], 'o')
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

#k = K_Means(data, 14, 1)
c = C_Means(data, 5, 10, 5)