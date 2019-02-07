#!/usr/bin/env python3

import sys
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    try:
        data = np.loadtxt('fitting.dat')
    except IOError:
        print('Could not read the file.')
        sys.exit()        

    def g(t,A,B):
        """
            Return the A*sp.jn(2,t) + B*t
        """
        return A*sp.jn(2,t) + B*t
        
    sigma = np.logspace(-1,-3,9)
    t = data[:,0]
    gold = g(t,1.05,-0.105)
    f = data[:,1]
    for k in range(len(sigma)):
        f = data[:,k+1]
        plt.plot(t,f,label = 'f(t) + err')
        plt.plot(t,gold,label = 'gold')
        plt.title('sigma' + '=' + str(sigma[k]))
        plt.xlabel('t');plt.ylabel('f(t) + error')
        plt.show()

    plt.plot(t,gold);plt.plot(t,data[:,1])
    plt.errorbar(t[::5],data[:,1][::5],fmt='ro')
    plt.title('Error bars')
    plt.xlabel('t');plt.ylabel('f(t) + error')
    plt.pause(3);plt.close()

    M = np.vstack((sp.jn(2,t),t)).T
    A = np.linspace(0,2,21)
    B = np.linspace(-0.2,0,21)
    error = np.zeros((len(A),len(B)))
    mean_error = np.zeros(9)
    for k in range(1,10):
        for i in range(len(A)):
            for j in range(len(B)):
                error[i,j] = (101**-1)*np.mean(np.square(data[:,k] - g(t,A[i],B[j])))
        X,_,_,_ = np.linalg.lstsq(M,data[:,k],rcond=-1)
        mean_error[k-1] = np.mean(np.square((M.dot(X) - data[:,k])))

    plt.plot(sigma ,mean_error);plt.loglog(sigma ,mean_error)
    plt.pause(3);plt.close()
    plt.contour(A,B,error)
    A,B = np.meshgrid(A,B)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(A,B,error)
    plt.show()

if __name__ == '__main__':
    main()