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
        
    N,k,A0,B0 = 101,9,1.05,-0.105
    sigma = np.logspace(-1,-3,9)
    t = data[:,0]
    gold = g(t,A0,B0)
    f = data[:,1]
    t = np.linspace(0,10,N)
    y = 1.05*sp.jn(2,t)-0.105*t # f(t) vector
    Y = np.meshgrid(y,np.ones(k),indexing='ij')[0] # make k copies
    n = np.dot(np.random.randn(N,k),np.diag(sigma))
    yy = Y+n
    plt.grid();plt.plot(t,yy);plt.plot(t,gold,color='k')
    plt.xlabel(r'$t$');plt.ylabel(r'$f(t)+error$')
    plt.title(r'Data to be fitted to theory')
    plt.legend(['0.1000','0.0562','0.0316','0.0178','0.0100','0.0056','0.0032','0.0018','0.0010','gold'])
    plt.show()

    plt.grid()
    plt.plot(t,gold)
    plt.errorbar(t[::5],f[::5],yerr=0.1,fmt = '.',capsize=3)
    plt.title('Error bars for sigma = 0.1')
    plt.xlabel('t')
    plt.ylabel('f(t) + error')
    plt.legend(['gold','error_bar'])
    plt.show()

    M = np.vstack((sp.jn(2,t),t)).T
    A = np.linspace(0,2,21)
    B = np.linspace(-0.2,0,21)
    error = np.zeros((len(A),len(B)))
    mean_error = np.zeros(9)
    a,b = [],[]
    for k in range(1,10):
        if k == 1:
            for i in range(len(A)):
                for j in range(len(B)):
                    error[i,j] = np.mean(np.square(data[:,k] - g(t,A[i],B[j])))
        X,_,_,_ = np.linalg.lstsq(M,data[:,k],rcond=-1)
        a.append(X[0])
        b.append(X[1])
        mean_error[k-1] = np.mean(np.square((M.dot(X) - data[:,k])))
    
    Aerr = abs(A0-np.asarray(a))
    Berr = abs(B0-np.asarray(b))
    plt.grid()
    plt.plot(sigma,Aerr,'o--')
    plt.plot(sigma,Berr,'o--')
    plt.title('Aerr and Berr for different sigma')
    plt.xlabel('sigma')
    plt.ylabel('Aerr and Berr')
    plt.legend(['Aerr','Berr'])
    plt.show()

    plt.grid()
    plt.loglog(sigma,Aerr,'o')
    plt.loglog(sigma,Berr,'o')
    plt.errorbar(sigma,Aerr,yerr=0.1,fmt = 'o')
    plt.errorbar(sigma,Berr,yerr=0.1,fmt = 'o')
    plt.title('Aerr and Berr for different sigma')
    plt.xlabel('sigma')
    plt.ylabel('Aerr and Berr in logscale')
    plt.legend(['Aerr in logscale ','Berr in logscale'])
    plt.show()

    plt.grid()
    plt.plot(sigma,mean_error,'o--')
    plt.title('MSError for f(t) different values of A and B')
    plt.xlabel('sigma')
    plt.ylabel('MSError')
    plt.show()

    plt.grid()
    plt.loglog(sigma,mean_error,'o--')
    plt.title('MSError of f(t) for different values of A and B')
    plt.xlabel('sigma in logscale')
    plt.ylabel('MSError in logscale')
    plt.show()

    plt.grid()
    plt.text(A0,B0,'exact location')
    cp = plt.contour(A,B,error,20)
    plt.clabel(cp, inline=True,fontsize=10)
    A,B = np.meshgrid(A,B)
    plt.title('Contour Plot for sigma = 0.1')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.show()

if __name__ == '__main__':
    main()