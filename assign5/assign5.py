#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    if len(sys.argv) != 5:
        print("Incorrect format.The correct format is $ python3 assign5.py Nx Ny Radius Niter")
        print("For best results, 40 <= Nx <= 200,40 <= Nx <= 200,Nx = Ny,Niter = 50*|Nx|,Radius = 0.35*Nx")
        sys.exit()
    else:
        Nx,Ny,Radius,Niter = list(map(int,sys.argv[1:]))
        if 2*Radius > Nx:
            print("Too large radius !!")
            sys.exit()

    V = np.zeros((Nx,Ny))
    x = np.linspace(-1,1,Nx)
    y = np.linspace(-1,1,Ny)
    Y,X = np.meshgrid(y,x)
    Rad_norm = (Radius*2)/Nx
    ii = np.where(X*X + Y*Y < Rad_norm*Rad_norm)
    V[ii] = 1.0

    error = np.zeros((Niter,))

    fig = plt.figure(figsize=(8,8))
    plt.grid()
    plt.scatter(ii[0]-Nx/2,ii[1]-Ny/2,s=4,color='r',marker='o')
    plt.title('Initial Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-Nx/2,Nx/2))
    plt.ylim((-Ny/2,Ny/2))
    plt.legend(['V = 1'])
    plt.show()

    for k in range(Niter):
        V_old = V.copy()
        V[1:-1,1:-1] = 0.25*(V[1:-1,0:-2] + V[1:-1,2:] + V[0:-2,1:-1] + V[2:,1:-1])
        V[0,1:-1],V[-1,1:-1],V[:,-1] = V[1,1:-1],V[-2,1:-1],V[:,-2]
        V[ii] = 1.0
        error[k] = np.max(abs(V - V_old))

    plt.figure(figsize=(8,8))
    plt.grid()
    plt.title('Contour plot of the potential')
    plt.scatter((ii[0]-Nx/2)*(2/Nx),(ii[1]-Ny/2)*(2/Ny),s=4,color='r',marker='o')
    cp = plt.contour(X,Y,V,20)
    levels = np.linspace(0,1,11)
    plt.clabel(cp,levels,inline=True,fontsize = 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    log_error = np.log(error)
    error_new = log_error
    n_iter = np.arange(1,Niter+1)

    plt.figure(figsize=(8,8))
    plt.grid()
    plt.title('Error in loglog scale for all iterations')
    plt.ylabel('Error in loglog scale')
    plt.xlabel('number of iterations')
    plt.plot(np.log(n_iter),error_new,'r')
    plt.legend(['actual'])
    plt.show()

    ones = np.ones_like(n_iter)
    n_iter = np.vstack((ones,n_iter)).T

    plt.figure(figsize=(8,8))
    plt.grid()
    plt.title('Error in semilog scale for all iterations')
    plt.ylabel('Error in semilog scale')
    plt.xlabel('number of iterations')
    plt.plot(n_iter,error_new,'r')
    A,_,_,_ = np.linalg.lstsq(n_iter,error_new,rcond=-1)
    print(A)
    plt.plot(n_iter,n_iter.dot(A),color='b')
    plt.xlim((0,Niter))
    plt.legend(['actual','reconstructed'])
    plt.show()

    error_new = log_error[int(Niter*0.2):]
    n_iter = np.arange(int(Niter*0.2),Niter)
    ones = np.ones_like(n_iter)
    n_iter = np.vstack((ones,n_iter)).T

    plt.figure(figsize=(8,8))
    plt.grid()
    plt.title('Error in semilog scale after {} iterations'.format(int(Niter*0.2)))
    plt.ylabel('Error in semilog scale')
    plt.xlabel('number of iterations')
    plt.plot(n_iter,error_new,'r')
    A,_,_,_ = np.linalg.lstsq(n_iter,error_new,rcond=-1)
    print(A)
    plt.plot(n_iter,n_iter.dot(A),color='b')
    plt.xlim((Niter*0.15,Niter*1.05))
    plt.legend(['actual','reconstructed'])
    plt.show()

    fig = plt.figure(figsize=(10,8))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, V,cmap = cm.jet,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('3-D Surface plot of the potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    Jx,Jy = np.zeros_like(V),np.zeros_like(V)
    plt.figure(figsize=(8,8))
    plt.grid()
    Jy[:,1:-1] = 0.5*(V[:,:-2] - V[:,2:])
    Jx[1:-1] = 0.5*(V[:-2] - V[2:])
    plt.scatter((ii[0]-Nx/2)*(2/Nx),(ii[1]-Ny/2)*(2/Ny),s=4,color='r',marker='o')
    plt.quiver(X,Y,Jx,Jy)
    plt.title('Quiver plot of the current densities')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Electrode','Current density'])
    plt.show()

    J_sq = Jx**2 + Jy**2
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.title('Contour plot of the heat generated')
    cp = plt.contour(X,Y,J_sq,10)
    plt.clabel(cp,inline=True,fontsize = 10,colors='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()