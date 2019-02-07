from pylab import *
import scipy.special as sp
N,k = 101,9
t=linspace(0,10,N)
y=1.05*sp.jn(2,t)-0.105*t # f(t) vector
Y=meshgrid(y,ones(k),indexing='ij')[0] # make k copies
scl=logspace(-1,-3,k) # noise stdev
n=dot(randn(N,k),diag(scl))
yy=Y+n
plot(t,yy)
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Plot of the data to be fitted')
grid(True)
savetxt("fitting.dat",c_[t,yy]) # write out matrix to file
show()