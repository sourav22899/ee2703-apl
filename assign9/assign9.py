import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

################# sin(sqrt(2)t) #################

N = 512
t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = np.sin(np.sqrt(2)*t)
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$sin(\sqrt{2}t)w(t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()

t = np.linspace(-8*np.pi,8*np.pi,N+1);t = t[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = np.sin(np.sqrt(2)*t)
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N
w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Magnitude\;of\;sin(\sqrt{2}t)$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.xlim([-5,5])
plt.stem(w,abs(y),markerfmt='.')
plt.show()

################# cos**3(0.86t) #################

N = 1024
t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(0.86*t))**3
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$cos^3(0.86t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()

t = np.linspace(-32*np.pi,32*np.pi,N+1);t = t[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(0.86*t))**3
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N
w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Magnitude\;of\;cos^3(0.86t)\;without\;Hamming\;window$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.xlim([-5,5])
plt.stem(w,abs(y),markerfmt='.')
plt.show()

t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(0.86*t))**3
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$cos^3(0.86t)w(t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()

t = np.linspace(-32*np.pi,32*np.pi,N+1);t = t[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(0.86*t))**3
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N
w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Magnitude\;of\;cos^3(0.86t)\;with\;Hamming\;window$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.xlim([-5,5])
plt.stem(w,abs(y),markerfmt='.')
plt.show()

################# cos(at+b) #################

seed_omega = 485
np.random.seed(seed_omega)
omega = np.random.random() + 0.5
seed_phi = 5655
np.random.seed(seed_phi)
phi = np.random.random()

print('Actual value of omega:{}'.format(omega))
print('Actual value of phi:{}'.format(phi))
N_sam = 128
t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N_sam+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N_sam+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = np.cos(omega*t + phi)
plt.figure(figsize=(16,8))
plt.grid(True)
plt.title(r'cos($\omega t + \phi$) without Hamming Window')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.plot(t1,y,'r')
plt.plot(t2,y,'r')
plt.show()

t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
n = np.arange(N_sam)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N_sam-1)))
y = y*wnd
plt.figure(figsize=(16,8))
plt.grid(True)
plt.title(r'cos($\omega t + \phi$) with Hamming Window')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.plot(t1,y,'r')
plt.plot(t2,y,'r')
plt.show()

t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
n = np.arange(N_sam)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N_sam-1)))
y = np.cos(omega*t + phi)
y = y*wnd
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N_sam
w = np.linspace(-np.pi*fmax,np.pi*fmax,N_sam+1);w = w[:-1]

t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
n = np.arange(N_sam)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N_sam-1)))
z = np.cos(t)
z = z*wnd
z[0] = 0
z = np.fft.fftshift(z)
z = np.fft.fftshift(np.fft.fft(z))/N_sam
w = np.linspace(-np.pi*fmax,np.pi*fmax,N_sam+1);w = w[:-1]

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.grid(True)
plt.title(r'Magnitude of cos($\omega t + \phi$)')
plt.xlabel(r'$k\rightarrow$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlim([-40,40])
plt.stem(w,abs(y),markerfmt='.')
# plt.stem(w,abs(z),markerfmt='+')

plt.subplot(2,1,2)
plt.grid(True)
plt.title(r'Phase of cos($\omega t + \phi$)')
plt.xlabel(r'$k\rightarrow$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlim([-40,40])
ii = np.where(abs(y)>=0.001)
jj = np.where(abs(z)>=0.001)
plt.stem(w[ii],np.angle(y[ii]),markerfmt='.')
plt.stem(w[jj],np.angle(z[jj]),markerfmt='+')
plt.show()

print('Calculated value of omega:{}'.format(np.sum(abs(w)*abs(y**2.1))/np.sum(abs(y**2.1))))
print('Calculated value of phi:{}'.format(-np.angle(z[N_sam//2 + 1])+np.angle(y[N_sam//2 +1])))

################# cos(at+b) with noise #################

N_sam = 128
t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N_sam+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N_sam+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
seed_noise = 565
np.random.seed(seed_noise)
y = np.cos(omega*t + phi) + 0.1*np.random.randn(N_sam)
plt.figure(figsize=(16,8))
plt.grid(True)
plt.title(r'cos($\omega t + \phi$)+Gaussian Noise without Hamming Window')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.plot(t1,y,'r')
plt.plot(t2,y,'r')
plt.show()

t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
n = np.arange(N_sam)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N_sam-1)))
y = y*wnd
plt.figure(figsize=(16,8))
plt.grid(True)
plt.title(r'cos($\omega t + \phi$)+Gaussian Noise with Hamming Window')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.plot(t1,y,'r')
plt.plot(t2,y,'r')
plt.show()

t = np.linspace(-np.pi,np.pi,N_sam+1);t = t[:-1]
n = np.arange(N_sam)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N_sam-1)))
y = np.cos(omega*t + phi) + 0.1*np.random.randn(N_sam)
y = y*wnd
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N_sam

# w = np.linspace(-np.pi*fmax,np.pi*fmax,N_sam+1);w = w[:-1]
# plt.figure(figsize=(16,8))
# plt.grid(True)
# plt.title(r'Magnitude of cos($\omega t + \phi$)+Gaussian Noise')
# plt.xlabel(r'$k\rightarrow$')
# plt.ylabel(r'$|Y|\rightarrow$')
# plt.xlim([-40,40])
# plt.stem(w,abs(y),markerfmt='.')
# plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.grid(True)
plt.title(r'Magnitude of cos($\omega t + \phi$) with Gaussian Noise')
plt.xlabel(r'$k\rightarrow$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlim([-40,40])
plt.stem(w,abs(y),markerfmt='.')
# plt.stem(w,abs(z),markerfmt='+')

plt.subplot(2,1,2)
plt.grid(True)
plt.title(r'Phase of cos($\omega t + \phi$) with Gaussian noise')
plt.xlabel(r'$k\rightarrow$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlim([-40,40])
ii = np.where(abs(y)>=0.001)
jj = np.where(abs(z)>=0.001)
plt.stem(w[ii],np.angle(y[ii]),markerfmt='.')
plt.stem(w[jj],np.angle(z[jj]),markerfmt='+')
plt.show()

print('Calculated value of omega with noise:{} '.format(np.sum(abs(w)*abs(y**2.6))/np.sum(abs(y**2.6))))
print('Calculated value of phi with noise:{}'.format(-np.angle(z[N_sam//2 + 1])+np.angle(y[N_sam//2 +1])))
# print('Angular Frequency calculated when noise is present:{}'.format((np.sum(abs(w)*abs(y**2))/np.sum(abs)(y**2))))

################# linear chirped signal #################

N = 1024
t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
t1 = np.linspace(-3*np.pi,-np.pi,N+1);t1 = t1[:-1]
t2 = np.linspace(np.pi,3*np.pi,N+1);t2 = t2[:-1]
dt = t[1] - t[0];fmax = 1/dt
y = (np.cos(16*t*(1.5+t/(2*np.pi))))
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$cos(16t(1.5+\frac{t}{2\pi}))w(t)$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.plot(t,y,'r')
plt.plot(t1,y,'b')
plt.plot(t2,y,'b')
plt.show()

t = np.linspace(-np.pi,np.pi,N+1);t = t[:-1]
dt = t[1] - t[0];fmax = 1/dt
n = np.arange(N)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
y = y*wnd
y[0] = 0
y = np.fft.fftshift(y)
y = np.fft.fftshift(np.fft.fft(y))/N
w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Magnitude\;of\;cos(16t(1.5+\frac{t}{2\pi}))$')
plt.ylabel(r'$|Y|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.xlim([-100,100])
plt.stem(w,abs(y),markerfmt='.')
plt.show()

################# Time-Frequency Plot ###########

t = np.reshape(t,(64,-1))
Y = np.zeros_like(t,dtype=np.complex128)
for i in range(t.shape[1]):
    x = t[:,i]
    y = Y[:,i]
    y = np.cos(16*x*(1.5+x/(2*np.pi)))
    n = np.arange(64)
    wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(63)))
    y = y*wnd
    y[0] = 0
    y = np.fft.fftshift(y)
    Y[:,i] = np.fft.fftshift(np.fft.fft(y))/64.0

x = np.arange(t.shape[1])
y = np.arange(t.shape[0])
y,x = np.meshgrid(y,x)

Y = np.fft.fftshift(Y,axes=0)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Time-Frequency\;Plot$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$x\rightarrow$')
cp = plt.contour(y,x,abs(Y.T),20,cmap=cm.jet)
plt.clabel(cp,inline=True,fontsize=7)
plt.show()

fig = plt.figure(figsize=(16,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(y,x,abs(Y.T),cmap=cm.jet,linewidth=0, antialiased=False)
plt.title(r'Surface Plot of Time-Frequency Plot')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()