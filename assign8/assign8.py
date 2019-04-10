import numpy as np
import matplotlib.pyplot as plt

################# sin(5t) ##################

x = np.linspace(0,2*np.pi,129);x = x[:-1]
y = np.sin(5*x)
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/128.0
w = np.linspace(-64,64,129);w = w[:-1]
print(np.angle(Y,deg = False))
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
plt.xlim([-10,10])
plt.title(r'Magnitude of DFT of $sin(5t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 10)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 10)
plt.xlim([-10,10])
plt.title(r'Phase of DFT of $sin(5t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()

################# (1+0.1cos(t))cos(10t) ##################

t = np.linspace(-4*np.pi,4*np.pi,513);t = t[:-1]
y = np.cos(10*t)*(1+0.1*np.cos(t))
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512.0
w = np.linspace(-64,64,513);w = w[:-1]
print(np.angle(Y,deg = False))
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
plt.xlim([-15,15])
plt.title(r'Magnitude of DFT of $(1+0.1cos(t))cos(10t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 5)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 5)
plt.xlim([-15,15])
plt.title(r'Phase of DFT of $(1+0.1cos(t))cos(10t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()

################# sin**3(t) ##################

t = np.linspace(-4*np.pi,4*np.pi,513);t = t[:-1]
y = (np.sin(t))**3
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512.0
w = np.linspace(-64,64,513);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
plt.xlim([-15,15])
plt.title(r'Magnitude of DFT of $sin^3(t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 5)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 5)
plt.xlim([-15,15])
plt.title(r'Phase of DFT of $sin^3(t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()

################# cos**3(t) ##################

t = np.linspace(-4*np.pi,4*np.pi,513);t = t[:-1]
y = (np.cos(t))**3
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512.0
w = np.linspace(-64,64,513);w = w[:-1]
print(np.angle(Y,deg = False))
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
plt.xlim([-15,15])
plt.title(r'Magnitude of DFT of $cos^3(t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 5)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 5)
plt.xlim([-15,15])
plt.title(r'Phase of DFT of $cos^3(t)$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()

# ################# cos(20t+5cos(t)) ##################

t = np.linspace(-4*np.pi,4*np.pi,513);t = t[:-1]
y = np.cos(20*t+5*np.cos(t))
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512.0
w = np.linspace(-64,64,513);w = w[:-1]
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
plt.xlim([-40,40])
plt.title(r'Magnitude of DFT of $cos(20t+5cos(t))$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 5)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 5)
plt.xlim([-40,40])
plt.title(r'Phase of DFT of $cos(20t+5cos(t))$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()

################# exp(-t**2/2) ##################

N = 2049
t = np.linspace(-64,64,N);t = t[:-1]
y = np.exp(-(t**2)/2.0)
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/float(N-1)
w = np.linspace(-64,64,N);w = w[:-1]
y1 =(np.sqrt(2*np.pi))*np.exp(-2*(np.pi**2)*(w**2))/float(N-1)
plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,abs(Y))
# plt.plot(w,abs(y1),'-.')
plt.xlim([-20,20])
plt.title(r'Magnitude of DFT of $e^{-t^2/2}$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$|Y|\rightarrow$',size = 16)
plt.show()

plt.figure(figsize=(16,8))
plt.grid()
plt.plot(w,np.angle(Y,deg=False),'+',markersize = 5)
ii = np.where(abs(Y)>=0.001)
plt.plot(w[ii],np.angle(Y[ii]),'ro',markersize = 5)
plt.xlim([-20,20])
plt.title(r'Phase of DFT of $e^{-t^2/2}$')
plt.xlabel(r'$k\rightarrow$',size = 16)
plt.ylabel(r'$\angle Y\rightarrow$',size = 16)
plt.show()







