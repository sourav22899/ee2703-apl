import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import csv
import string

filter = []
with open('h.csv','r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        filter.append(row)

filter = np.reshape(np.asarray(filter),(-1,))
filter = np.asarray(filter,dtype=np.float32)
w,h = sp.freqz(filter,1,whole=True)
h = np.fft.fftshift(h)

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.grid()
plt.title(r'Magnitude Response of Digital Filter')
plt.xlabel(r'$f\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.plot(w-np.pi,abs(h))

plt.subplot(2,1,2)
plt.grid()
plt.title(r'Phase Response of Digital Filter')
plt.xlabel(r'$f\rightarrow$')
plt.ylabel(r'$\angle H(j\omega)\rightarrow$')
plt.plot(w-np.pi,np.angle(h))
plt.show()

n = np.arange(1,1025)
x = np.cos(0.2*np.pi*n)
x1 = np.cos(0.85*np.pi*n)

plt.figure(figsize=(16,8))
plt.subplot(3,1,1)
plt.grid()
plt.title(r'cos(0.2$\pi$t)')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$x\rightarrow$')
plt.xlim([0,100])
plt.plot(n,x,'g-.')
plt.legend([r'cos(0.2$\pi$t)'])

plt.subplot(3,1,2)
plt.grid()
plt.title(r'cos(0.85$\pi$t)')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$x\rightarrow$')
plt.xlim([0,100])
plt.plot(n,x1,'r-.')
plt.legend([r'cos(0.85$\pi$t)'],loc=1)

plt.subplot(3,1,3)
plt.grid()
plt.title(r'cos(0.2$\pi$t)+cos(0.85$\pi$t)')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$x\rightarrow$')
plt.xlim([0,100])
plt.plot(n,x+x1)
plt.legend([r'cos(0.2$\pi$t)+cos(0.85$\pi$t)'])
plt.show()

y = sp.fftconvolve(x,filter,mode='full')
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'Output Signal')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([0,200])
plt.plot(y)
plt.show()

############## Circular Conv ###################

padded_filter = np.zeros_like(x)
padded_filter[:len(filter)] = filter

y1 = np.fft.ifft(np.fft.fft(x)*np.fft.fft(padded_filter))
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'Output Signal')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([0,200])
plt.plot(np.real(y1))
plt.show()

x += x1

############### Circular using Linear ################

P = 2**4
N = 2**5 - 1
pad_filter = np.zeros(N)
pad_filter[:len(filter)] = filter
X = np.reshape(x,(-1,P))
Y = np.zeros(len(x)+len(filter)-1,dtype=np.complex)
for i in range(64):
    x_t = X[i]
    x_t_pad = np.pad(x_t,(0,N-P),'constant',constant_values=(0))
    y_t = np.fft.ifft(np.fft.fft(x_t_pad)*np.fft.fft(pad_filter))
    if i < 63:
        Y[16*i:16*i + N] += y_t
    else:
        Y[16*i:] += y_t[:27]

plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'Output Signal by Linear Convolution using Circular Convolution')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([0,200])
plt.plot(np.real(Y))
plt.show()

############### Zadoff-Chu ###################

zc_filter_raw,zc_filter = [],[]
with open('x1.csv','r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        zc_filter_raw.append(row)

for x in zc_filter_raw:
    for obj in x:
        zc_filter.append(complex(obj.replace('i','j')))

zc_filter = np.asarray(zc_filter,dtype=np.complex)

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.grid()
plt.title(r'Magnitude of Zadoff-Chu Filter')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([0,100])
plt.stem(np.abs(zc_filter),markerfmt='.')

plt.subplot(2,1,2)
plt.grid()
plt.title(r'Phase of Zadoff-Chu Filter')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([0,100])
plt.stem(np.angle(zc_filter),markerfmt='.')
plt.show()

y1 = np.fft.ifft(np.fft.fft(np.roll(zc_filter,5))*np.conj(np.fft.fft(zc_filter)))
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'Auto correlation of shifted Zadoff-Chu Filter')
plt.xlabel(r'$n\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.xlim([-5,100])
plt.stem(abs(y1),markerfmt='.')
plt.show()