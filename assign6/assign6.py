import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

p = np.polymul([1,1,2.5],[1,0,2.25])
Y = sp.lti([1,0.5],p)
H = sp.lti([1],[1,0,2.25])
t,x = sp.impulse(H,None,np.linspace(0,50,1000))

plt.figure(figsize=(8,8))
plt.grid()
plt.plot(t,x)
plt.title('Impulse response of the system')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.show()

w,s,phi = Y.bode()
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.semilogx(w,s)
plt.grid()
plt.title('Magnitude response of the system for Q1')
plt.ylabel(r'$|Y(j\omega)|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.grid()
plt.title('Phase response of the system for Q1')
plt.ylabel(r'$\angle Y(j\omega)\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.show()

t = np.linspace(0,50,1e4)
t,y = sp.impulse(Y,None,np.linspace(0,50,1000))
plt.figure(figsize=(8,8))
plt.plot(t,y)
plt.grid()
plt.title('Response of the system corresponding to the input')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.show()

##########################  Question 2 ################################

p = np.polymul([1,0.1,2.2525],[1,0,2.25])
Y1 = sp.lti([1,0.05],p)
H1 = sp.lti([1],[1,0,2.25])
t,x = sp.impulse(H1,None,np.linspace(0,50,1000))

w,s,phi = Y1.bode()
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.semilogx(w,s)
plt.grid()
plt.title('Magnitude response of the system for Q2')
plt.ylabel(r'$|Y(j\omega)|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.grid()
plt.title('Phase response of the system for Q2')
plt.ylabel(r'$\angle Y(j\omega)\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.show()

t = np.linspace(0,50,1e4)
t,y = sp.impulse(Y1,None,np.linspace(0,50,1000))
plt.figure(figsize=(8,8))
plt.plot(t,y)
plt.grid()
plt.title('Response of the system corresponding to the input')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.show()

##########################  Question 3 ################################

t = np.linspace(0,100,2e4)
freq = np.linspace(1.4,1.6,5)
for f in freq:
    u = np.cos(f*t)*np.exp(-0.05*t)*(t>0)
    t,y,_ = sp.lsim(H1,u,t)
    plt.figure(figsize=(8,8))
    plt.plot(t,y)
    plt.grid()
    plt.title('Response of the system corresponding to the input at f = {}'.format(f))
    plt.ylabel(r'$y\rightarrow$')
    plt.xlabel(r'$t\rightarrow$')
    plt.show()

##########################  Question 4 ################################

H = sp.lti([2],[1,0,3,0])
w,s,phi = H.bode()
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.semilogx(w,s)
plt.grid()
plt.title('Magnitude response of Y for Q4')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.grid()
plt.title('Phase response of Y for Q4 ')
plt.ylabel(r'$\angle H(j\omega)\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.show()

H1 = sp.lti([1,0,2],[1,0,3,0])
w,s,phi = H1.bode()
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.semilogx(w,s)
plt.grid()
plt.title('Magnitude response of X for Q4')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.grid()
plt.title('Phase response of X for Q4 ')
plt.ylabel(r'$\angle H(j\omega)\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.show()


t,y = sp.impulse(H,None,np.linspace(0,20,1000))
t,x = sp.impulse(H1,None,np.linspace(0,20,1000))
plt.figure(figsize=(8,8))
plt.grid()
plt.plot(t,y)
plt.plot(t,x)
plt.title('Solutions of the ODE')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$x,y\rightarrow$')
plt.legend(['y','x'])
plt.show()

##########################  Question 5 ################################

Y1 = sp.lti([1],[10**-12,10**-4,1])
w,s,phi = Y1.bode()
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.semilogx(w,s)
plt.grid()
plt.title('Magnitude response of the system for Q5')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.subplot(212)
plt.semilogx(w,phi)
plt.grid()
plt.title('Phase response of the system for Q5')
plt.ylabel(r'$\angle H(j\omega)\rightarrow$')
plt.xlabel(r'$\omega\rightarrow$')
plt.show()

##########################  Question 6 ################################

H1 = sp.lti([1],[10**-12,10**-4,1])

t = np.linspace(0,30*(10**-6),1000)
u = (np.cos(1000*t)-np.cos((10**6)*t))*(t>0)
t,y,_ = sp.lsim(H1,u,t)

plt.figure(figsize=(8,8))
plt.plot(t,y)
plt.grid()
plt.title(r'Response of the system corresponding to the input upto $30\mu s$')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.xticks(rotation=45)
plt.show()

t = np.linspace(0,10*(10**-3),1e5)
u = (np.cos(1000*t)-np.cos((10**6)*t))*(t>0)
t,y,_ = sp.lsim(H1,u,t)

plt.figure(figsize=(8,8))
plt.plot(t,y)
plt.grid()
plt.title('Response of the system corresponding to the input upto 10ms')
plt.ylabel(r'$y\rightarrow$')
plt.xlabel(r'$t\rightarrow$')
plt.xticks(rotation=45)
plt.show()


