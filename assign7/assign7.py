import numpy as np
import sympy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt

################### Question 1 ########################

s = sp.symbols('s')
def LPF(R1,R2,C1,C2,G,Vi):
    A = sp.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0], \
        [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sp.Matrix([0,0,0,Vi/R1])
    V = A.inv()*b
    return (A,b,V)

A,b,V = LPF(10000,10000,1e-9,1e-9,1.586,1)
A1,b1,V1 = LPF(10000,10000,1e-9,1e-9,1.586,1/s)
Vo = V[3]
Vp = V1[3]
w = np.logspace(-1,8,100)
ss = 1j*w
hf = sp.lambdify(s,Vo,'numpy')
hf1 = sp.lambdify(s,Vp,'numpy')
v = hf(ss)
v1 = hf1(ss)

plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Impulse\;and\;step\;response\;of\;LPF$')
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.loglog(w,abs(v))
plt.loglog(w,abs(v1))
plt.legend(['impulse response','step impulse'])
plt.show()

H = sig.lti([-0.0001586],[2e-14,4.414e-9,0.0002])
t = np.linspace(0,0.001,2e5)
u = (t>0)
t,y,_ = sig.lsim(H,u,t)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Input\;and\;output\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u)
plt.plot(t,y)
plt.legend(['input','output'])
plt.show()

################### Question 2 ########################

t = np.linspace(0,0.001,2e5)
u1 = np.sin(2000*np.pi*t) # t = 1ms
u2 = np.cos(200000*np.pi*t) # t = 1us
Vo = sp.simplify(Vo)
n,d = sp.fraction(Vo)
print(n,d)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Components\;of\;the\;input\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u1)
plt.plot(t,u2)
plt.legend(['low frequency input','high frequency input'])
plt.show()

H = sig.lti([-0.0001586],[2e-14,4.414e-9,0.0002])
t,y,_ = sig.lsim(H,u1+u2,t)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Output\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.show()


################### Question 3 ########################

def HPF(R1,R2,C1,C2,G,Vi):
    A = sp.Matrix([[0,0,1,-1/G],[-1/(1+1/(s*R2*C2)),1,0,0], \
        [0,-G,G,1],[0-s*C1-s*C2-1/R1,s*C2,0,1/R1]])
    b = sp.Matrix([0,0,0,Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)

A,b,V = HPF(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]
Vo = sp.simplify(Vo)
n,d = sp.fraction(Vo)
print(n,d)
w = np.logspace(-1,8,1000)
ss = 1j*w
hf = sp.lambdify(s,Vo,'numpy')
v = hf(ss)

plt.figure(figsize=(16,8))
plt.semilogx(w,abs(v))
plt.grid()
plt.title(r'$Impulse\;response\;of\;HPF$')
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.show()

################### Question 4 ########################

t = np.linspace(0,0.0001,2e5)
u1 = np.sin(2000*np.pi*t)*np.exp(-50000*t)
u2 = np.sin(2000000*np.pi*t)*np.exp(-50000*t)

plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Components\;of\;the\;input\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u1)
plt.plot(t,u2)  
plt.legend(['low frequency input','high frequency input'])
plt.show()

H = sig.lti([-1.586e-9,0,0],[2e-9,4.414e-4,20.0])

t,y,_ = sig.lsim(H,u1+u2,t)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Output\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,y)
plt.show()

################### Question 5 ########################

H = sig.lti([-1.586e-9,0,0],[2e-9,4.414e-4,20.0])
u = (t>0)
t,y,_ = sig.lsim(H,u,t)
plt.figure(figsize=(16,8))
plt.grid()
plt.title(r'$Input\;and\;output\;to\;the\;filter$')
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u)
plt.plot(t,y)
plt.legend(['input','output'])
plt.show()
