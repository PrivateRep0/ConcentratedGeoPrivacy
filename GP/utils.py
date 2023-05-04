import numpy as np
from scipy.special import lambertw


def get_lap_noise(d=1):
    Z = np.random.laplace(0,1,size=d)
    return Z


def get_laplacian(eps):
    theta = np.random.uniform(0,2*np.pi)
    u = np.random.uniform(0,1)
    w = lambertw((u-1)*np.exp(-1),k=-1)
    assert(w.imag==0)
    r = -(w.real+1)/eps
    return (r, theta)


def SVT(X,eps,T,func,b_inf=True,args=None,K=1.0):
    T_tilde = T + np.random.laplace(scale=2.0*K/eps)
    i = 0
    while True:
        if not(b_inf) and i >= args[0]:
            break
        Qi = func(X,i,args) + np.random.laplace(scale=4.0*K/eps)
        if Qi <= T_tilde:
            break
        i = i + 1
    return i


def norm_func(x, i, args):
    n = args[0]
    p = args[1]
    i_n = i%n
    g_i = np.linalg.norm(x[i_n]-p)
    return g_i



