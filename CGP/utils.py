import numpy as np
from GP.algo import GP_PNN_gen


def get_gauss_noise(d=2):
    Z = np.random.normal(0,1,size=d)
    return Z


def CGP_PNN_gen(X,rho,func,args=None,gamm_fact=0.0,K=1.0):
    eps = np.sqrt(2*rho)
    return GP_PNN_gen(X, eps, func=func, args=args, gamm_fact=gamm_fact, K=K)


def norm_func(x, i, args):
    n = args[0]
    p = args[1]
    i_n = i%n
    g_i = np.linalg.norm(x[i_n]-p)
    return g_i
