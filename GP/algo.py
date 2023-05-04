import numpy as np
from GP.utils import get_lap_noise, get_laplacian, SVT, norm_func


def GP_Loc(z, eps):
    (r, theta) = get_laplacian(eps)
    z_1 = r*np.cos(theta)
    z_2 = r*np.sin(theta)
    return [(z[0]+z_1),(z[1]+z_2)]

def GP_Center(x, eps):
    c0 = 0.5*(np.max(x[:,0])+np.min(x[:,0]))
    c1 = 0.5*(np.max(x[:,1])+np.min(x[:,1]))
    c_tilde = GP_Loc(np.array([c0,c1]), eps/np.sqrt(2.0))
    return c_tilde

def GP_Radius(x, c_tilde, eps, beta=0.1):
    noise = get_lap_noise()[0]
    R_max = np.max(np.linalg.norm(x-c_tilde,axis=1))
    R_tilde = R_max + 1./eps*noise + np.log(1/beta)/eps
    return R_tilde

def GPBasic(x, eps):
    n = len(x)
    eps_i = eps/n
    y = np.zeros_like(x)
    for i in range(n):
        y[i] = GP_Loc(x[i],eps_i)
    return y

def GP_PNN_gen(x, eps, func=norm_func, args=[], gamm_fact=0.0, K=1.0):
    n = len(x)
    args_ext = [n]
    args_ext.extend(args)
    gamma0 = gamm_fact*3.0*K/eps
    hx = np.min([func(x,i,args_ext) for i in range(n)])
    noise = get_lap_noise(d=1)[0]
    T= hx+gamma0+3.0*K/eps*noise
    t = SVT(x,2.0*eps/3.0,T,func=func,args=args_ext,K=K)
    return t%n

def GP_kPNN(x, p, eps, k, func=norm_func, gamm_fact=0.0):
    n = len(x)
    I = [i for i in range(n)]
    J = []
    eps_k = eps/k
    for j in range(k):
        xI = x[I]
        t = GP_PNN_gen(xI, eps_k, func=func, args=[p], gamm_fact=gamm_fact)
        tj = I[t]
        J.append(tj)
        del I[t]
    return J

def GP_PCH_point_adaptk(x, eps, beta=0.1, gamm_fact=0.0, kl=16, kr=128, alpha=0.0):
    eps_0 = eps*0.05
    eps_1 = eps-eps_0
    n = len(x)
    c_tilde = GP_Center(x, 2.0*eps_0/3.0)
    R_tilde = GP_Radius(x, c_tilde, eps_0/3.0, beta=0.5*beta)
    R_tilde = max(R_tilde,10)
    c11 = 15.0*(np.log(2/beta)+np.log(4*n+2))+3.0*np.sqrt(2.0*np.log(2/beta)+2.0*np.log(4*n+2))
    c22 = np.sqrt(2.0*np.log(2/beta))+np.log(2/beta)

    if alpha > 0.0:
        aa = alpha
    else:
        aa = c11/(c22+c11)
    c_1 = c11/aa+c22/(1-aa)
    k = int(np.sqrt(2.0*eps_1*np.pi*R_tilde/c_1))
    k = min(max(k,kl),kr)

    eps_j = aa*eps_1/k
    A = []
    I = [i for i in range(n)]
    for j in range(k):
        theta_j = 2*np.pi*(j)/k
        Pj = c_tilde+np.array([R_tilde*np.cos(theta_j),R_tilde*np.sin(theta_j)])
        t = GP_PNN_gen(x[I],eps_j,func=norm_func,args=[Pj],gamm_fact=gamm_fact)
        aj = I[t]
        A.append(aj)
        del I[t]
    x_tilde = GPBasic(x[A],(1.0-aa)*eps_1)
    return x_tilde
