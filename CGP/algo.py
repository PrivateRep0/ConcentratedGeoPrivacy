import numpy as np
from CGP.utils import get_gauss_noise, CGP_PNN_gen, norm_func


def CGP_Loc(z,rho):
    noise = get_gauss_noise()
    z_p = z + 1/np.sqrt(2*rho)*noise
    return z_p

def CGP_Center(x, rho):
    c0 = 0.5*(np.max(x[:,0])+np.min(x[:,0]))
    c1 = 0.5*(np.max(x[:,1])+np.min(x[:,1]))
    noise = get_gauss_noise()
    c_tilde = np.array([c0,c1])+1./np.sqrt(rho)*noise
    return c_tilde

def CGP_Radius(x, c_tilde, rho, beta=0.1):
    noise = get_gauss_noise(d=1)[0]
    R_max = np.max(np.linalg.norm(x-c_tilde,axis=1))
    R_tilde = R_max + 1./np.sqrt(2.*rho)*noise + np.sqrt(np.log(1/beta)/rho)
    return R_tilde

def CGPBasic(x,rho):
    n = len(x)
    y = np.zeros_like(x)
    for i in range(n):
        y[i] = CGP_Loc(x[i],rho/n)
    return y

def CGP_kPNN(x, p, rho, k, func=norm_func, gamm_fact=0.0):
    n = len(x)
    I = [i for i in range(n)]
    J = []
    rho_k = rho/k
    for j in range(k):
        xI = x[I]
        t = CGP_PNN_gen(xI, rho_k, func=func, args=[p], gamm_fact=gamm_fact)
        tj = I[t]
        J.append(tj)
        del I[t]
    return J

def CGP_PCH_point_adaptk(x, rho, beta=0.1, gamm_fact=0.0, kl=16, kr=128, alpha=0.0):
    n = len(x)
    rho_0 = rho*0.05
    rho_1 = rho-rho_0
    c_tilde = CGP_Center(x, 2.0*rho_0/3.0)
    R_tilde = CGP_Radius(x, c_tilde, rho_0/3.0, beta=0.5*beta)
    R_tilde = max(R_tilde,10)
    c11 = (15.0*(np.log(2/beta)+np.log(4*n+2))+3.0*np.sqrt(2.0*np.log(2/beta)+2.0*np.log(4*n+2)))/np.sqrt(2.0)
    c11 = c11**(2.0/3.0)
    c22 = (np.sqrt(np.log(2/beta)))**(2.0/3.0)
    
    if alpha > 0.0:
        aa = alpha
    else:
        aa = c11/(c11+c22)
    c_1 = c11/np.sqrt(aa)+c22/np.sqrt(1-aa)
    k = int((4.0*np.pi*R_tilde*np.sqrt(rho_1)/c_1)**(2.0/3.0))
    k = min(max(k,kl),kr)

    rho_j = aa*rho_1/k
    A = []
    I = [i for i in range(n)]
    for j in range(k):
        theta_j = 2*np.pi*(j)/k
        Pj = c_tilde+np.array([R_tilde*np.cos(theta_j),R_tilde*np.sin(theta_j)])
        t = CGP_PNN_gen(x[I],rho_j,func=norm_func,args=[Pj],gamm_fact=gamm_fact)
        aj = I[t]
        A.append(aj)
        del I[t]
    x_tilde = CGPBasic(x[A],(1.0-aa)*rho_1)
    return x_tilde