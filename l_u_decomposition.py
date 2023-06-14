import math
import numpy as np



'''
Decomposing a sqaure matrix into a lower traingular matrix and an upper triangular matrix

'''
def lu_decomposition(A:np.array):
    n = A.shape[0]
    
    l = np.eye(n)
    u = A.copy()
    
    for p in range(n-1):
        
        for r in range(p+1, n):
            l[r,p]+=u[r,p]*(l[p,p]/u[p,p])
            
            u[r,:]-=u[r,p]*(u[p,:]/u[p,p])
    return l,u 


def plu_decomposition(A:np.array):
    m,n = A.shape
    
    l = np.eye(m)
    
    u = A.copy()
    
    lim = m
    permutes = []
    
    for p in range(lim-1):
        i1 = p
        i2 = np.argmax(u[p:,p])+p
        
        if i1!=i2:
            temp = u[i1,:].copy()
            u[i1,:] = u[i2,:]
            u[i2,:] = temp
            permutes.append([i1,i2])
        
        if u[p,p]==0:
            return (None,None,None,None)
        
        for r in range(p+1, lim):
            l[r,p]+=u[r,p]*(l[p,p]/u[p,p])
            
            u[r,:]-=u[r,p]*(u[p,:]/u[p,p])
    
    # compute permutation matrix
    p_mat = np.eye(m)
    
    for i in permutes:
        i1 = i[0]
        i2 = i[1]
        
        temp = p_mat[i1,:].copy()
        
        p_mat[i1,:] = p_mat[i2,:]
        p_mat[i2,:] = temp
    
    return (p_mat, l, u, len(permutes))


