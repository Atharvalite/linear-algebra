import math
import numpy as np
from l_u_decomposition import plu_decomposition


def naive_determinant(A:np.array):
    n = A.shape[0]
    
    if n==1:
        return A[0,0]
    elif n==2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    else:
        ans = 0
        m = -1
        for i in range(n):
            m*=-1
            if A[0,i]==0:
                continue
            else:
                ans+=A[0,i]*m*naive_determinant(np.concatenate((A[1:,:i], A[1:,i+1:]), axis=1))
            
        return ans


def determinant(A:np.array):
    ans = 0
    p,l,u,n = plu_decomposition(A)
    if(p is None):
        return 0
    
    if n%2==0:
        m = 1
    else:
        m=-1
    
    ans = m*np.prod(np.diagonal(l))*np.prod(np.diagonal(u))

    return ans

            
    
        

