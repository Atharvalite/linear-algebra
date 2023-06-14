import numpy as np
import math
from matrix_determinant import determinant


def inverse_gauss_jordan(A:np.array):
    n = A.shape[0]
    
    I = np.eye(n)
    for i in range(n):
        p = A[i,i]
        
        if p==0:
            print('Singular Matrix')
            return None
        
        A[i,:]/=p
        I[i,:]/=p
        row = A[i,:]
        
        
        for j in range(n):
            if i!=j:
                
                I[j,:]-=A[j,i]*I[i,:]
                A[j,:]-=A[j,i]*row
                
    return I


def inverse_laplace_expansion(A:np.array):
    n = A.shape[0]
    
    det = determinant(A)
    print(det)
    
    if det==0:
        print('Singular Matrix')
        return None
    inv = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            up = np.concatenate([A[:i,:j], A[:i, j+1:]], axis=1)
            down = np.concatenate([A[i+1:, :j], A[i+1:, j+1:]], axis=1)
            mat = np.concatenate([up, down], axis=0)
            
            m = (-1)**(i+j)
            inv[i,j] = m*determinant(mat)
    
    inv/=det
    return inv.T


