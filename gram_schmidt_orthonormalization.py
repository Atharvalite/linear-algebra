import numpy as np
import math


def qr_compute(A):
    n = A.shape[1]
    
    q = np.empty(shape=A.shape)
    
    for i in range(n):
        vec = A[:,i]
        components = vec.copy()
        for j in range(0,i):
            d = np.dot(vec, q[:,j])/np.linalg.norm(q[:,j])
            d = d*q[:,j]
            components-=d 
        q[:,i] = components/np.linalg.norm(components) # normalize to unit vector
    
    r = q.T@A
    return (q,r)

    
    
    
    
            
    