import numpy as np
import math


'''
Power Iteration is a numerical method widely used in numericalcomputations
In this case we will use power iteration to converge to the eigen vector
The basic intuition is that if we keep multiplying A to a random vector V
AV=> A^2=>....at infinity A(A'V) = l(A'V) limit at infinity, so (A'V) becomes our eigen vector
'''
def power_iteration(A, diff=0.0001):
    v = np.random.normal(loc=0, scale=1, size=(A.shape[0],1))
    prev = np.empty(shape=(A.shape[0],1))
    
    while True:
        prev[:,:] = v
        v = A@v
        v = v/np.linalg.norm(v)
        
        if np.allclose(v, prev, atol=diff):
            break
    return v
    

'''
The algorithm that is used in practice
It is an extended version of power iteration
To basically calculate all the eigen vectors, we leverage an interesting property of symmetric matrices having orthogonal eigen vectors
This simultaneous orthogonalisation using gram-schmidt helps us in finding all the eigen vectors in one go
'''
def simultaneous_orthogonalization(A, tol=0.0001):
    Q, R = np.linalg.qr(A) 
    previous = np.empty(shape=Q.shape)
    for i in range(100):
        previous[:] = Q
        X = A @ Q
        Q, R = np.linalg.qr(X)
        if np.allclose(Q, previous, atol=tol):
            break
    return


