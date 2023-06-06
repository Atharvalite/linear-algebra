import numpy as np
import threading
import time
import matplotlib.pyplot as plt

'''
naive implementation of vector-matrix multiplication

weight matrix shape: (n,n)
vector shape: (n,1)
'''
global y

def naive_vector_matrix_mult(w, x):
    n = w.shape[0]
    # y = np.zeros((n,1))
    for i in range(n):
        for j in range(n):
            y[i]+=w[i,j]*x[j]
        # y[i] = np.dot(w[i,:], x)
    return


def vector_matrix_thread(row, x, ind):
    y[ind] = np.dot(row, x)
    return
    
def singledim_partition(w, x, num_threads=16):
    n = w.shape[0]

    
    threads = []
    #8 cores, hyperthreading doubles to 16
    # we will broadcast to make 16 copies of x
    xb = np.broadcast_to(x.T, shape=(num_threads,n))
    
    vec_ind = 0
    thread_ind = 0
    
    while(vec_ind<n):
        if vec_ind<num_threads:
            z = threading.Thread(target=vector_matrix_thread, args=(w[vec_ind,:], xb[vec_ind,:], vec_ind))
            threads.append(z)
            z.start()
            vec_ind+=1
        else:
            threads[0].join()
            threads.pop(0)
            
            z = threading.Thread(target=vector_matrix_thread, args=(w[vec_ind,:], xb[thread_ind,:], vec_ind))
            threads.append(z)
            z.start()
            
            vec_ind+=1
            thread_ind+=1
            thread_ind%=num_threads
    for i in threads:
        i.join()
    
    return


    
    