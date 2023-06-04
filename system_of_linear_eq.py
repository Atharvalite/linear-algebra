import numpy as np
'''
Reduced row echelon form is a transformation of a matrix such that it satisfies these conditions:
1. All rows that contain only zeros are at the bottom of the matrix; corre-spondingly, all rows that contain at least one nonzero element are on
top of rows that contain only zeros.
2. The first non-zero number in each row from the left(pivot variable) is strictly right to the previous pivot
3. For reduced ref, every pivot elment should be equal to 1, and
4. The pivot is the only non-zero entry in the column
'''

def rref(A, b):

    rows, cols = A.shape
    print(f"This matrix has {rows} Rows and {cols} Columns")

    pivot_cols = []

    r=0
    count=0
    zero_count = 0
    while(count!=rows):
        eq = A[r,:]

        ind = (eq!=0).argmax()
        
        if(ind==0 and eq[0]==0):
            A[[r,rows-1-zero_count]] = A[[rows-1-zero_count, r]]
            zero_count+=1
            count+=1
            
        else:
            pivot_cols.append(ind)

            # making pivot element 1
            eq/=eq[ind]
            b[r]/=eq[ind]

            b_val = b[r]

            # making entire row as zero
            for k in range(rows):
                if k!=r:
                    b[k]-=A[k,ind]*b_val
                    A[k,:]+= (-1)*(A[k,ind])*(eq)
            A[r,:] = eq
            r+=1
            count+=1
            
    return (A,b, pivot_cols)


'''
This function takes in reduced row echelon form as an input, and outputs the solution set
of the given system of linear equations
'''
def compute_soln_set(A, b, pivot_cols):
    row, cols = A.shape
    free_num = cols-len(pivot_cols)
    
    soln = np.zeros((cols, free_num+1))
    
    # compute free var map
    free_cols = [i for i in range(cols) if i not in pivot_cols]
        
    # fill for pivot variables
    r=0
    for i in range(len(pivot_cols)):
        c = pivot_cols[i]
        
        b_val = b[r]
        soln[c,0] = b_val
        for k in range(c+1,cols):
            if A[r,k]!=0:
                s_ind = free_cols.index(k)+1
                soln[c,s_ind] = (-1)*A[r,k]
        r+=1
    
    # fill free variables too
    for i in range(len(free_cols)):
        r = free_cols[i]
        
        soln[r,i+1] = 1
    
    return soln



