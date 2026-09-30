import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def buildLaplacian(Nx, sparse = True):
    N = Nx*(Nx-1)
    if sparse:
        D_1 = sp.coo((N,N))
    else:
        D = np.zeros((N,N))
    I = np.eye(N)
    # Build the first order laplacian
    for i in range(N):
        if i != 0:
            L[i,i-1] = 1
        if i != N-1:
            L[i,i+1] = 1
        L[i,i] = -2
    D2_x = sp.kron(L,I)
    D2_y = sp.kron(I,L)
    L = D2_y + D2_x
    return L

def buildDeriv(Nx, sparse = True, direction = 'y'):
    N = Nx*(Nx-1)
    if sparse:
        D = sp.coo((N,N))
    else:
        D = np.zeros((N,N))
    for i in range(N):
        if i != 0:
            D[i,i-1] = -1
        if i != N-1:
            D[i, i+1] = 1
    I = np.eye(Nx)
    if direction == 'y':
        D = sp.kron(I,D)     
    if direction =='x':
        D = sp.kron(D,I)
    return D

def main():
    # Control Panel
    mu = 1
    Nx = 101
    x_0 = 0
    x_L = 6

    # Control Panel (touch if you know what you need)
    Ny = Nx
    y_0, y_L = x_0, x_L

    # Calculations for 
    D = buildDeriv(3)
    plt.spy(D)
    plt.show()

if __name__ == "__main__":
    main()