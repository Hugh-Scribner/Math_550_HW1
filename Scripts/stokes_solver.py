import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def buildLaplacian(Nx,Ny, sparse = True):
    N = Nx*Ny
    if sparse:
        L = sp.coo_matrix((N,N))
    else:
        L = np.zeros((N,N))
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
        D = sp.coo_matrix((N,N))
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
    mu = 1.0 # Viscousity
    Nx = 101 # number of nodes in the type 1 grid in each direction
    x_0 = 0.0 # x left boundary
    x_L = 6.0 # x right boundary

    # Control Panel (touch if you know what you need)
    y_0, y_L = x_0, x_L # y left and right boundaries
    dx = (x_L-x_0)/Nx # distance between nodes on type 1 grid (and type two grid)

    # Calculate each block matrix
    L_u = mu*buildLaplacian(Nx-1,Nx-1)
    L_v = mu*buildLaplacian(Nx-2,Nx-1)
    G_x = -dx*buildDeriv(Nx, direction = 'x')
    G_y = -dx*buildDeriv(Nx)
    D_x = buildDeriv(Nx, direction = 'x')
    D_y = buildDeriv(Nx)

    grid = [[L_u, None, G_x],
            [None. L_v, G_y],
            [D_x, D_y, None]]

    A = sp.block_array(grid, format = 'csr')

if __name__ == "__main__":
    main()