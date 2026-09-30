import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def buildLaplacian(Nx,Ny):
    rows, cols, vals = [], [], []
    I = np.eye(Ny)
    # Build the first order laplacian
    for i in range(Nx):
        if i != 0:
            rows.append(i); cols.append(i-1); vals.append(1)
        if i != Nx-1:
            rows.append(i); cols.append(i+1); vals.append(1)
        rows.append(i); cols.append(i); vals.append(-2)
    L = sp.coo_array((vals, (rows, cols)), shape=(Nx,Nx))
    D2_x = sp.kron(L,I)
    D2_y = sp.kron(I,L)
    L = D2_y + D2_x
    return L

def buildDeriv(Nx,Ny, direction = 'y'):
    rows, cols, vals = [], [], []
    if direction == 'y':
        for i in range(Nx):
            if i != 0:
                rows.append(i); cols.append(i-1); vals.append(-1)
            if i < Ny-1:
                rows.append(i); cols.append(i+1); vals.append(1)
        D1 = sp.coo_array((vals, (rows, cols)), shape=(Nx,Ny))
        I = np.eye(Ny)
        D = sp.kron(D1,I)     
    if direction =='x':
        for i in range(Nx):
            if i != 0:
                rows.append(i); cols.append(i-1); vals.append(-1)
            if i < Ny-1:
                rows.append(i); cols.append(i+1); vals.append(1)
        D1 = sp.coo_array((vals, (rows, cols)), shape=(Nx,Ny))
        I = np.eye(Ny)
        D = sp.kron(I,D1)
    return D

def sysAssembly(mu, Nx, Ny, dx, verbose=False):
    # Calculate each block matrix
    L_u = mu*buildLaplacian(Nx-1,Ny-1)
    L_v = mu*buildLaplacian(Nx-1,Ny-2)
    G_x = buildDeriv(Nx-1, Ny-1, direction = 'x')
    G_y = buildDeriv(Nx-2,Ny-1, direction = 'y')
    D_x = G_x.T #buildDeriv(Nx-1, Ny-1, direction = 'x')
    D_y = G_y.T #buildDeriv(Nx-1, Ny-2, direction = 'y')

    # print(f"Lu: {L_u.shape}")
    # print(f"Lv: {L_v.shape}")
    # print(f"Gx: {G_x.shape}")
    # print(f"Gy: {G_y.shape}")
    # print(f"Dx: {D_x.shape}")
    # print(f"Dy: {D_y.shape}")

    grid = [[L_u, None, -dx*G_x],
            [None, L_v, -dx*G_y],
            [D_x, D_y, None]]

    A = sp.block_array(grid, format = 'csr')
    return A

def main():
    # Control Panel
    mu = 1.0 # Viscousity
    Nx = 10 # number of nodes in the type 1 grid in each direction
    x_0 = 0.0 # x left boundary
    x_L = 6.0 # x right boundary
    tp = 2*np.pi
    f = lambda x: (tp - 2*tp**2) * np.sin(tp*x[1]) * np.sin(tp*x[0])
    g = lambda x: np.cos(tp*x[0]) * np.cos(tp*x[1]) * (tp - 2*tp**2) + 2*tp**2 * np.cos(tp*x[0])

    # Control Panel (touch if you know what you need)
    Ny = Nx
    y_0, y_L = x_0, x_L # y left and right boundaries
    dx = (x_L-x_0)/Nx # distance between nodes on type 1 grid (and type two grid)

    A = sysAssembly(mu, Nx, Ny, dx)


    # visualize the model
    plt.spy(A)
    plt.show()

if __name__ == "__main__":
    main()