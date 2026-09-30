import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import matplotlib.pyplot as plt

def fd2_periodic(Nx):
    rows, cols, vals = [], [], []
    for i in range(Nx):
            if i == 0:
                rows.append(i); cols.append(Nx-1); vals.append(1)
            if i != 0:
                rows.append(i); cols.append(i-1); vals.append(1)
            if i != Nx-1:
                rows.append(i); cols.append(i+1); vals.append(1)
            if i == Nx-1
                rows.append(i); cols.append(0); vals.append(1)
            rows.append(i); cols.append(i); vals.append(-2)
    L = sp.coo_array((vals, (rows, cols)), shape=(Nx,Nx))
    return L

def fd2_neumann(Nx):
    rows, cols, vals = [], [], []
    for i in range(Nx):
        if i == 0:
            rows.append(i); cols.append(i+1); vals.append(2)
            rows.append(i); cols.append(i); vals.append(-2)
        if i != 0 or i != Nx-1:
            rows.append(i); cols.append(i-1); vals.append(1)
            rows.append(i); cols.append(i); vals.append(-2)
            rows.append(i); cols.append(i+1); vals.append(1)
        if i == Nx-1:
            rows.append(i); cols.append(i); vals.append(2)
            rows.append(i); cols.append(i-1); vals.append(-2)
    L = sp.coo_array((vals, (rows, cols)), shape=(Nx,Nx))
    return L

def fd2_dirichlet(Nx):
    rows, cols, vals = [], [], []
    for i in range(Nx):
        if i == 0 or i == Nx-1:
            rows.append(i); cols.append(i); vals.append(1)
        else:
            rows.append(i); cols.append(i-1); vals.append(1)
            rows.append(i); cols.append(i+1); vals.append(1)
            rows.append(i); cols.append(i); vals.append(-2)
    L = sp.coo_array((vals, (rows, cols)), shape=(Nx,Nx))
    return L

def buildLaplacian(Nx,Ny, BCx, BCy):
    rows, cols, vals = [], [], []
    I = np.eye(Ny)
    # Build the first order laplacian
    match BCx:
        case 0: # Dirichlet
            Lx = fd2_dirichlet(Nx)
        case 1: # Neumann
            Lx = fd2_neumann(Nx)
        case 2: # periodic
            Lx = fd2_periodic(Nx)
    match BCy:
        case 0: # Dirichlet
            Ly = fd2_dirichlet(Nx)
        case 1: # Neumann
            Ly = fd2_neumann(Nx)
        case 2: # periodic
            Ly = fd2_periodic(Nx)
    L = sp.coo_array((vals, (rows, cols)), shape=(Nx,Nx))
    D2_x = sp.kron(Lx,I)
    D2_y = sp.kron(I,Ly)
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

def sysAssembly(mu, Nx, Ny, dx, x, y, f, g, verbose=False):
    # Calculate each block matrix and Build A
    L_u = mu*buildLaplacian(Nx-1,Ny-1)
    L_v = mu*buildLaplacian(Nx-1,Ny-2)
    G_x = buildDeriv(Nx-1, Ny-1, direction = 'x')
    G_y = buildDeriv(Nx-2,Ny-1, direction = 'y')
    D_x = G_x.T #buildDeriv(Nx-1, Ny-1, direction = 'x')
    D_y = G_y.T #buildDeriv(Nx-1, Ny-2, direction = 'y')

    print(f"Lu: {L_u.shape}")
    print(f"Lv: {L_v.shape}")
    print(f"Gx: {G_x.shape}")
    print(f"Gy: {G_y.shape}")
    print(f"Dx: {D_x.shape}")
    print(f"Dy: {D_y.shape}")

    A_grid = [[L_u, None, -dx*G_x],
            [None, L_v, -dx*G_y],
            [D_x, D_y, None]]

    A = sp.block_array(A_grid, format = 'coo')

    # Build out b using a meshgrid
    xx, yy = np.meshgrid(np.linspace(x[0], x[1]-dx, Ny-1, endpoint = True), np.linspace(y[0], y[1]-dx, Nx-1, endpoint = True), indexing ='xy')
    xxy = np.column_stack((xx.flatten(), yy.flatten() + dx/2))
    F = f(xxy) # Forcing in X
    xx, yy = np.meshgrid(np.linspace(x[0], x[1]-dx, Ny-2, endpoint = True), np.linspace(y[0], y[1]-dx, Nx-1, endpoint = True), indexing ='xy')
    xyy = np.column_stack((xx.flatten(), yy.flatten() + dx/2))
    G = g(xyy) # Forcing in Y
    O = np.zeros(((Nx-1) * ( Ny-1),))

    b = np.hstack((F,G,O))

    # Apply Boundary conditions
    # Top Boundary
    for i in range(Nx):
        mask = (A.row == i) & (A.col == i)
        A


    return A, b

def main():
    # Control Panel
    mu = 1.0 # Viscousity
    Nx = 10 # number of nodes in the type 1 grid in each direction
    x_0 = 0.0 # x left boundary
    x_L = 6.0 # x right boundary
    tp = 2*np.pi
    f = lambda x: (tp - 2*tp**2) * np.sin(tp*x[:,1]) * np.sin(tp*x[:,0])
    g = lambda x: np.cos(tp*x[:,0]) * np.cos(tp*x[:,1]) * (tp - 2*tp**2) + 2*tp**2 * np.cos(tp*x[:,0])

    # U horizontal BCs
    U_y_0 = 0
    U_y_L = 0

    # U vertical BCs, periodic

    # V horizontal BCs,
    V_y_0 = -3.5
    V_y_L = -3.5

    # V vertical BCs, periodic

    # Control Panel (touch if you know what you need)
    Ny = Nx
    y_0, y_L = x_0, x_L # y left and right boundaries
    x, y = (x_0, x_L), (y_0, y_L)
    dx = (x_L-x_0)/Nx # distance between nodes on type 1 grid (and type two grid)

    # inner linear system
    A, b = sysAssembly(mu, Nx, Ny, dx, x, y, f, g, V_y_0, U_y_0)

    # Solve
    #x_direct = spa.spsolve(A, b)
    #x_iter = spa.gmres(A, b)

    # visualize the model
    #plt.spy(A)
    #plt.show()
    return 0

if __name__ == "__main__":
    main()