import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import matplotlib.pyplot as plt

def buildLaplacian(Nr, Nc, u_mat = False):
    row, col, val = [], [], []
    # Build the basic Laplacian
    for i in range(Nr):
        for j in range(Nc):
            if u_mat and ( i == 0 or i==Nr-1): # apply dirichlet BC for U
                row.append(Nc*i+j); col.append(Nc*i+j); val.append(-5.0)
            else:    
                row.append(Nc*i+j); col.append(Nc*i+j); val.append(-4.0)
            row.append(Nc*i+j); col.append(Nc*i+(j-1)%Nc); val.append(1.0) # loop back for periodicity
            row.append(Nc*i+j); col.append(Nc*i+(j+1)%Nc); val.append(1.0) # loop back for periodicity
            if i > 0: # Dirichlet BCs for bottom
                row.append(Nc*i+j); col.append(Nc*i+j-Nc); val.append(1.0)
            if i < Nr-1: # Dirichlet BCs for top
                row.append(Nc*i+j); col.append(Nc*i+j+Nc); val.append(1.0)
    # Apply BCs to rows
    A = sp.coo_array((val, (row, col)), shape=(Nr*Nc, Nr*Nc))
    return A

def buildDeriv(Nr,Nc, dir = 'x'):
    row, col, val = [], [], []
    # Build the basic Laplacian
    if dir =='x':
        for i in range(Nr):
            for j in range(Nc):
                row.append(Nc*i+j); col.append(Nc*i+j % Nc); val.append(-1.0) # loop back for periodicity
                row.append(Nc*i+j); col.append(Nc*i+(j+1) % Nc); val.append(1.0) # loop back for periodicity
        A = sp.coo_array((val, (row, col)), shape=(Nr*Nr, Nr*Nc))
    if dir =='y':
        for i in range(Nr):
            for j in range(Nc):
                if i > 0:
                    row.append(Nc*i+j); col.append(Nc*i+j); val.append(-1.0)
                if i < Nr-1:
                    row.append(Nc*i+j); col.append(Nc*i+(j+Nc)); val.append(1.0)
        if Nc > Nr:
            A = sp.coo_array((val, (row, col)), shape=(Nr*Nc, Nc*Nc))
        else:
            A = sp.coo_array((val, (row, col)), shape=(Nr*Nr, Nr*Nc))
    return A

def sysAssembly(mu, Nc, Nr, dx, x, y, f, g, U_top, V_top, U_bot, V_bot, verbose=False):
    # Calculate each block matrix and Build A
    L_u = mu*buildLaplacian(Nr, Nc, u_mat = True)
    L_v = mu*buildLaplacian(Nr-1,Nc)
    G_x = buildDeriv(Nr, Nc, dir = 'x')*dx
    G_y = buildDeriv(Nr-1,Nc, dir = 'y')*dx
    D_x = buildDeriv(Nr, Nc, dir = 'x') #buildDeriv(Nx-1, Ny-1, direction = 'x')
    D_y = buildDeriv(Nr,Nc-1, dir = 'y') #buildDeriv(Nx-1, Ny-2, direction = 'y')

    # print(f"Lu: {L_u.shape}")
    # print(f"Lv: {L_v.shape}")
    # print(f"Gx: {G_x.shape}")
    # print(f"Gy: {G_y.shape}")
    # print(f"Dx: {D_x.shape}")
    # print(f"Dy: {D_y.shape}")

    A_grid = [[L_u, None, G_x],
            [None, L_v, G_y],
            [D_x, D_y, None]]

    A = sp.block_array(A_grid, format = 'coo')

    # Build out b using a meshgrid
    x_u = np.linspace(x[0], x[1], Nc)
    y_u = np.linspace(y[0] , y[1], Nr) + dx/2
    xx, yy = np.meshgrid(x_u, y_u, indexing ='ij')
    F = f(xx, yy)*dx**2 # Forcing in X
    x_v = np.linspace(x[0], x[1], Nc) + dx/2
    y_v = np.linspace(y[0] + dx , y[1], Nr-1)
    xx, yy = np.meshgrid(x_v, y_v, indexing ='ij')
    G = g(xx, yy)*dx**2 # Forcing in Y
    O = np.zeros(((Nc) * ( Nr),))

    # apply BCs to forcing matricies
    F[0,:] -= U_top
    G[0,:] -= V_top
    F[-1,:] -= U_bot
    G[-1,:] -= V_bot
    b = np.hstack((F.flatten(),G.flatten(),O))

    # print(dx)
    # print(F.shape)
    # print(G.shape)
    # print(O.shape)
    # print(b.shape)
    # print(A.shape)
        
    return A, b

def StokesSolver(mu, Nx, Ny, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L):
       # inner linear system
    Nr = Ny - 1
    Nc = Nx - 1
    A, b = sysAssembly(mu, Nc, Nr, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L)
    A = A.tocsr()
    # Solve
    #x_direct = spa.spsolve(A, b)
    x, exit_code = spa.gmres(A, b)
    U = x[0:Nc*Nr]
    V = x[Nc*Nr:(Nc*Nr+(Nr-1)*(Nc-1))]
    P = x[(Nc*Nr+(Nr-1)*(Nc-1)):-1]
    print(x.shape)
    print(U.shape)
    print(V.shape)
    print(P.shape)

    # P = P - np.mean(P)

    # visualize the model
    plt.spy(A)
    plt.show()
    return U, V, P

def main():
    # Control Panel
    mu = 1.0 # Viscousity
    Nx = 5 # number of nodes in the type 1 grid in each direction
    x_0 = 0.0 # x left boundary
    x_L = 6.0 # x right boundary
    tp = 2*np.pi
    f = lambda x, y: (tp - 2*tp**2) * np.sin(tp*x) * np.sin(tp*y)
    g = lambda x, y: np.cos(tp*x) * np.cos(tp*y) * (tp - 2*tp**2) + 2*tp**2 * np.cos(tp*x)

    # U horizontal BCs
    U_y_0 = 0
    U_y_L = 0
    # V horizontal BCs,
    V_y_0 = -3.5
    V_y_L = -3.5

    # U and V vertical BCs, periodic

    # Set up the square
    Ny = Nx
    y_0, y_L = x_0, x_L # y left and right boundaries
    x, y = (x_0, x_L), (y_0, y_L)
    dx = (x_L-x_0)/Nx # distance between nodes on type 1 grid (and type two grid)

    # inner linear system
    StokesSolver(mu, Nx, Ny, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L)
    return 0

if __name__ == "__main__":
    main()