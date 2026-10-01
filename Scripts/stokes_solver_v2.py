import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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

def buildDeriv(Nr,Nc, dir = 'x', p_mat= False):
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
                if p_mat:
                    row.append(Nc*i+j); col.append(Nc*i+j); val.append(-1.0)
                    row.append(Nc*i+j); col.append(Nc*i+j+Nc); val.append(1.0)
                else:
                    if i == 0:
                        row.append(Nc*i+j); col.append(Nc*i+j); val.append(1.0)                        
                    elif i == Nr-1:
                        row.append(Nc*i+j); col.append(Nc*i+j-Nc); val.append(-1.0)
                    else:
                        row.append(Nc*i+j); col.append(Nc*i+j); val.append(1.0)
                        row.append(Nc*i+j); col.append(Nc*i+j-Nc); val.append(-1.0)
        if p_mat:
            if Nc < Nr:
                A = sp.coo_array((val, (row, col)))
            else:
                 A = sp.coo_array((val, (row, col)))
        elif Nc > Nr:
            A = sp.coo_array((val, (row, col)))
        else:
            A = sp.coo_array((val, (row, col)))
    return A

def sysAssembly(mu, Nc, Nr, dx, x, y, f, g, U_top, V_top, U_bot, V_bot, verbose=False):
    # Calculate each block matrix and Build A
    L_u = mu*buildLaplacian(Nr, Nc, u_mat = True)
    L_v = mu*buildLaplacian(Nr-1,Nc)
    G_x = -1*buildDeriv(Nr, Nc, dir = 'x')*dx
    G_y = -1*buildDeriv(Nr-1,Nc, dir = 'y', p_mat = True)*dx
    D_x = buildDeriv(Nr, Nc, dir = 'x') #buildDeriv(Nx-1, Ny-1, direction = 'x')
    D_y = buildDeriv(Nr,Nc, dir = 'y') #buildDeriv(Nx-1, Ny-2, direction = 'y')
    print(D_y)

    print(L_u.shape)
    print(L_v.shape)
    print(G_x.shape)
    print(G_y.shape)
    print(D_x.shape)
    print(D_y.shape)

    pin = True
    if pin:
        G_x = G_x + sp.coo_array(([-1], ([0],[0])), shape=(Nr*Nr, Nr*Nc))

    A_grid = [[L_u, None, G_x],
              [None, L_v, G_y],
              [D_x, D_y, None]]
              #[G_x.T, G_y.T, None]]

    A = sp.block_array(A_grid, format = 'coo')

    #plt.spy(A, markersize=4, marker='.')
    fig, ax = plt.subplots()
    sc = ax.scatter(A.col, A.row, c=A.data, s=8, cmap="coolwarm", marker="s")
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)      # row 0 at the top, like spy
    ax.set_aspect("equal")
    ax.xaxis.tick_top()                       # optional: matches spy's axis placement
    fig.colorbar(sc, ax=ax, label="value")
    plt.savefig("Images\\Spy.png")

    # Build out b using a meshgrid
    x_u = np.linspace(x[0], x[1], Nc)
    y_u = np.linspace(y[0] , y[1], Nr) + dx/2
    xx_u, yy_u = np.meshgrid(x_u, y_u, indexing ='xy')
    F = f(xx_u, yy_u)*dx**2 # Forcing in X
    x_v = np.linspace(x[0], x[1], Nc) + dx/2
    y_v = np.linspace(y[0] + dx , y[1], Nr-1)
    xx_v, yy_v = np.meshgrid(x_v, y_v, indexing ='xy')
    G = g(xx_v, yy_v)*dx**2 # Forcing in Y
    O = np.zeros(((Nc) * ( Nr),))

    # apply BCs to forcing matricies
    F[0,:] -= U_top
    G[0,:] -= V_top
    F[-1,:] -= U_bot
    G[-1,:] -= V_bot
    b = np.hstack((F.flatten(),G.flatten(),O))

        
    return A, b

def StokesSolver(mu, Nx, Ny, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose = False):
       # inner linear system
    Nr = Ny - 1
    Nc = Nx - 1
    if verbose:
        print("Building Linear System...", end='\r')
    A, b = sysAssembly(mu, Nc, Nr, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose = verbose)
    A = A.tocsr()
    if verbose:
        print("Linear System Built.           ", end='\n')
        print("Solving for velocities...", end='\r')
    # Solve
    x = spa.spsolve(A, b)
    # x, exit_code = spa.gmres(A, b)
    if verbose:
        print("Velocities solved.              ", end='\n')
    U = x[0:Nc*Nr]
    V = x[Nc*Nr:(Nc*Nr+(Nr-1)*(Nc))]
    P = x[(Nc*Nr+(Nr-1)*(Nc)):]

    P = P - np.mean(P)

    # Put U, V, P into expected shapes
    U = U.reshape(Nr,Nc)
    V = V.reshape((Nr-1, Nc))
    P = P.reshape(Nr,Nc)

    return U, V, P

def grid_eval(u, v, p, Nx, Ny, dx, x, y):
    Nc = Nx - 1
    Nr = Ny - 1
    x_u = np.linspace(x[0], x[1], Nc)
    y_u = np.linspace(y[0] , y[1], Nr) + dx/2
    xx_u, yy_u = np.meshgrid(x_u, y_u, indexing ='xy')
    U = u(xx_u, yy_u)
    x_v = np.linspace(x[0], x[1], Nc) + dx/2
    y_v = np.linspace(y[0] + dx , y[1], Nr-1)
    xx_v, yy_v = np.meshgrid(x_v, y_v, indexing ='xy')
    V = v(xx_v, yy_v)
    x_p = np.linspace(x[0], x[1], Nc) + dx/2
    y_p = np.linspace(y[0] , y[1], Nr) + dx/2
    xx_p, yy_p = np.meshgrid(x_p, y_p, indexing ='xy')
    P = p(xx_p, yy_p)
    u_grids = [xx_u, yy_u, U]
    v_grids = [xx_v, yy_v, V]
    p_grids = [xx_p, yy_p, P]
    return u_grids, v_grids, p_grids

def main():
    # Control Panel
    mu = 1.0 # Viscousity
    Nx = 40 # number of nodes in the type 1 grid in each direction
    x_0 = 1.0 # x left boundary
    x_L = 6.0 # x right boundary
    tp = 2*np.pi
    f = lambda x, y: (tp - 2*tp**2) * np.sin(tp*x) * np.sin(tp*y)
    g = lambda x, y: np.cos(tp*x) * np.cos(tp*y) * (tp - 2*tp**2) + 2*tp**2 * np.cos(tp*x)

    u_exact = lambda x,y: np.sin(tp*y) * np.sin(tp*x)
    v_exact = lambda x,y: -3.5 + np.cos(tp*x)*(np.cos(tp*y)-1)
    p_exact = lambda x,y: np.sin(tp*y) * np.sin(tp*x)

    colormap = 'viridis'

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

    # Solve the problem
    U_approx,V_approx,P_approx = StokesSolver(mu, Nx, Ny, dx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose=True)
    
    # Error analysis
    U_bundle,V_bundle,P_bundle = grid_eval(u_exact, v_exact, p_exact, Nx, Ny, dx, x, y)
    U_exact = U_bundle[2]
    V_exact = V_bundle[2]
    P_exact = P_bundle[2]

    U_err = np.abs(U_approx - U_exact)/np.abs(U_exact)
    V_err = np.abs(V_approx - V_exact)/np.abs(V_exact)
    P_err = np.abs(P_approx - P_exact)/np.abs(P_exact)

    # vmin = min(u_err.min(), v_err.min(), p_err.min())
    # vmax = max(u_err.max(), v_err.max(), p_err.max())

    fig1, ax1 = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    im0 = ax1[0].pcolormesh(U_bundle[0], U_bundle[1], U_err, cmap=colormap)#, vmin=vmin, vmax=vmax)
    im1 = ax1[1].pcolormesh(V_bundle[0], V_bundle[1], V_approx, cmap=colormap)#, vmin=vmin, vmax=vmax)
    im2 = ax1[1].pcolormesh(P_bundle[0], P_bundle[1], P_approx, cmap=colormap)#, vmin=vmin, vmax=vmax)
    for ax in ax1:
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.xaxis.set_major_locator(MultipleLocator(1))
    fig1.colorbar(im0, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.6)
    ax1[0].set_title('Error in U')
    ax1[1].set_title('Error in V')
    ax1[2].set_title('Error in P')
    plt.savefig("Images\\Error_Surfaces.png", bbox_inches='tight')
    return 0

if __name__ == "__main__":
    main()