import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def buildGrids(x, y, Nc, Nr):
    dx = (x[1]-x[0])/(Nc)
    x_1 = np.linspace(x[0]+dx, x[1], Nc)
    x_2 = np.linspace(x[0]+dx/2, x[1]-dx/2, Nc)
    y_1 = np.linspace(y[0]+dx, y[1]-dx, Nr-1)
    y_2 = np.linspace(y[0]+dx/2, y[1]-dx/2, Nr)
    
    xx_u, yy_u = np.meshgrid(x_1, y_2, indexing ='xy')
    xx_v, yy_v = np.meshgrid(x_2, y_1, indexing ='xy')
    xx_p, yy_p = np.meshgrid(x_2, y_2, indexing ='xy')
    
    #O = np.zeros((Nc, Nr))
    
    ugrids = [xx_u, yy_u]
    vgrids = [xx_v, yy_v]
    pgrids = [xx_p, yy_p]

    return ugrids, vgrids, pgrids

def coloredSpy(A, colormap = "coolwarm", filepath = "Images\\Spy.png"):
    #plt.spy(A, markersize=4, marker='.')
    fig, ax = plt.subplots()
    sc = ax.scatter(A.col, A.row, c=A.data, s=8, cmap=colormap, marker="s")
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)      # row 0 at the top, like spy
    ax.set_aspect("equal")
    ax.xaxis.tick_top()                       # optional: matches spy's axis placement
    fig.colorbar(sc, ax=ax, label="value")
    plt.savefig(filepath)
    plt.close(fig)
    return 0

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
                if p_mat:
                    row.append(Nc*i+j); col.append(Nc*i+(j+1) % Nc); val.append(1.0) # loop back for periodicity
                    row.append(Nc*i+j); col.append(Nc*i+(j) % Nc); val.append(-1.0) # loop back for periodicity
                else:
                    row.append(Nc*i+j); col.append(Nc*i+(j-1) % Nc); val.append(-1.0) # loop back for periodicity
                    row.append(Nc*i+j); col.append(Nc*i+(j) % Nc); val.append(1.0) # loop back for periodicity
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

def sysAssembly(mu, Nc, Nr, x, y, f, g, U_top, V_top, U_bot, V_bot, verbose=False, spy=False):
    dx = (x[1]-x[0])/(Nr)

    # Calculate each block matrix and Build A
    L_u = mu*buildLaplacian(Nr, Nc, u_mat = True)
    L_v = mu*buildLaplacian(Nr-1,Nc)
    G_x = -1.0*buildDeriv(Nr, Nc, dir = 'x', p_mat = True)
    G_y = -1.0*buildDeriv(Nr-1,Nc, dir = 'y', p_mat = True)
    D_x = buildDeriv(Nr, Nc, dir = 'x') #buildDeriv(Nx-1, Ny-1, direction = 'x')
    D_y = buildDeriv(Nr,Nc, dir = 'y') #buildDeriv(Nx-1, Ny-2, direction = 'y')
    
    # print(L_u.shape)
    # print(L_v.shape)
    # print(G_x.shape)
    # print(G_y.shape)
    # print(D_x.shape)
    # print(D_y.shape)

    pin = False
    if pin:
        G_x_pinned = G_x + sp.coo_array(([-1], ([0],[0])), shape=(Nr*Nr, Nr*Nc))
        A_grid = [[L_u, None, dx*G_x_pinned],
                  [None, L_v, dx*G_y],
                  [D_x, D_y, None]] # Need to revisit how we are building D, it doesn't match G_x.T and it should (at least need to understand where we went wrong)
                  #[G_x.T, G_y.T, None]]
    else:
         A_grid = [[L_u, None, dx*G_x],
                          [None, L_v, dx*G_y],
                          [D_x, D_y, None]] # Need to revisit how we are building D, it doesn't match G_x.T and it should (at least need to understand where we went wrong)
                          #[G_x.T, G_y.T, None]]        

    A = sp.block_array(A_grid, format = 'coo')

    if spy:
        coloredSpy(A)

    u_grid, v_grid, p_grid = buildGrids(x, y, Nc, Nr)
    F = f(u_grid[0], u_grid[1])*dx**2 # Forcing in X
    G = g(v_grid[0], v_grid[1])*dx**2 # Forcing in Y
    O = np.zeros((Nc, Nr))

    # apply BCs to forcing matricies
    F[0,:] -= 2*U_bot   # Ghost point for u
    G[0,:] -= mu*V_bot  # Dirichlet value
    O[0,:] += V_bot     # dirichlet
    F[-1,:] -= 2*U_top
    G[-1,:] -= mu*V_top
    O[-1,:] -= V_top

    b = np.hstack((F.flatten(),G.flatten(),O.flatten()))
       
    return A, b

def StokesSolver(mu, Nx, Ny, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose = False, make_spy = False):
    Nr = Ny - 1
    Nc = Nx - 1
    dx = (x[1]-x[0])/(Nx-1) # distance between nodes on type 1 grid (and type two grid)
    if verbose:
        print("Building Linear System...", end='\r')
    A, b = sysAssembly(mu, Nc, Nr, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, spy=make_spy)
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

def grid_eval(u, v, p, Nx, Ny, x, y):
    dx = (x[1]-x[0])/(Nx-1)
    Nc = Nx - 1
    Nr = Ny - 1

    u_grids, v_grids, p_grids = buildGrids(x, y, Nc, Nr)

    U = u(u_grids[0], u_grids[1])
    V = v(v_grids[0], v_grids[1])
    P = p(p_grids[0], p_grids[1])

    u_grids.append(U)
    v_grids.append(V)
    p_grids.append(P)

    return u_grids, v_grids, p_grids

def bcApply(u,v, BCs):
    U = np.hstack((u[:,-1:],u)) # Apply Periodic BC in U
    V = np.vstack((BCs[0]*np.ones((1,v.shape[1])),v,BCs[1]*np.ones((1,v.shape[1])))) # Apply Dirichlet BCs in Y
    return U, V

def gridInterp(u,v, BCs):
    u_temp, v_temp = bcApply(u,v, BCs)
    u_vec = np.zeros((u_temp.shape[0], u_temp.shape[1]-1))
    v_vec = np.zeros((v_temp.shape[0]-1, v_temp.shape[1]))
    for i in range(v_vec.shape[0]):
        u_vec[:,i] = (u_temp[:, i+1] + u_temp[:, i])/2.0
        v_vec[i,:] = (v_temp[i+1,:] + v_temp[i,:])/2.0
    return [u_vec, v_vec]

def rel_error(u_exact, u_approx):
    L_2_error = np.linalg.norm(u_exact-u_approx, ord = 'fro')/np.linalg.norm(u_exact, ord = 'fro')
    return L_2_error

def convTest(mesh_range, num_trials, exact_sol, num_Operator, x, y, BCs, filepath = 'Images\\convergence_plot.png'):
    meshes = np.floor(np.linspace(mesh_range[0], mesh_range[1], num_trials)).astype(int)
    L_2 = np.zeros((3,num_trials))
    for i in range(num_trials):
        print(f"Simulating mesh with {meshes[i]**2} nodes")
        U_temp,V_temp,P_approx = num_Operator(meshes[i])
        U_approx, V_approx = gridInterp(U_temp, V_temp, BCs)
        u_grid, v_grid, p_grid = buildGrids(x, y, meshes[i]-1, meshes[i]-1)
        xx, yy = p_grid[0], p_grid[1]
        U_exact, V_exact, P_exact = exact_sol(xx,yy)
        
        L_2[0,i] = rel_error(U_exact, U_approx) #calculate relative errors for increasingly finer meshes.
        L_2[1,i] = rel_error(V_exact, V_approx) #calculate relative errors for increasingly finer meshes.
        L_2[2,i] = rel_error(P_exact, P_approx) #calculate relative errors for increasingly finer meshes.

    fig3 = plt.figure(3)
    ax3 = plt.axes()
    plt.rcParams['lines.linewidth'] = 3 
    plt.loglog(meshes, L_2[1,:], label = "V")
    plt.loglog(meshes, L_2[2,:], label = "P")
    plt.loglog(meshes, 1/meshes**2, label = "Reference")
    plt.loglog(meshes, L_2[0,:], label = "U")
    plt.legend()
    ax3.set_xlabel("Number of FD Nodes")
    ax3.set_ylabel("Relative Error")
    plt.savefig(filepath)
    return 0

def main():
    # Control Panel
    mu = 1.0 # Viscousity
    Nx = 100 # number of nodes in the type 1 grid in each direction
    x_0 = 0.0 # x left boundary
    x_L = 1.0 # x right boundary
    tp = 2*np.pi
    
    f = lambda x, y: -2*tp**2*np.sin(tp*x)*np.sin(tp*y) - tp*np.cos(tp*x)*np.sin(tp*y)
    g = lambda x, y: -2*tp**2*np.cos(tp*x)*np.cos(tp*y) + tp**2*np.cos(tp*x) - tp*np.sin(tp*x)*np.cos(tp*y)

    u_exact = lambda x,y: np.sin(tp*y) * np.sin(tp*x)
    v_exact = lambda x,y: -3.5 + np.cos(tp*x)*(np.cos(tp*y)-1)
    p_exact = lambda x,y: np.sin(tp*y) * np.sin(tp*x)

    colormap = 'viridis'

    # U and V vertical BCs, periodic
    # U horizontal BCs
    U_y_0 = 0
    U_y_L = 0
    # V horizontal BCs,
    V_y_0 = -3.5
    V_y_L = -3.5
    V_BCs = [V_y_0, V_y_L]

    # Set up the square
    Ny = Nx
    y_0, y_L = x_0, x_L # y left and right boundaries
    x, y = (x_0, x_L), (y_0, y_L)
    
    # Solve the problem
    U_approx,V_approx,P_approx = StokesSolver(mu, Nx, Ny, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose=True, make_spy=True)

    # Error analysis
    U_bundle,V_bundle,P_bundle = grid_eval(u_exact, v_exact, p_exact, Nx, Ny, x, y)
    U_exact = U_bundle[2]
    V_exact = V_bundle[2]
    P_exact = P_bundle[2]

    U_err = np.abs(U_approx - U_exact)
    V_err = np.abs(V_approx - V_exact)
    P_err = np.abs(P_approx - P_exact)

    #vmin = min(U_approx.min(), V_approx.min(), P_approx.min())
    #vmax = max(U_approx.max(), V_approx.max(), P_approx.max())
    vmin = min(U_err.min(), V_err.min(), P_err.min())
    vmax = max(U_err.max(), V_err.max(), P_err.max())

    fig1, ax1 = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    im0 = ax1[0].pcolormesh(U_bundle[0], U_bundle[1], U_err, cmap=colormap, vmin=vmin, vmax=vmax)
    im1 = ax1[1].pcolormesh(V_bundle[0], V_bundle[1], V_err, cmap=colormap, vmin=vmin, vmax=vmax)
    im2 = ax1[2].pcolormesh(P_bundle[0], P_bundle[1], P_err, cmap=colormap, vmin=vmin, vmax=vmax)
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

    # Plot convergence curve
    O_stokes = lambda Nx: StokesSolver(mu, Nx, Nx, x, y, f, g, U_y_0, V_y_0, U_y_L, V_y_L, verbose=False, make_spy=False)
    exact_sol = lambda x,y: [u_exact(x,y), v_exact(x,y), p_exact(x,y)]
    convTest([10,100], 10, exact_sol, O_stokes, x, y, V_BCs)

    # Plot mag of u
    U_vec = gridInterp(U_approx, V_approx, V_BCs)
    U_mag = np.sqrt(U_vec[0]**2 + U_vec[1]**2)
    plt.figure()
    ax = plt.axes()
    plt.pcolormesh(P_bundle[0], P_bundle[1], U_mag, cmap = 'viridis')
    plt.streamplot(P_bundle[0], P_bundle[1], U_vec[0], U_vec[1], density = 0.5, color = "white", broken_streamlines=False, num_arrows = 5)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig1.colorbar(im0, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.6)
    ax1[0].set_title('Error in U')
    ax1[1].set_title('Error in V')
    ax1[2].set_title('Error in P')
    plt.savefig("Images\\mag_u.png", bbox_inches='tight')

    # Profile code
    
    # Call this done

    return 0

if __name__ == "__main__":
    main()