import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import casadi as cas
plt.style.use('seaborn')
import utils


n_vehicles = 2
N = 10000
n = 6
dt = 0.01

x_true = np.zeros((n * n_vehicles,N))


S_hat = []
P = []
for i in range(n_vehicles):
    S_hat.append(np.zeros((n * n_vehicles,N)))
    P.append(np.zeros((n * n_vehicles, n * n_vehicles, N)))
    P[i][:,:,0] = np.eye(n_vehicles * n) * 10

x_true[0,0] = 10
S_hat[0][:,0] = x_true[:,0]
S_hat[1][:,0] = x_true[:,0]

sigma_eps_gps = 1e-2
sigma_eps_radar = 1e-2
sigma_u = 1e-1

Q = np.eye(n*n_vehicles) * sigma_u**2
R = np.diag([sigma_eps_gps**2, sigma_eps_gps**2,sigma_eps_radar**2, sigma_eps_radar**2, sigma_eps_gps**2, sigma_eps_gps**2])
R_r = np.diag([sigma_eps_radar**2, sigma_eps_radar**2])

u = [np.array([0.1, 0.0001]), np.array([0.1,0.])]
u = np.array(u).flatten()

x_sym_dist = cas.SX.sym("x_sym_dist", n * n_vehicles)
u_sym_dist = cas.SX.sym("u_sym_dist", 2 * n_vehicles)
nu_sym_dist = cas.SX.sym("nu_sym_dist", n * n_vehicles)

x0 = x_sym_dist[0]
y0 = x_sym_dist[1]
d0 = x_sym_dist[2]
x1 = x_sym_dist[6]
y1 = x_sym_dist[7]
d1 = x_sym_dist[8]

f_dist = utils.sys_model(dt, n_vehicles, n, x_sym_dist, u_sym_dist, nu_sym_dist)



f_fun = cas.Function('F_fun_dist', [x_sym_dist, u_sym_dist , nu_sym_dist], [f_dist])
# h_fun_dist= cas.Function('h_fun', [x_sym_dist], [h_dist])

A_dist = cas.jacobian(f_dist, x_sym_dist)
G_dist = cas.jacobian(f_dist, nu_sym_dist)
# H_dist = cas.jacobian(h_dist, x_sym_dist)
A_dist_fun= cas.Function('A_fun', [x_sym_dist, u_sym_dist, nu_sym_dist], [A_dist])
G_dist_fun = cas.Function('G_fun', [x_sym_dist, u_sym_dist], [G_dist])
# H_dist_fun = cas.Function('H_fun_dist', [x_sym_dist], [H_dist])

log = [[], []]

for t in range(N-1):
    F = []
    a = []

    for i in range(n_vehicles):
        nu = np.random.multivariate_normal(np.zeros((Q.shape[0])), Q)

        x_true[:,t+1] = f_fun(x_true[:, t], u, 0*nu).full().flatten()
        

        p_hat = f_fun(S_hat[i][:,t], u, nu).full().flatten()
        P[i][:,:,t+1] = A @ P[i][:,:,t] @ A.T + G @ Q @ G.T

        H = H_dist_fun(p_hat).full()

        I = zi.T - h_fun_dist(p_hat).full().flatten()
        s = H @ P[i][:,:,t+1] @ H.T + (R)
        w = P[i][:,:,t+1] @ H.T @ np.linalg.inv(s)
        S_hat[i][:,t+1] = p_hat + w @ I
        P[i][:,:,t+1] =  (np.eye(n*n_vehicles) - w @ H) @ P[i][:,:,t+1]


        # UKF
        # Sigma points
        # mean = S_hat[i][:, t]
        # sigma_pts = []
        # L = np.linalg.cholesky((n * n_vehicles) * P[i][:,:, t])
        # for j in range(n * n_vehicles):
        #     sigma_pts.append(mean - L[:, j])
        #     sigma_pts.append(mean + L[:, j])

        # sigma_uns = []
        # weight = 1/(2*n * n_vehicles)
        # for j in range(2*n * n_vehicles):
        #     sigma_uns.append(f_fun(sigma_pts[j], u, nu).full().flatten())

        # S_hat[i][:, t+1] = weight * sum([x for x in sigma_uns])
        # P[i][:,:, t+1] = Q + weight * sum([((np.atleast_2d(x - S_hat[i][:, t+1]).T @ np.atleast_2d(x - S_hat[i][:, t+1]))) for x in sigma_uns])


        # z_sigma = []    
        # for j in range(2*n * n_vehicles):
        #     z_sigma.append(h_fun_dist(sigma_pts[j]).full().flatten())
        
        # mu_z = weight * sum([x for x in z_sigma])
        # s = R + weight * sum([((np.atleast_2d(z-mu_z).T @ np.atleast_2d(z-mu_z))) for z in z_sigma])
        # Pxz =  weight * sum([((np.atleast_2d(x - S_hat[i][:, t+1]).T @ np.atleast_2d(z - mu_z))) for (x,z) in zip(sigma_uns, z_sigma)])
        # W = Pxz @ np.linalg.inv(s)
        # S_hat[i][:, t+1] += W @ (zi - mu_z)
        # P[i][:,:, t+1] = P[i][:,:, t+1].copy() - W @ s @ W.T

        # F.append(np.zeros((3,3)))
        # a.append(np.zeros((3)))
        # F[i] =  H.T @ np.linalg.inv( R ) @ H
        # a[i] =  H.T @ np.linalg.inv( R ) @ zi

    # m = 10 #Messages exchange
    # for k in range(m):
    #     A = np.zeros((n_vehicles, n_vehicles))
    #     for j in range(n_vehicles):
    #         for k in range(j+1, n_vehicles):
    #             d = x_true_split[i][:2] - x_true_split[j][:2]
    #             A[i,j] = np.sqrt(d@d.T) <= CR
    #     A = A + A.T
    #     D = A @ np.ones((n_vehicles,1))

    #     F_store = F.copy()
    #     a_store = a.copy()
        # for i in range(n_vehicles):
        #     for j in range(n_vehicles):
        #         if (A[i,j] == 1):
        #             F[i] = F[i] + 1/np.max(D) * (F_store[j] - F_store[i])  
        #             a[i] = a[i] + 1/np.max(D) * (a_store[j] - a_store[i])  
    
    # Estimates
    # for i in range(n_vehicles):
    #     # [i][:, t+1] = (np.linalg.pinv(F[i])@a[i]).copy()
    #     M = np.linalg.pinv(P[i][:,:,t+1] @ F[i])
    #     S_hat[i][:, t+1] = M @ (P[i][:,:,t+1] @ p_hat + a[i])
    #     P[i][:,:,t+1] = M

    

   