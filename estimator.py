import numpy as np
import conf
import casadi as cas
class Estimator():
    def __init__(self, vehicle, n, n_vehicles, N, all_vehicles = []):
        
        self.EKF = True
        self.vehicle = vehicle
        self.dt = vehicle.dt
        self.n = n
        self.N = N
        self.n_vehicles = n_vehicles
        self.n_state = n * n_vehicles

        self.S_hat = np.zeros((self.n_state, self.N))
        
        self.P = np.zeros((self.n_state, self.n_state, self.N))
        self.P[:,:, 0] = np.eye(self.n_state) * 10

        if self.EKF:
            self.S_sym = cas.SX.sym("S", self.n_state)
            self.U_sym = cas.SX.sym("U", self.n_vehicles * 2)
            self.nu_sym = cas.SX.sym("U", self.n_state)
            self.make_f_EKF(self.S_sym, self.U_sym, self.nu_sym)

        self.log_t = []

       


    def run_filter(self, GT, u, nu, eps, i, Q, R, visible_vehicles):
        # self.S_hat[:, i+1] = self.f(self.S_hat[:, i], u, nu)

        # Sigma points
        # mean = self.S_hat[:, i]
        # sigma_pts = []
        # L = np.linalg.cholesky((self.n_state) * self.P[:,:, i])
        # for j in range(self.n_state):
        #     sigma_pts.append(mean - L[:, j])
        #     sigma_pts.append(mean + L[:, j])

        # sigma_uns = []
        # weight = 1/(2 * self.n_state)
        # for j in range(2 * self.n_state):
        #     sigma_uns.append(self.f(sigma_pts[j], u, nu))

        # self.S_hat[:, i+1] = weight * sum([x for x in sigma_uns])
        # self.P[:,:, i+1] = Q + weight * sum([((np.atleast_2d(x - self.S_hat[:, i+1]).T @ np.atleast_2d(x - self.S_hat[:, i+1]))) for x in sigma_uns])

        # z_sigma = []    
        # for j in range(2 * self.n_state):
        #     zi = self.h(sigma_pts[j], eps*0, visible_vehicles, vsible_estimator, measuring=False)
        #     z_sigma.append(zi)
        
        # z = self.h(GT, eps, visible_vehicles, vsible_estimator)
        # mu_z = weight * sum([x for x in z_sigma])
        # s = R + weight * sum([((np.atleast_2d(z-mu_z).T @ np.atleast_2d(z-mu_z))) for z in z_sigma])
        # Pxz =  weight * sum([((np.atleast_2d(x - self.S_hat[:, i+1]).T @ np.atleast_2d(z - mu_z))) for (x,z) in zip(sigma_uns, z_sigma)])
        # W = Pxz @ np.linalg.inv(s)
        
        # self.S_hat[:, i+1] += W @ (z - mu_z)
        # self.P[:,:, i+1] = self.P[:,:, i+1].copy() - W @ s @ W.T
 
        self.make_h_EKF(self.S_sym, visible_vehicles)
        G = self.G_fun(self.S_hat[:,i], u).full()
        A = self.A_fun(self.S_hat[:,i], u, nu*0).full()
    
        # Prediction step
        self.S_hat[:,i+1] = self.f_fun(self.S_hat[:,i], u, nu).full().flatten()
        self.P[:,:, i+1] = A @ self.P[:,:,i] @ A.T + G @ Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:,i+1]).full()
        # z = self.h(i, GT, eps, visible_vehicles, vsible_estimator)
        z = self.h_fun(GT).full().flatten()

        z += eps
        S = H @ self.P[:,:,i+1] @ H.T + R
        w = self.P[:,:,i+1] @ H.T @ np.linalg.inv(S)
        self.S_hat[:,i+1] = self.S_hat[:,i+1] + (w @ (z.T - self.h_fun(self.S_hat[:,i+1]).full().flatten()))
        self.P[:,:,i+1] =  (np.eye(self.P.shape[0]) - w @ H) @ self.P[:,:,i+1]


    def h(self, S, eps, visible_vehicles, visible_estimator, measuring = False):
        x = S[0 + self.idx * self.n]
        y = S[1 + self.idx * self.n]
        d = S[2 + self.idx * self.n]

        z_tmp = [x, y, d]
        for (e,v) in zip(visible_estimator, visible_vehicles):
            # Meare the other vehicle via lidar
            x1 = S[self.n * self.vehicle2idx[v]]
            y1 = S[1 + self.n * self.vehicle2idx[v]]
            z_tmp.append(cas.sqrt((x1-x)**2 + (y1-y)**2))
            z_tmp.append(cas.arctan2(((y1-y)),(x1-x))-d)
            
        z = np.array(z_tmp) + eps

        # x1_b = x1 * np.cos(d) + y1 * np.sin(d) - x*np.cos(d) - y * np.sin(d)
        # y1_b = -x1 * np.sin(d) + y1 * np.cos(d) + x*np.sin(d) - y * np.cos(d)

        return z
    
    
    # def f(self, S, U, nu):
    #     f = []
    #     for i in range(self.n_vehicles):
    #         delta = S[2 + i*self.n]
    #         alpha = S[3 + i*self.n]
    #         v = S[4 + i*self.n]
    #         a = S[5 + i*self.n]
    #         u0 = U[i*self.n_vehicles]
    #         u1 = U[1 + i*self.n_vehicles]

    #         f.append(np.array([
    #             np.cos(delta) * v,
    #             np.sin(delta) * v,
    #             np.tan(alpha)/conf.L  *v,
    #             u1,
    #             a,
    #             1/conf.tau * (-a + u0)
    #         ]) * self.dt + 
    #         S[i*self.n : (i+1)*self.n] +
    #         nu[i*self.n : (i+1)*self.n])
    #     f = np.array(f).flatten()
    #     return f

    def f(self, S, U, nu):
        f = []
        for i in range(self.n_vehicles):
            delta = S[2 + i*self.n]
            u0 = U[i*2]
            u1 = U[1 + i*2]

            f.append(np.array([
                np.cos(delta) * u0,
                np.sin(delta) * u0,
                u1,
            ]) * self.dt + 
            S[i*self.n : (i+1)*self.n] +
            nu[i*self.n : (i+1)*self.n])
        f = np.array(f).flatten()
        return f
    
    def make_h_EKF(self, S_sym, visible_vehicles):  
        x = S_sym[0 + self.idx * self.n]
        y = S_sym[1 + self.idx * self.n]
        d = S_sym[2 + self.idx * self.n]

        h_tmp = [x, y, d]

        for v in visible_vehicles:
            x1 = S_sym[self.n * self.vehicle2idx[v]]
            y1 = S_sym[1 + self.n * self.vehicle2idx[v]]
            d1 = S_sym[2 + self.n * self.vehicle2idx[v]]

            h_tmp.append(cas.sqrt((x1-x)**2 + (y1-y)**2 +1e-6))
            h_tmp.append(cas.arctan2((y1-y) + 1e-6,(x1-x) + 1e-6)-d)

            # Get front vehicle information via communicatio
            # h_tmp.append(x1)
            # h_tmp.append(y1)
            # h_tmp.append(d1)
            # R_tmp.append(e.P[self.n * self.vehicle2idx[v], self.n * self.vehicle2idx[v], i])
            # R_tmp.append(e.P[self.n * self.vehicle2idx[v] + 1, self.n * self.vehicle2idx[v] +1, i])
            # R_tmp.append(e.P[self.n * self.vehicle2idx[v] + 2, self.n * self.vehicle2idx[v] +2, i])

        h=[]
        for e in h_tmp: h = cas.vertcat(h, e)

        H = cas.jacobian(h, self.S_sym)
        self.H_fun = cas.Function('H_fun', [self.S_sym], [H])
        self.h_fun = cas.Function('h_fun', [self.S_sym], [h])  
        
    
    # def make_f_EKF(self, S_sym, U_sym, nu_sym):
    #     f_tmp = []
    #     for i in range(self.n_vehicles):
    #         delta = S_sym[2 + i*self.n]
    #         alpha = S_sym[3 + i*self.n]
    #         v = S_sym[4 + i*self.n]
    #         a = S_sym[5 + i*self.n]
    #         u0 = U_sym[i*self.n_vehicles]
    #         u1 = U_sym[1 + i*self.n_vehicles]

    #         f_tmp.append(cas.vertcat(
    #             cas.cos(delta) * v,
    #             cas.sin(delta) * v,
    #             cas.tan(alpha)/conf.L *v,
    #             u1,
    #             a,
    #             1/conf.tau * (-a + u0)
    #         ) * self.dt + 
    #         S_sym[i*self.n : (i+1)*self.n] +
    #         nu_sym[i*self.n : (i+1)*self.n])

    #     f=[]
    #     for e in f_tmp: f = cas.vertcat(f, e)

    #     A = cas.jacobian(f, S_sym)
    #     G = cas.jacobian(f, nu_sym)
    #     self.A_fun = cas.Function('A_fun', [S_sym, U_sym, nu_sym], [A])
    #     self.G_fun = cas.Function('G_fun', [S_sym, U_sym], [G])
    #     self.f_fun = cas.Function('f_fun', [S_sym, U_sym, nu_sym], [f])

    def make_f_EKF(self, S_sym, U_sym, nu_sym):
        f_tmp = []
        for i in range(self.n_vehicles):
            delta = S_sym[2 + i*self.n]
            u0 = U_sym[i*2]
            u1 = U_sym[1 + i*2]

            f_tmp.append(cas.vertcat(
                cas.cos(delta) * u0,
                cas.sin(delta) * u0,
                u1,
            ) * self.dt + 
            S_sym[i*self.n : (i+1)*self.n] +
            nu_sym[i*self.n : (i+1)*self.n])

        f=[]
        for e in f_tmp: f = cas.vertcat(f, e)

        A = cas.jacobian(f, S_sym)
        G = cas.jacobian(f, nu_sym)
        self.A_fun = cas.Function('A_fun', [S_sym, U_sym, nu_sym], [A])
        self.G_fun = cas.Function('G_fun', [S_sym, U_sym], [G])
        self.f_fun = cas.Function('f_fun', [S_sym, U_sym, nu_sym], [f])

    def set_fleet(self, all_vehicles):
        self.vehicle2idx = {}
        self.idx = all_vehicles.index(self.vehicle)
        for i, v in enumerate(all_vehicles):
            self.vehicle2idx[v] = i

    def get_vehicle_estimate(self, vehicle, i):
        x1 = self.S_hat[self.n * self.vehicle2idx[vehicle], i]
        y1 = self.S_hat[1 + self.n * self.vehicle2idx[vehicle], i]
        d1 = self.S_hat[2 + self.n * self.vehicle2idx[vehicle], i]

        return x1, y1, d1
    
    def get_vehicle_P_estimate(self, vehicle, i):
        Px = self.P[self.n * self.vehicle2idx[vehicle], self.n * self.vehicle2idx[vehicle], i]
        Py = self.P[1 + self.n * self.vehicle2idx[vehicle], 1 + self.n * self.vehicle2idx[vehicle], i]
        Pd = self.P[2 + self.n * self.vehicle2idx[vehicle], 2 + self.n * self.vehicle2idx[vehicle], i]

        return Px, Py, Pd