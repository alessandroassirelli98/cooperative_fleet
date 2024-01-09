import numpy as np
import conf
import casadi as cas

# Class definition for the Estimator
class Estimator():
    # Constructor method to initialize the Estimator object
    def __init__(self, vehicle, n, n_vehicles, N, all_vehicles=[]):

        # Initialize parameters and properties of the estimator
        self.EKF = True  # Enable Extended Kalman Filter (EKF)
        self.vehicle = vehicle  # Current vehicle associated with the estimator
        self.dt = vehicle.dt  # Time step
        self.n = n  # Number of state variables for each vehicle
        self.N = N  # Number of time steps
        self.n_vehicles = n_vehicles  # Total number of vehicles
        self.n_state = n * n_vehicles  # Total number of state variables for all vehicles

        # Initialize state and covariance matrices
        self.S_hat = np.zeros((self.n_state, self.N))  # State estimate matrix
        self.P = np.zeros((self.n_state, self.n_state, self.N))  # Covariance matrix
        self.P[:, :, 0] = np.eye(self.n_state) * 10  # Initial covariance matrix

        # Check if Extended Kalman Filter (EKF) is enabled
        if self.EKF:
            # Define symbolic variables for state, inputs, and process noise
            self.S_sym = cas.SX.sym("S", self.n_state)  # State vector
            self.U_sym = cas.SX.sym("U", self.n_vehicles * 2)  # Input vector
            self.nu_sym = cas.SX.sym("U", self.n_state)  # Process noise vector
            # Call the method to set up EKF functions
            self.make_f_EKF(self.S_sym, self.U_sym, self.nu_sym)

        # Log variable for debugging
        self.log_t = []

    # Method to run the filter
    def run_filter(self, GT, u, nu, eps, i, Q, R, visible_vehicles):
        # Call method to set up measurement functions for EKF
        self.make_h_EKF(self.S_sym, visible_vehicles)
        
        # Compute Jacobians for prediction step
        G = self.G_fun(self.S_hat[:, i], u).full()
        A = self.A_fun(self.S_hat[:, i], u, nu * 0).full()

        # Prediction step
        self.S_hat[:, i + 1] = self.f_fun(self.S_hat[:, i], u, nu).full().flatten()
        self.P[:, :, i + 1] = A @ self.P[:, :, i] @ A.T + G @ Q @ G.T

        # Update step
        H = self.H_fun(self.S_hat[:, i + 1]).full()
        z = self.h_fun(GT).full().flatten()
        z += eps
        S = H @ self.P[:, :, i + 1] @ H.T + R
        w = self.P[:, :, i + 1] @ H.T @ np.linalg.inv(S)
        self.S_hat[:, i + 1] = self.S_hat[:, i + 1] + (w @ (z.T - self.h_fun(self.S_hat[:, i + 1]).full().flatten()))
        self.P[:, :, i + 1] = (np.eye(self.P.shape[0]) - w @ H) @ self.P[:, :, i + 1]

    # Method to define the measurement function for EKF
    def make_h_EKF(self, S_sym, visible_vehicles):
        # Extract relevant state variables
        x = S_sym[0 + self.idx * self.n]
        y = S_sym[1 + self.idx * self.n]
        d = S_sym[2 + self.idx * self.n]
        alpha = S_sym[3 + self.idx * self.n]
        vel = S_sym[4 + self.idx * self.n]
        a = S_sym[5 + self.idx * self.n]

        # Initialize temporary list for measurement function
        h_tmp = [x, y, d, alpha, vel]
        M10 = np.array([[cas.cos(d), cas.sin(d), - x * cas.cos(d) - y * cas.sin(d)],
                    [-cas.sin(d), cas.cos(d), -y * cas.cos(d) + x * cas.sin(d)],
                     [0,0,1]])

        # Loop over visible vehicles to add measurements
        for v in visible_vehicles:
            x1 = S_sym[self.n * self.vehicle2idx[v]]
            y1 = S_sym[1 + self.n * self.vehicle2idx[v]]             
            d1 = S_sym[2 + self.n * self.vehicle2idx[v]]            


            # Append distance and angle measurements
            # h_tmp.append(x1)
            # h_tmp.append(y1)
            h_tmp.append(cas.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + 1e-4))
            h_tmp.append(cas.arctan2((y1 - y) + 1e-4, (x1 - x) + 1e-4) - d)          
            # h_tmp.append((M10 @ np.array([x1,y1,1]))[0])
            # h_tmp.append((M10 @ np.array([x1,y1,1]))[1])
            h_tmp.append(d1 - d)          
            # h_tmp.append(cas.arctan(((y1 - y) + 1e-4) / ((x1 - x) + 1e-4)) - d)


        # Concatenate the temporary list to form the measurement vector
        h = []
        for e in h_tmp:
            h = cas.vertcat(h, e)

        # Compute Jacobian matrix for the measurement function
        H = cas.jacobian(h, self.S_sym)

        # Create CasADi functions for the measurement and Jacobian
        self.H_fun = cas.Function('H_fun', [self.S_sym], [H])
        self.h_fun = cas.Function('h_fun', [self.S_sym], [h])

    # Method to define the state transition function for EKF
    def make_f_EKF(self, S_sym, U_sym, nu_sym):
        # Initialize temporary list for state transition function
        f_tmp = []
        for i in range(self.n_vehicles):
            delta = S_sym[2 + i * self.n]
            alpha = S_sym[3 + i * self.n]
            v = S_sym[4 + i * self.n]
            a = S_sym[5 + i * self.n]
            u0 = U_sym[i * 2]
            u1 = U_sym[1 + i * 2]

            # Add the state transition equations to the temporary list
            f_tmp.append(cas.vertcat(
                cas.cos(delta) * v,
                cas.sin(delta) * v,
                cas.tan(alpha) / conf.L * v,
                u1,
                a,
                1 / conf.tau * (-a + u0)
            ) * self.dt +
                         S_sym[i * self.n: (i + 1) * self.n] +
                         nu_sym[i * self.n: (i + 1) * self.n])

        # Concatenate the temporary list to form the state transition vector
        f = []
        for e in f_tmp:
            f = cas.vertcat(f, e)

        # Compute Jacobians for the state transition function
        A = cas.jacobian(f, S_sym)
        G = cas.jacobian(f, nu_sym)

        # Create CasADi functions for the state transition and Jacobians
        self.A_fun = cas.Function('A_fun', [S_sym, U_sym, nu_sym], [A])
        self.G_fun = cas.Function('G_fun', [S_sym, U_sym], [G])
        self.f_fun = cas.Function('f_fun', [S_sym, U_sym, nu_sym], [f])

    # Method to set the vehicles in the fleet and initialize mapping
    def set_fleet(self, all_vehicles):
        self.vehicle2idx = {}  # Initialize dictionary for vehicle indices
        self.idx = all_vehicles.index(self.vehicle)  # Get index of current vehicle
        # Loop over all vehicles to create mapping
        for i, v in enumerate(all_vehicles):
            self.vehicle2idx[v] = i

    # Method to get the estimated state of a specific vehicle at a given time step
    def get_vehicle_estimate(self, vehicle, i):
        x1 = self.S_hat[self.n * self.vehicle2idx[vehicle], i]
        y1 = self.S_hat[1 + self.n * self.vehicle2idx[vehicle], i]
        d1 = self.S_hat[2 + self.n * self.vehicle2idx[vehicle], i]

        return x1, y1, d1

    # Method to get the estimated covariance of a specific vehicle at a given time step
    def get_vehicle_P_estimate(self, vehicle, i):
        Px = self.P[self.n * self.vehicle2idx[vehicle], self.n * self.vehicle2idx[vehicle], i]
        Py = self.P[1 + self.n * self.vehicle2idx[vehicle], 1 + self.n * self.vehicle2idx[vehicle], i]
        Pd = self.P[2 + self.n * self.vehicle2idx[vehicle], 2 + self.n * self.vehicle2idx[vehicle], i]

        return Px, Py, Pd