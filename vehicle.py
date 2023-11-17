import numpy as np
from street import Street, Lane
from estimator import Estimator
import conf

class Vehicle:
    def __init__(self, street:Street, lane:Lane, s, v, dt):
        self.cnt = 0

        self.dt = dt
        self.street = street
        self.lane = lane
        self.L = conf.L

        self.x, self.y = self.lane.s_to_xy(s)
        self.s = s
        self.delta = 0
        self.alpha = 0
        self.v = v
        self.v_des = v
        self.omega = 0
        self.a = 0
        self.u_fwd = 0
        self.u = np.array([self.u_fwd, self.omega])
        
        self.S0 = np.array([self.x, self.y, self.delta])
        self.S = self.S0
    
    def update(self, u, nu):

        self.S = self.f(self.S, u, nu)

        self.x = self.S[0]
        self.y = self.S[1]
        self.delta = self.S[2] 
        self.s = self.street.xy_to_s(self.x, self.y)
        # self.alpha = self.S[3] 
        # self.v = self.S[4] 
        # self.a = self.S[5]
        

    # def f(self, X, U, nu):
    #     delta = X[2] 
    #     alpha = X[3] 
    #     v = X[4] 
    #     a = X[5]
    #     u0 = U[0]
    #     u1 = U[1]

    #     xp1 = np.array([
    #         np.cos(delta) * v,
    #         np.sin(delta) * v,
    #         np.tan(alpha)/self.L  *v,
    #         u1,
    #         a,
    #         1/conf.tau * (-a + u0)
    #     ]) * self.dt + X + nu

    #     return xp1

    def f(self, X, U, nu):
        delta = X[2] 
        u0 = U[0]
        u1 = U[1]

        xp1 = np.array([
            np.cos(delta) * u0,
            np.sin(delta) * u0,
            u1
        ]) * self.dt + X + nu

        return xp1
