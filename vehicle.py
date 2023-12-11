import numpy as np
from street import Street, Lane
from estimator import Estimator
from ppc import PPC
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
        
        self.S0 = np.array([self.x, self.y, self.delta, self.alpha, self.v, self.a])
        self.S = self.S0

        self.steer_control = PPC()
        self.steer_control.lookAheadDistance = 15
        self.path = np.array([[lane.x_start, lane.y_start], [lane.x_end, lane.y_end]])

        self.starting_battery = 100
        self.life = self.starting_battery
        self.c0 = 100/100 / 1000  
        self.c1 = 100/100 / 1000
        self.overtaking = False
        self.which_lane = 0
        self.lead = False

    def change_lane(self, lane):
        d = self.street.angle
        x = self.x
        y = self.y
        M10 = np.array([[np.cos(d), np.sin(d), - x * np.cos(d) - y * np.sin(d)],
                    [-np.sin(d), np.cos(d), -y * np.cos(d) + x * np.sin(d)],
                     [0,0,1]])
        M01 = np.array([[np.cos(d), -np.sin(d), x],
                        [np.sin(d), np.cos(d), y],
                        [0,0,1]])
        
        if lane == 1:
            target = M01 @ np.array([self.street.lane_width, self.street.lane_width/2,1])
            x_target = target[0]
            y_target = target[1]
            self.path = np.array([[self.x, self.y], 
                                [x_target, y_target], 
                                [self.lane.x_end, y_target]]) # Change lane
            self.which_lane = 1
        elif lane == 0 :
            target = M01 @ np.array([self.street.lane_width, self.street.lane_width/2,1])
            x_target = target[0]
            y_target = self.lane.y_end
            self.path = np.array([[self.x, self.y], 
                                [x_target, y_target], 
                                [self.lane.x_end, self.lane.y_end]]) # Change lane
            self.which_lane = 0
            

    def compute_steering(self, x, y):
        alpha_des = self.steer_control.ComputeSteeringAngle(self.path, np.array([x,y]), self.delta, self.L)
        return alpha_des

    def update(self, u, nu):
        self.S = self.f(self.S, u, nu)


        self.s_ = self.s
        self.x = self.S[0]
        self.y = self.S[1]
        self.delta = self.S[2] 
        self.alpha = self.S[3] 
        self.v = self.S[4] 
        self.a = self.S[5]
        self.u_fwd = u[0]
        self.omega = u[1]
        self.s = self.street.xy_to_s(self.x, self.y)

        self.life -=  ( self.s - self.s_ ) * (self.c0 + self.c1 if self.lead else self.c0)
        self.lead = False

        
        

    def f(self, X, U, nu):
        delta = X[2] 
        alpha = X[3] 
        v = X[4] 
        a = X[5]
        u0 = U[0]
        u1 = U[1]

        xp1 = np.array([
            np.cos(delta) * v,
            np.sin(delta) * v,
            np.tan(alpha)/self.L  *v,
            u1,
            a,
            1/conf.tau * (-a + u0)
        ]) * self.dt + X + nu

        return xp1

    # def f(self, X, U, nu):
    #     delta = X[2] 
    #     u0 = U[0]
    #     u1 = U[1]

    #     xp1 = np.array([
    #         np.cos(delta) * u0,
    #         np.sin(delta) * u0,
    #         u1
    #     ]) * self.dt + X + nu

    #     return xp1
