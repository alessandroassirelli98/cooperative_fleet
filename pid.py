import numpy as np

class PID:
    def __init__ (self, kp=0, kd=0, ki=0, integral_sat=100):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_sat = integral_sat
        self.error = 0
        self.error_prev = 0
        self.error_sum = 0
        self.error_log = []

    def compute(self, error, dt):
        self.error = error
        self.error_log.append(self.error)

        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -self.integral_sat, self.integral_sat)
        self.error_diff = (self.error - self.error_prev)/dt
        self.error_prev = self.error
        return self.kp * self.error + self.kd * self.error_diff + self.ki * self.error_sum