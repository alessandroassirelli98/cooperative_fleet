import numpy as np

animate = True
optimal_schedule = True

# String stability analysis
kp = 0.1
kd = 0.7
kdd = 0

pid_kp = 5
pid_kd = 0.5
pid_ki = 0.0

vel_gain = 0.5

tau = 0.1
L = 8

h = 0.8
r = 50

v_max = 20
a_max = 60

comm_range = 200
radar_range = 150
comm_prob = 0.3

mul = 3
sigma_thr = 5


### Measurement uncertainties
sigma_alpha = 1e-2
sigma_v = 1e-1
sigma_mag = 1e-1
sigma_x_gps = 1e-2
sigma_y_gps = 1e-2
sigma_radar_rho = 5*1e-2
sigma_radar_phi = 5*1e-2
sigma_stereo = 1e-2
sigma_d = 1e-2


sigma_u = 1e-2

PATH_TOL = 10