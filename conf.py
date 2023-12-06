import numpy as np

animate = False

# String stability analysis
kp = 0.2
kd = 0.7
kdd = 0

pid_kp = 0.1
pid_kd = 0.1
pid_ki = 0.1

tau = 0.1
L = 5

h = 0.7
r = 30

v_max = 20
a_max = 60

comm_range = 100
lidar_range = 250
comm_prob = 0.9

P_gps = 0.5
P_radar = 0.0

### Measurement uncertainties
sigma_y = 1e-2
sigma_delta = 1e-2
sigma_alpha = 1e-2
sigma_v = 1e-1
sigma_a = 1e-1
sigma_mag = 1e-2
sigma_beta = 1e-2
sigma_x_gps = 1e-2
sigma_y_gps = 1e-2
sigma_radar = 1e-1
sigma_lidar_rho = 1e-1
sigma_lidar_phi = 1e-2

sigma_u = 1e-2
sigma_omega = 1e-3

PATH_TOL = 10