import matplotlib.pyplot as plt
import numpy as np
from vehicle import Vehicle
from estimator import Estimator
import utils
from street import Street, Lane
import conf
plt.style.use("seaborn")

n_vehicles = 3
n = 3
T = 25
dt = 0.1
N = int(T/dt)

Q = np.eye(n) * conf.sigma_u**2
Q_arr = []
for i in range(n * n_vehicles):
    Q_arr.append(conf.sigma_u**2)
Q_arr = np.diag(Q_arr)

street = Street(0, 0, 2500, 0)
lanes = street.lanes

vehicles_list = [Vehicle(street, lanes[0], 0, 10, dt)]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].x - 10, 10, dt)) for i in range(n_vehicles-1)]
estimators_list = []
[estimators_list.append(Estimator(v, n, n_vehicles, N)) for v in vehicles_list]

for e in estimators_list: e.set_fleet(vehicles_list)

log = []
for i in range(n_vehicles):
    log.append(np.zeros((2, N)))

for v in vehicles_list:
    log[i][:, 0] = np.array([v.x,v.y])

for t in range(N-1):

    u_vec = []
    gt = []
    for i,v in enumerate(vehicles_list):
        u = np.array([1, np.random.randn()])
        if i == 1 and t>50 and t<100: u = np.array([15, np.random.randn()])
        if i == 2 and t>120: u = np.array([20, np.random.randn()])
        u_vec.append(u)
        gt.append(np.array(v.S)) # Store last robot state

        v.update(u, np.zeros((Q.shape[0])))
        log[i][:, t+1] = np.array([v.x,v.y])

    A = np.zeros([n_vehicles, n_vehicles])
    for i in range(n_vehicles):
        for j in range(i+1, n_vehicles):
            A[i,j] = 1 if j <= i+1 else 0
    A = A.transpose()
    D = A @ np.ones(n_vehicles).transpose()

    O = utils.order_matrix(vehicles_list)

    sel_vector = np.arange(0, n_vehicles) # Create a vector to permute the states
    sel_vector = (O @ sel_vector).astype(int)
    ordered_vehicles = [vehicles_list[i] for i in sel_vector]
    ordered_estimators = [estimators_list[i] for i in sel_vector]

    GT = np.array(gt).flatten()
    u_array = np.array(u_vec).flatten()

    for i, e in enumerate(ordered_estimators):
        visible_vehicles = []
        visible_estimator = []
        R_tmp = [conf.sigma_x_gps**2, conf.sigma_y_gps**2, conf.sigma_mag**2]
        for j, vj in enumerate(ordered_vehicles):
            if A[i,j] == 1:
                ej = ordered_estimators[j]
                visible_vehicles.append(vj)
                visible_estimator.append(ej)
                R_tmp.append(conf.sigma_radar **2)
                R_tmp.append(conf.sigma_radar **2)

        R = np.diag(R_tmp)
        nu = np.random.multivariate_normal(np.zeros((Q_arr.shape[0])), Q_arr).T
        eps = np.random.multivariate_normal(np.zeros((R.shape[0])), R).T
        e.run_filter(GT, u_array, nu, eps, t, Q_arr, R, visible_vehicles, visible_estimator)
