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
T = 350
dt = 0.1
N = int(T/dt)

Q = np.eye(n) * conf.sigma_u**2
Q_arr = []
for i in range(n * n_vehicles):
    Q_arr.append(conf.sigma_u**2)
Q_arr = np.diag(Q_arr)

street = Street(0, 0, 3000, 0)
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





overtaking = False
for t in range(N-1):

    u_vec = []
    gt = []

    A = np.zeros([n_vehicles, n_vehicles])
    for i in range(n_vehicles):
        for j in range(i+1, n_vehicles):
            A[i,j] = 1 if j <= i+1 else 0
    A = A.transpose()
    D = A @ np.ones(n_vehicles).transpose()

    O = utils.order_matrix(vehicles_list)

    sel_vector = np.arange(0, n_vehicles) # Create a vector to permute the states
    initial2orderd = (O @ sel_vector).astype(int)
    orderd2initial = (sel_vector @ O).astype(int)
    ordered_vehicles = [vehicles_list[i] for i in initial2orderd]
    ordered_estimators = [estimators_list[i] for i in initial2orderd]

    # Compute control actions:
    ordered_u = []
    for i, (e,v) in enumerate(zip(ordered_estimators, ordered_vehicles)):
        for j, vj in enumerate(ordered_vehicles):
            if i == 0: 
                vel = 10
                break
            if A[i,j] == 1:
                x, y, d = e.get_vehicle_estimate(v, t)
                xf, yf, df = e.get_vehicle_estimate(vj, t)
                r = np.sqrt((xf-x)**2 + (yf-y)**2)

                vel =  0.1 * (r - 50) 

        omega = 2*(v.compute_steering() - v.delta)
        ordered_u.append(np.array([vel, omega]))


    initial_u = [ordered_u[orderd2initial.tolist().index(i)] for i in range(n_vehicles)]
    initial_u = np.array(initial_u).flatten()

    for i, (e,v) in enumerate(zip(estimators_list, vehicles_list)):
        gt.append(np.array(v.S)) # Store last robot state
        
        if t == 100 and i == 1 and not overtaking:
            overtaking = True
            if t == 100: v.change_lane(1)

        if i == 1 and overtaking:
            x_vehicles = []
            for vj in vehicles_list:
                if vj != v:
                    Px, _, _ = e.get_vehicle_P_estimate(vj, t)
                    x, _, _ = e.get_vehicle_estimate(vj, t)
                    if np.sqrt(Px) * 4 < 5: x_vehicles.append(x) # Store the vehicle position only if it is reasonably sure

            x_max = max(x_vehicles)
            x, _, _ = e.get_vehicle_estimate(v, t) # Ego estimate

            if x > x_max + 120: 
                overtaking=False # If the front vehicle has been overtaken, then stop
                v.change_lane(0)

            omega = initial_u[i*2+1]
            v.update(np.array([20, omega]), np.zeros((Q.shape[0]))) # Accelerate
        
        else:
            v.update(initial_u[ i*2: (i+1)*2], np.zeros((Q.shape[0])))
        
        log[i][:, t+1] = np.array([v.x,v.y])


    GT = np.array(gt).flatten()

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
        e.run_filter(GT, initial_u, nu, eps, t, Q_arr, R, visible_vehicles, visible_estimator)
