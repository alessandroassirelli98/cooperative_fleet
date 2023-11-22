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

v_cruise = 10

Q = np.eye(n) * conf.sigma_u**2
Q_arr = []
for i in range(n * n_vehicles):
    Q_arr.append(conf.sigma_u**2)
Q_arr = np.diag(Q_arr)

street = Street(0, 0, 4000, 0)
lanes = street.lanes

vehicles_list = [Vehicle(street, lanes[0], 0, v_cruise, dt)]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].x - 10, v_cruise, dt)) for i in range(n_vehicles-1)]
vehicles_list[0].c1 = 100/1000

estimators_list = []
[estimators_list.append(Estimator(v, n, n_vehicles, N)) for v in vehicles_list]

for e in estimators_list: e.set_fleet(vehicles_list)

log = []
for i in range(n_vehicles):
    log.append(np.zeros((2, N)))

for v in vehicles_list:
    log[i][:, 0] = np.array([v.x,v.y])

S_hat_distributed = []
P_distributed = []
for i in range(n_vehicles):
    S_hat_distributed.append(np.zeros((n * n_vehicles, N)))
    P_distributed.append(np.zeros((n * n_vehicles, n * n_vehicles, N)))

    P_distributed[i][:,:,0] = np.eye(n * n_vehicles) * 10

for t in range(N-1):

    u_vec = []
    gt = []



    # Order of vehicles in the platoon
    O = utils.order_matrix(vehicles_list)
    sel_vector = np.arange(0, n_vehicles) # Create a vector to permute the states
    initial2orderd = (O @ sel_vector).astype(int)
    ordered_vehicles = [vehicles_list[i] for i in initial2orderd]
    ordered_estimators = [estimators_list[i] for i in initial2orderd]
    ordered_vehicles[0].lead = True # Tell the vehicle it is the first, for battery purposes

    # Sensing matrix
    A_SENS = np.zeros([n_vehicles, n_vehicles])
    for i,v in enumerate(vehicles_list):
        for j,vj in enumerate(vehicles_list):
            if ordered_vehicles.index(vj) == ordered_vehicles.index(v) - 1:
                A_SENS[i,j] = 1 

    # Commmunication matrix
    A_COMM = np.zeros([n_vehicles, n_vehicles])
    for i in range(n_vehicles):
        for j in range(i+1, n_vehicles):
            # A_COMM[i,j] = 1 if np.random.rand() < 0.5 else 0
            A_COMM[i,j] = 1 if np.abs(vehicles_list[i].s - vehicles_list[j].s) < 200 else 0
    A_COMM = A_COMM + A_COMM.T
    # A_COMM = A_SENS + A_SENS.T
    D = A_COMM @ np.ones(n_vehicles).T

    # Optimal schedule for energy efficiency
    if t==0 : schedule = utils.compute_truck_scheduling(vehicles_list, ordered_vehicles)

    # Compute control actions:
    ordered_u = []
    for i, v in enumerate(vehicles_list):
        x = S_hat_distributed[i][i * n, t-1]
        y = S_hat_distributed[i][i * n + 1, t-1]
        for j, vj in enumerate(vehicles_list):
            if v == ordered_vehicles[0]: 
                vel = v_cruise
                break

            if A_SENS[i,j] == 1:
                Px = P_distributed[i][j*n + 0, j*n+0, t-1]
                xf = S_hat_distributed[i][j * n + 0, t-1]
                yf = S_hat_distributed[i][j * n + 1, t-1]
                r = np.sqrt((xf-x)**2 + (yf-y)**2)

                vel =  0.1 * (r - 50) if np.sqrt(Px) * 4 < 5 else v_cruise

        if v in schedule:
            if x >= schedule[v]:
                vel = 20 # Accelerate
                v.overtaking = True
                if v.which_lane == 0 : v.change_lane(1)
                x_vehicles = []
                for j, vj in enumerate(vehicles_list):
                    if vj != v:
                        Px = P_distributed[i][j * n + 0, j * n+0, t-1].copy()
                        xf = S_hat_distributed[i][j * n + 0, t-1].copy()
                        if np.sqrt(Px) * 4 < 5: 
                            x_vehicles.append(xf) # Store the vehicle position only if it is reasonably sure

                if len(x_vehicles) != 0: 
                    x_max = max(x_vehicles) 
                    if x > x_max + 100: 
                        v.overtaking = False # If the front vehicle has been overtaken, then stop
                        v.change_lane(0)
                        del schedule[v]
         
        omega = 2*(v.compute_steering(x, y) - v.delta)
        ordered_u.append(np.array([vel, omega]))

    initial_u = [ordered_u[i] for i in range(n_vehicles)]
    initial_u = np.array(initial_u).flatten()

    F = []
    a = []
    for i in range(n_vehicles):
        F.append(np.zeros((n * n_vehicles, n* n_vehicles)))
        a.append(np.zeros((n* n_vehicles)))

    for i, (e,v) in enumerate(zip(estimators_list, vehicles_list)):
        gt.append(np.array(v.S)) # Store last robot state
        v.update(initial_u[ i*2: (i+1)*2], np.zeros((Q.shape[0])))
        log[i][:, t+1] = np.array([v.x, v.y])


    GT = np.array(gt).flatten()

    for i, e in enumerate(estimators_list):
        visible_vehicles = []
        visible_estimator = []
        R_tmp = [conf.sigma_x_gps**2, conf.sigma_y_gps**2, conf.sigma_mag**2]
        for j, vj in enumerate(vehicles_list):
            if A_SENS[i,j] == 1:
                ej = estimators_list[j]
                visible_vehicles.append(vj)
                visible_estimator.append(ej)

        R = np.diag(R_tmp)
        nu = np.random.multivariate_normal(np.zeros((Q_arr.shape[0])), Q_arr).T
        eps = np.random.multivariate_normal(np.zeros((R.shape[0])), R).T
        e.run_filter(GT, initial_u, nu, eps, t, Q_arr, R, visible_vehicles, visible_estimator)

    for i,(e,v) in enumerate(zip(estimators_list, vehicles_list)):

        H = np.eye(n * n_vehicles)

        zi = e.S_hat[:, t].copy()
        P = e.P[:,:, t].copy()
        F[i] = H.T @ np.linalg.inv(H @ P @ H.T) @ H
        a[i] = H.T @ np.linalg.inv(H @ P @ H.T) @ zi

    m=100
    for _ in range(m):
        Fstore = F.copy()
        aStore = a.copy()
        for i in range(n_vehicles):
            for j in range(n_vehicles):
                if A_COMM[i,j] == 1:
                    F[i] = F[i] + 1/(1+max(D[i], D[j])) * (Fstore[j] - Fstore[i])
                    a[i] = a[i] + 1/(1+max(D[i], D[j])) * (aStore[j] - aStore[i])
    
    for i in range(n_vehicles):
        S_hat_distributed[i][:, t] = np.linalg.inv(F[i]) @ a[i]
        P_distributed[i][:,:, t] = np.linalg.inv(F[i])


plt.figure()
for i in range(n_vehicles):
    plt.plot(log[i][0,1:])

times = np.arange(0,N)
# for i in range(n_vehicles):
#     plt.figure()
#     plt.title("Vehicle: " + str(i))
#     cix1 = np.sqrt(P_distributed[i][1, 1, :])
#     cix2 = np.sqrt(P_distributed[i][4, 4, :])
#     cix3 = np.sqrt(P_distributed[i][7, 7, :])
#     mul = 4
#     plt.subplot(3,1,1)
#     y = S_hat_distributed[i][1, :]
#     y_gt = log[0][1,:]
#     plt.plot(times, y)
#     plt.plot(times, y_gt)
#     plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)

#     plt.subplot(3,1,2)
#     y = S_hat_distributed[i][4, :]
#     y_gt = log[1][1,:]
#     plt.plot(times, y)
#     plt.plot(times, y_gt)
#     plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)

#     plt.subplot(3,1,3)
#     y = S_hat_distributed[i][7, :]
#     y_gt = log[2][1,:]
#     plt.plot(times, y)
#     plt.plot(times, y_gt)
#     plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)



for i in range(n_vehicles):
    plt.figure()
    plt.title("Vehicle: " + str(i))
    cix1 = np.sqrt(P_distributed[i][0, 0, :])
    cix2 = np.sqrt(P_distributed[i][3, 3, :])
    cix3 = np.sqrt(P_distributed[i][6, 6, :])
    mul = 4
    plt.subplot(3,1,1)
    y = S_hat_distributed[i][0, :]
    y_gt = log[0][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)

    plt.subplot(3,1,2)
    y = S_hat_distributed[i][3, :]
    y_gt = log[1][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)

    plt.subplot(3,1,3)
    y = S_hat_distributed[i][6, :]
    y_gt = log[2][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)



plt.show()