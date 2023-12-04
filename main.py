import matplotlib.pyplot as plt
import numpy as np
from vehicle import Vehicle
from estimator import Estimator
from pid import PID
import utils
from street import Street, Lane
import conf
plt.style.use("seaborn")

n_vehicles = 3
n = 3
T = 350
dt = 0.1
N = int(T/dt)

schedule = []
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
PIDs = []
[estimators_list.append(Estimator(v, n, n_vehicles, N)) for v in vehicles_list]
[PIDs.append(PID(conf.pid_kp, conf.pid_kd, conf.pid_ki)) for _ in range(n_vehicles)]
for e in estimators_list: e.set_fleet(vehicles_list)

log = []
for i in range(n_vehicles):
    log.append(np.zeros((2, N)))

for v in vehicles_list:
    log[i][:, 0] = np.array([v.x,v.y])

S_hat_DKF = []
P_DKF = []
for i in range(n_vehicles):
    S_hat_DKF.append(np.zeros((n * n_vehicles, N)))
    P_DKF.append(np.zeros((n * n_vehicles, n * n_vehicles, N)))
    P_DKF[i][:,:,0] = np.eye(n * n_vehicles) * 10

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
            if vj != v and v != ordered_vehicles[0]: 
                if ordered_vehicles.index(vj) == ordered_vehicles.index(v) - 1:
                    A_SENS[i,j] = 1

    # Commmunication matrix
    A_COMM = np.zeros([n_vehicles, n_vehicles])
    for i in range(n_vehicles):
        for j in range(i+1, n_vehicles):
            if not np.random.rand() < conf.comm_prob: break
            # if t<1000 or t>1500: break
            if np.abs(vehicles_list[i].s - vehicles_list[j].s) < conf.comm_range:
                A_COMM[i,j] = 1
    A_COMM = A_COMM + A_COMM.T
    D = A_COMM @ np.ones(n_vehicles).T

    # Optimal schedule for energy efficiency
    if t==0 : schedule = utils.compute_truck_scheduling(vehicles_list, ordered_vehicles)

    # Compute control actions:
    ordered_u = []
    for i, v in enumerate(vehicles_list):
        x = S_hat_DKF[i][i * n, t]
        y = S_hat_DKF[i][i * n + 1, t]
        for j, vj in enumerate(vehicles_list):
            if v == ordered_vehicles[0]: 
                vel = v_cruise
                break

            if A_SENS[i,j] == 1:
                Px = P_DKF[i][j*n + 0, j*n+0, t]
                xf = S_hat_DKF[i][j * n + 0, t]
                yf = S_hat_DKF[i][j * n + 1, t]
                r = np.sqrt((xf-x)**2 + (yf-y)**2)

                if np.sqrt(Px) * 4 < 5:
                    vel = PIDs[i].compute(r - 150, dt)
                else:
                    vel = v_cruise

        if v in schedule:
            if x >= schedule[v]:
                vel = 20 # Accelerate
                v.overtaking = True
                if v.which_lane == 0 : v.change_lane(1)
                x_vehicles = []
                for j, vj in enumerate(vehicles_list):
                    if vj != v:
                        Px = P_DKF[i][j * n + 0, j * n+0, t].copy()
                        xf = S_hat_DKF[i][j * n + 0, t].copy()
                        if np.sqrt(Px) * 4 < 5: 
                            x_vehicles.append(xf) # Store the vehicle position only if it is reasonably sure

                if len(x_vehicles) != 0: 
                    x_max = max(x_vehicles) 
                    if x > x_max + 100: 
                        v.overtaking = False # If the front vehicle has been overtaken, then stop
                        v.change_lane(0)
                        del schedule[v]
         
        omega = (v.compute_steering(x, y) - v.delta)
        ordered_u.append(np.array([vel, omega]))

    initial_u = [ordered_u[i] for i in range(n_vehicles)]
    initial_u = np.array(initial_u).flatten()

    F = []
    a = []
    for i in range(n_vehicles):
        F.append(np.zeros((n * n_vehicles, n* n_vehicles)))
        a.append(np.zeros((n* n_vehicles)))

    for i, v in enumerate(vehicles_list):
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
                visible_vehicles.append(vj)
                R_tmp.append(conf.sigma_radar **2)
                R_tmp.append(conf.sigma_radar **2)
                pass

        R = np.diag(R_tmp)
        nu = np.random.multivariate_normal(np.zeros((Q_arr.shape[0])), Q_arr).T
        eps = np.random.multivariate_normal(np.zeros((R.shape[0])), R).T
        e.run_filter(GT, initial_u, nu, eps, t, Q_arr, R, visible_vehicles)
        
        
        
        zi = e.h_fun(GT).full().flatten() + eps
        H = e.H_fun(S_hat_DKF[i][:,t]).full()

        F[i] = H.T @ np.linalg.inv(R.copy()) @ H
        a[i] = H.T @ np.linalg.inv(R.copy()) @ zi


    m=10
    for _ in range(m):
        Fstore = F.copy()
        aStore = a.copy()
        for i in range(n_vehicles):
            for j in range(n_vehicles):
                if A_COMM[i,j] == 1:
                    F[i] = F[i] + 1/(1+max(D)) * (Fstore[j] - Fstore[i])
                    a[i] = a[i] + 1/(1+max(D)) * (aStore[j] - aStore[i])
    
    for i, (e,v) in enumerate(zip(estimators_list, vehicles_list)):
        G = e.G_fun(S_hat_DKF[i][:,t], initial_u).full()
        A = e.A_fun(S_hat_DKF[i][:,t], initial_u, nu*0).full()
        H = e.H_fun(S_hat_DKF[i][:,t+1]).full()
        nu = np.random.multivariate_normal(np.zeros((Q_arr.shape[0])), Q_arr).T

        pred = e.f_fun(S_hat_DKF[i][:,t], initial_u, nu).full().flatten()
        
        M = np.linalg.inv(np.linalg.inv(P_DKF[i][:,:, t]) + F[i])
        S_hat_DKF[i][:, t+1] = pred + M @ (a[i]- F[i] @ S_hat_DKF[i][:, t])
        P_DKF[i][:,:, t+1] = A @ M @ A.T + G @ Q_arr @ G.T



plt.figure()
for i in range(n_vehicles):
    plt.plot(log[i][0,1:])

times = np.arange(0,N)
for i in range(n_vehicles):
    plt.figure()
    plt.title("Vehicle: " + str(i))
    cix1 = np.sqrt(P_DKF[i][0, 0, :])
    cix2 = np.sqrt(P_DKF[i][3, 3, :])
    cix3 = np.sqrt(P_DKF[i][6, 6, :])
    mul = 4
    plt.subplot(3,1,1)
    y = S_hat_DKF[i][0, :]
    y_gt = log[0][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)

    plt.subplot(3,1,2)
    y = S_hat_DKF[i][3, :]
    y_gt = log[1][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)

    plt.subplot(3,1,3)
    y = S_hat_DKF[i][6, :]
    y_gt = log[2][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)


for i in range(n_vehicles):
    plt.figure()
    plt.title("Vehicle: " + str(i))
    cix1 = np.sqrt(P_DKF[i][1, 1, :])
    cix2 = np.sqrt(P_DKF[i][4, 4, :])
    cix3 = np.sqrt(P_DKF[i][7, 7, :])
    mul = 4
    plt.subplot(3,1,1)
    y = S_hat_DKF[i][1, :]
    y_gt = log[0][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)

    plt.subplot(3,1,2)
    y = S_hat_DKF[i][4, :]
    y_gt = log[1][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)

    plt.subplot(3,1,3)
    y = S_hat_DKF[i][7, :]
    y_gt = log[2][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)

plt.show()