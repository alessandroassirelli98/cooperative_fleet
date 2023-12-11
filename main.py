import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from vehicle import Vehicle
from estimator import Estimator
from pid import PID
import utils
from street import Street, Lane
import conf
plt.style.use("seaborn")

def update_animation(frame):
    ax.clear()
    log_xydelta = [v.log_xydelta for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)

    xmin = min(log_xydelta[:, frame, 0])
    xmax= max(log_xydelta[:, frame, 0])
    ax.set(xlim=[xmin-10, xmax+10], ylim=[-7, 7], xlabel='Time [s]', ylabel='Z [m]')
    ax.set_aspect('equal')
    width = conf.L
    height = 2.5
    street.plot_street()
    for i  in range(n_vehicles):
        color = 'C'+str(i)
        scat[i] = plt.gca().add_patch(Rectangle((log_xydelta[i, frame, 0] - width/2,log_xydelta[i, frame, 1] - height/2), width, height,
                    angle=log_xydelta[i, frame, 2]*180/np.pi,
                    facecolor = color,
                    lw=4))
        # scat[i] = plt.plot(log_xydelta[i, frame, 0], log_xydelta[i, frame, 1],
        #                    marker = (3, 0, -90 + log_xydelta[i, frame, 2]*180/np.pi), markersize=10, linestyle='None')
    return scat

n_vehicles = 3
n = 6
T = 300
dt = 0.1
N = int(T/dt)

schedule = []
v_cruise = 10
v_overtake = 25

Q = np.eye(n) * conf.sigma_u**2
Q_arr = []
for i in range(n * n_vehicles):
    Q_arr.append(conf.sigma_u**2)
Q_arr = np.diag(Q_arr)

street = Street(0, 0, 4500, 0)
lanes = street.lanes

vehicles_list = [Vehicle(street, lanes[0], 0, v_cruise, dt)]
[vehicles_list.append(Vehicle(street, lanes[0], vehicles_list[i].x - conf.r/2, v_cruise, dt)) for i in range(n_vehicles-1)]
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


times = np.arange(0, T, dt)
freq = 0.36
u_first_vehicle = np.sin(freq* times)
error = np.zeros((n_vehicles, N))
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
                if np.abs(v.s - vj.s) > conf.lidar_range: continue
                if ordered_vehicles.index(vj) == ordered_vehicles.index(v) - 1:
                    A_SENS[i,j] = 1
                if v.overtaking and  ordered_vehicles.index(vj) < ordered_vehicles.index(v):
                    A_SENS[i,j] = 1

    # Commmunication matrix
    A_COMM = np.zeros([n_vehicles, n_vehicles])
    for i in range(n_vehicles):
        for j in range(i+1, n_vehicles):
            if not np.random.rand() < conf.comm_prob: break
            if np.abs(vehicles_list[i].s - vehicles_list[j].s) < conf.comm_range:
                A_COMM[i,j] = 1
    A_COMM = A_COMM + A_COMM.T
    D = A_COMM @ np.ones(n_vehicles).T

    # Optimal schedule for energy efficiency
    if t==0 : schedule = utils.compute_truck_scheduling(vehicles_list, ordered_vehicles)

    # Compute control actions:
    ordered_u = []
    for i, v in enumerate(vehicles_list):
        u_fwd = 0
        x = S_hat_DKF[i][i * n, t]
        y = S_hat_DKF[i][i * n + 1, t]
        vel = S_hat_DKF[i][i * n + 4, t]
        acc = S_hat_DKF[i][i * n + 5, t]

        for j, vj in enumerate(vehicles_list):
            if v == ordered_vehicles[0]: 
                u_fwd = conf.vel_gain * (v_cruise - v.v)
                # u_fwd = u_first_vehicle[t]
                break

            if A_SENS[i,j] == 1:
                Px = P_DKF[i][j*n + 0, j*n+0, t]
                xf = S_hat_DKF[i][j * n + 0, t]
                yf = S_hat_DKF[i][j * n + 1, t]
                vel_f = S_hat_DKF[i][j * n + 4, t]                
                acc_f = S_hat_DKF[i][j * n + 5, t]
                r = np.sqrt((xf-x)**2 + (yf-y)**2)

                # Using real state                        
                e1 = r - conf.r - conf.h * v.v
                e2 = vj.v - v.v -  conf.h * v.a
                e3 = vj.a - v.a - (1/conf.tau * (- v.a + v.u_fwd) * conf.h)
                u_fwd = v.u_fwd + 1/conf.h * ( - v.u_fwd + 
                                  conf.kp * e1 + 
                                  conf.kd * e2 + 
                                  conf.kdd * e3 + vj.u_fwd) * dt
                
                # Using Predictions
                # e1 = r - conf.r - conf.h * v.v
                # e2 = vel_f - vel -  conf.h * acc
                # e3 = acc_f - acc - (1/conf.tau * (- acc + v.u_fwd) * conf.h)
                # u_fwd = v.u_fwd + 1/conf.h * ( - v.u_fwd + 
                #                 conf.kp * e1 + 
                #                 conf.kd * e2 + 
                #                 conf.kdd * e3 + vj.u_fwd) * dt
                
                error[i, t] = e1



                # if np.sqrt(Px) * 4 < 5:
                #     u_fwd = PIDs[i].compute(r - 80, dt)
                # else:
                #     u_fwd = PIDs[i].compute(v_cruise - v.v, dt)

        if v in schedule:
            if x >= schedule[v]:
                u_fwd = conf.vel_gain * (v_overtake - v.v) # Accelerate
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
                    if x > x_max + conf.r: 
                        v.overtaking = False # If the front vehicle has been overtaken, then stop
                        v.change_lane(0)
                        del schedule[v]

        alpha_des = v.compute_steering(x, y)
        omega = PIDs[i].compute(alpha_des - v.alpha, dt)
        ordered_u.append(np.array([u_fwd, omega]))

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
        R_tmp = [conf.sigma_x_gps**2, conf.sigma_y_gps**2, conf.sigma_mag**2, conf.sigma_alpha**2, conf.sigma_v**2]
        for j, vj in enumerate(vehicles_list):
            if A_SENS[i,j] == 1:
                visible_vehicles.append(vj)
                R_tmp.append(conf.sigma_lidar_rho **2)
                R_tmp.append(conf.sigma_lidar_phi **2)
                R_tmp.append(conf.sigma_stereo **2)
                R_tmp.append(conf.sigma_stereo **2)
                # R_tmp.append(conf.sigma_v **2)
                # R_tmp.append(conf.sigma_v **2)
                # pass

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

for i in range(n_vehicles):
    plt.figure()
    plt.title("Vehicle: " + str(i))
    cix1 = np.sqrt(P_DKF[i][0, 0, :])
    cix2 = np.sqrt(P_DKF[i][6, 6, :])
    cix3 = np.sqrt(P_DKF[i][12, 12, :])
    mul = 4
    plt.subplot(3,1,1)
    y = S_hat_DKF[i][0, :]
    y_gt = log[0][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)

    plt.subplot(3,1,2)
    y = S_hat_DKF[i][6, :]
    y_gt = log[1][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)

    plt.subplot(3,1,3)
    y = S_hat_DKF[i][12, :]
    y_gt = log[2][0,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)


for i in range(n_vehicles):
    plt.figure()
    plt.title("Vehicle: " + str(i))
    cix1 = np.sqrt(P_DKF[i][1, 1, :])
    cix2 = np.sqrt(P_DKF[i][7, 7, :])
    cix3 = np.sqrt(P_DKF[i][13, 13, :])
    mul = 4
    plt.subplot(3,1,1)
    y = S_hat_DKF[i][1, :]
    y_gt = log[0][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix1), (y+mul*cix1), color="red", alpha=0.2)
    # plt.ylim((-10,10))

    plt.subplot(3,1,2)
    y = S_hat_DKF[i][7, :]
    y_gt = log[1][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix2), (y+mul*cix2), color="red", alpha=0.2)
    # plt.ylim((-10,10))


    plt.subplot(3,1,3)
    y = S_hat_DKF[i][13, :]
    y_gt = log[2][1,:]
    plt.plot(times, y)
    plt.plot(times, y_gt)
    plt.fill_between(times, (y-mul*cix3), (y+mul*cix3), color="red", alpha=0.2)
    # plt.ylim((-10,10))

plt.show()

if conf.animate:
    scat = []
    fig, ax = plt.subplots()
    ax.set(xlim=[0, street.x_end], ylim=[lanes[0].y_start, lanes[0].y_end], xlabel='Time [s]', ylabel='Z [m]')
    log_xydelta = [v.log_xydelta for v in vehicles_list]
    log_xydelta = np.array(log_xydelta)
    for i  in range(n_vehicles):
        scat.append(plt.plot(log_xydelta[i, 0, 0], log_xydelta[i, 0, 1],
                    marker = (3, 0, log_xydelta[i, 0, 2]*180/np.pi), markersize=20, linestyle='None'))

    ani = animation.FuncAnimation(fig=fig, func=update_animation, frames=N, interval=dt* 500)
    plt.show()



