import casadi as cas
import conf
import numpy as np
from vehicle import Vehicle
import cvxpy as cp

def sys_model(dt, n_vehicles, n, S_sym, U_sym, Nu_sym):
    tmp_f = []
    for i in range(n_vehicles):
        delta_other = S_sym[n * i + 2]
        alpha_other = S_sym[n * i + 3]
        v_other = S_sym[n * i + 4]
        a_other = S_sym[n * i + 5]
        u_other = U_sym[2 * i]
        omega_other = U_sym[2 * i + 1]
        tmp_f.append(cas.vertcat(cas.cos(delta_other) * v_other,
                                cas.sin(delta_other) * v_other,
                                cas.tan(alpha_other)/conf.L * v_other,
                                omega_other,
                                a_other,
                                1/conf.tau * (-a_other + (u_other)))
                                * dt + S_sym[i*n: n + (i)*n] + Nu_sym[i*n: n + (i)*n])
    f = []
    for e in tmp_f: f = cas.vertcat(f, e)
    return f


def order_matrix(vehicles_list):
    order_vehicles = []
    for v in vehicles_list:
        order_vehicles.append(v)
    order_vehicles.sort(key=lambda x: x.s, reverse=True)
    
    O = np.zeros((len(vehicles_list), len(vehicles_list)))
    for i in range(len(vehicles_list)):
        idx = vehicles_list.index(order_vehicles[i])
        O[i, idx] = 1
    return O


def compute_truck_scheduling(vehicles_list, ordered_vehicles, store=False):
        # The scheduling is computed in the street reference frame
        c0s = [v.c0 for v in vehicles_list]
        c1s = [v.c1 for v in vehicles_list]
        ls = [v.life for v in vehicles_list]
        S = max([v.street.length - v.x for v in vehicles_list])
        schedule = {}
        if S > 0:
            n = len(vehicles_list)
            A = np.zeros((2*n, 2*n))

            for j in range(2*n):
                A[0, j] = 1 if (j % 2 == 0  and j + 2 != 2*n) else 0
                A[0, -1] = -1

            for i in range(n - 1):
                for j in range(2 * n):
                    if j == 2*i + 1:
                        A[i + 1,j] = -1
                    elif j % 2 == 0 and j != 2*i:
                        A[i + 1,j] = 1

            for i in range(0, n):
                for j in range(2*i, 2*n):
                    A[n + i, j : j + 2] = np.array([1,1])
                    break

            P = np.zeros((n, 2* n))
            for i in range(0, n):
                for j in range(2*i, 2*n):
                    P[i, j : j + 2] = np.array([c0s[i] + c1s[i], c0s[i]])
                    break

            b = np.zeros((2*n))
            b[n:] = np.ones(n) * S

            q = np.concatenate([np.array(ls), np.zeros(n)])
            G = np.eye(2*n)*-1
            h = np.zeros(2*n)

            x = cp.Variable(2*n)
            obj = cp.Minimize(0.5*cp.sum_squares(cp.diff(P@x)))
            constraints = [A@x == b, G@x <= h]
            prob = cp.Problem(obj, constraints)
            sol = prob.solve()

            if store:
                # Calculation withoptimized values
                ls = np.array(ls)
                consumptions_opt = P@x.value
                total_consumption_opt = sum(consumptions_opt)
                final_autonomy_opt = ls - consumptions_opt
                np.savez('optimized', consumption=consumptions_opt, ls=ls)

                # If no optimization was used
                x_not_opt = np.zeros((2*len(vehicles_list)))
                x_not_opt[0] = S
                for i in range(2, 2 * len(vehicles_list)):
                    if i%2 != 0: x_not_opt[i] = S

                consumptions = P@x_not_opt
                total_consumption = sum(consumptions)
                final_autonomy = ls - consumptions
                np.savez('not_optimized', consumption=consumptions, ls=ls)





            # opti = cas.Opti()
            # U = opti.variable(n*2)

            # # life = q - P @ U

            # opti.subject_to( A @ U - b == 0)
            # opti.subject_to( U >= h)

            # cost = 0.5*cas.sumsqr(cas.diff(P@U))
            # opti.minimize(cost)
            # # opti.minimize(cas.sumsqr(P @ U)) 

            # p_opts = {"expand":True}
            # s_opts = {"max_iter": 10000}
            # opti.solver('ipopt', p_opts, s_opts)
            # sol = opti.solve()

            # Obtain a sequence
            # [S_head, S_in_queue] for each vehicle
            u = np.array(np.split(np.round(x.value,0),n))

            # not_optimal_u = np.array([S,0])
            # not_optimal_u = np.concatenate([not_optimal_u, np.array([0, S] * (n-1))])
            # np.savetxt('not_optimal_life.out', q - P @ not_optimal_u, delimiter=',')
            # np.savetxt('optimal_life.out', sol.value(life), delimiter=',')

                
            # Store the scheduling for each vehicle
            # Starting from the actual leading vehicle to the last one
            # The schedule must be shifted in order to consider the actual leading vehicle
            # As the first vehicle executing the leader role of the platoon

            # {v0: [None, S_head, S_in_queue], v1: [S_overtaking, S_head, S_in_queue], ...}
            # For the actual leader there is no need to overtake
            # The others must do it at the right moment

            # If you want the vehicles ordered by the amount of street they have to perform as leader
            # decreasing_order = u[:, 1].argsort()

            # for i, ve in enumerate(vehicles_list):
            #     if ve == ordered_vehicles[0]:
            #         target = i
            # shift = np.where(decreasing_order == target)[0][0]
            # order = np.roll(decreasing_order, -shift)
            # u = u[order]

            # vehicles = np.array([v for v in vehicles_list])
            # vehicles = vehicles[order]

            # s = 0
            # for i,v in enumerate(vehicles):
            #     if i == 0:
            #         schedule[v] = [0, u[i]]
            #     else:
            #         schedule[v] = [s, u[i]]
            #     s += u[i][0]

            # s = 0
            # for i,v in enumerate(vehicles):
            #     if i!=0: schedule[v] = [s]
            #     s += u[i][0]

            # Just keep actual order
            s = 0
            for i,v in enumerate(ordered_vehicles):
                if i!=0: schedule[v] = [s]
                s += u[i][0]


        #     self.set_schedule(schedule)

        return schedule
            
        # else:
        #     if S <= 0:
        #         print("No scheduling needed, the road is finished")
        #     else:
        #         print("Cannot compute scheduling, no leader")   
        #     schedule = {"overtaking": None}
        #     self.set_schedule(schedule)