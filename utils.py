import casadi as cas
import conf
import numpy as np

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