import numpy as np
import ast

err_threshold = 1e-10


def ssq(j, i, x_prime_sum, x_prime_sum_sq):
    if j > 0:
        muji = (x_prime_sum[i] - x_prime_sum[j - 1]) / (i - j + 1)
        a = muji[0]
        b = muji[1]
        sji = x_prime_sum_sq[i] - x_prime_sum_sq[j - 1] - (i - j + 1) * np.array([a**2, 2*a*b, b**2])
    else:
        muji = (x_prime_sum[i]) / (i - j + 1)
        a = muji[0]
        b = muji[1]
        sji = x_prime_sum_sq[i] - (i - j + 1) * np.array([a ** 2, 2 * a * b, b ** 2])

    return sji


def check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
    if f_tau_t[0] < min_quad_term:
        return True
    elif f_tau_t[0] == min_quad_term:
        if f_tau_t[1] > min_quad_term_funct_cost[1][1]:
            return True
        elif f_tau_t[1] == min_quad_term_funct_cost[1][1]:
            if f_tau_t[2] < min_quad_term_funct_cost[1][2]:
                return True

    return False


def check_zero(value):
    if - err_threshold <= value <= err_threshold:
        return 0

    return value


def check_coef_zero(f_tau_t):
    if - err_threshold <= f_tau_t[0] <= err_threshold:
        f_tau_t[0] = 0

    if - err_threshold <= f_tau_t[1] <= err_threshold:
        f_tau_t[1] = 0

    if - err_threshold <= f_tau_t[2] <= err_threshold:
        f_tau_t[2] = 0

    return f_tau_t


def quadratic_solver(a, b, c):
    delta = b**2 - 4*a*c

    if delta < 0:
        return None, None

    sqrt_delta = np.sqrt(delta)
    x_1 = (-b - sqrt_delta) / (2*a)
    x_2 = (-b + sqrt_delta) / (2*a)

    if x_1 <= x_2:
        return x_1, x_2
    else:
        return x_2, x_1


def quadratic_min(a, b, c):
    if a == 0:
        if b != 0:
            print('ERROR')
        else:
            return 0, c
    else:
        min_z = (-b) / (2 * a)
        min_f = a * (min_z ** 2) + b * min_z + c
        return min_z, min_f


def second_pruning(list_remove_funct_cand, list_z, list_opt_funct_corres_to_z, beta):
    # for key, funct in list_remove_funct_cand.items():
    #     print(funct)
    # print('LIST Z', list_z)
    # print('opt', list_opt_funct_corres_to_z[0][1])
    # print('beta', beta)

    temp_list_remove_funct_cand = list_remove_funct_cand.copy()

    list_remove_key = []
    z_curr = list_z[0]
    funct_curr_key = list_opt_funct_corres_to_z[0][0]
    funct_curr = list_opt_funct_corres_to_z[0][1]

    new_temp_list_remove_funct_cand = temp_list_remove_funct_cand.copy()
    for key, funct in temp_list_remove_funct_cand.items():
        if funct[0] < funct_curr[0]:
            del new_temp_list_remove_funct_cand[key]
        elif funct[0] == funct_curr[0]:
            if funct[1] > funct_curr[1]:
                del new_temp_list_remove_funct_cand[key]
            elif funct[1] == funct_curr[1]:
                if (funct[2] - beta) < funct_curr[2]:
                    del new_temp_list_remove_funct_cand[key]
                elif (funct[2] - beta) >= funct_curr[2]:
                    list_remove_key.append(key)
                    del new_temp_list_remove_funct_cand[key]

    temp_list_remove_funct_cand = new_temp_list_remove_funct_cand.copy()

    for i in range(1, len(list_z)):
        new_temp_list_remove_funct_cand = temp_list_remove_funct_cand.copy()
        next_z_curr = list_z[i]

        for key, funct in temp_list_remove_funct_cand.items():
            a = funct[0] - funct_curr[0]
            b = funct[1] - funct_curr[1]
            c = funct[2] - funct_curr[2] - beta

            # a = check_zero(a)
            # b = check_zero(b)
            # c = check_zero(c)

            if a == 0:
                if b == 0:
                    if c < 0:
                        print('error')

                    del new_temp_list_remove_funct_cand[key]
                    list_remove_key.append(key)
                else:
                    x_1 = x_2 = - c / b
                    if x_1 <= z_curr:
                        del new_temp_list_remove_funct_cand[key]
                        list_remove_key.append(key)
                    else:
                        if x_1 <= next_z_curr:
                            del new_temp_list_remove_funct_cand[key]

            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):
                    del new_temp_list_remove_funct_cand[key]
                    list_remove_key.append(key)

                else:
                    if x_2 <= z_curr:
                        del new_temp_list_remove_funct_cand[key]
                        list_remove_key.append(key)
                    elif (x_1 <= next_z_curr) or (x_2 <= next_z_curr):
                        del new_temp_list_remove_funct_cand[key]

        temp_list_remove_funct_cand = new_temp_list_remove_funct_cand.copy()
        z_curr = next_z_curr
        funct_curr = list_opt_funct_corres_to_z[i][1]

    return list_remove_key

def find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost):
    tempt_f_cost_dict = local_f_cost_dict.copy()
    opt_funct_set = {}
    list_remove_funct = {}

    z_curr = np.NINF
    funct_curr_key = min_quad_term_funct_cost[0]
    funct_curr = min_quad_term_funct_cost[1]

    list_z = [z_curr]
    list_opt_funct_corres_to_z = [[funct_curr_key, funct_curr]]

    opt_funct_set.update({funct_curr_key: funct_curr})

    while len(tempt_f_cost_dict) > 1:
        new_funct_set = tempt_f_cost_dict.copy()
        list_new_z = []

        for key, funct in tempt_f_cost_dict.items():
            if np.array_equal(funct, funct_curr):
                if key != funct_curr_key:
                    list_remove_funct.update({key: funct})
                    del new_funct_set[key]

                continue

            a = funct[0] - funct_curr[0]
            b = funct[1] - funct_curr[1]
            c = funct[2] - funct_curr[2]

            a = check_zero(a)
            b = check_zero(b)
            c = check_zero(c)

            if a == 0:
                if b == 0:
                    # if c < 0:
                    #     print('error')

                    if key not in opt_funct_set:
                        list_remove_funct.update({key: funct})

                    del new_funct_set[key]
                else:
                    x_1 = x_2 = - c / b
                    if x_1 <= z_curr:
                        if key not in opt_funct_set:
                            list_remove_funct.update({key: funct})

                        del new_funct_set[key]
                    else:
                        list_new_z.append([x_1, key])
            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):

                    if key not in opt_funct_set:
                        list_remove_funct.update({key: funct})

                    del new_funct_set[key]
                else:
                    if x_2 <= z_curr:
                        if key not in opt_funct_set:
                            list_remove_funct.update({key: funct})

                        del new_funct_set[key]
                    elif x_1 <= z_curr < x_2:
                        list_new_z.append([x_2, key])
                    elif z_curr < x_1:
                        list_new_z.append([x_1, key])

        if len(list_new_z) == 0:
            break

        sorted_list_new_z = sorted(list_new_z)

        z_curr = sorted_list_new_z[0][0]
        funct_curr_key = sorted_list_new_z[0][1]
        funct_curr = tempt_f_cost_dict[funct_curr_key]


        list_z.append(z_curr)
        list_opt_funct_corres_to_z.append([funct_curr_key, funct_curr])
        opt_funct_set.update({funct_curr_key: funct_curr})

        tempt_f_cost_dict = new_funct_set.copy()

    return opt_funct_set, list_remove_funct, list_z, list_opt_funct_corres_to_z


def pelt_perturbed_x(x_prime, sg_results, n, beta):
    # data_1 = []
    # data_2 = []

    opt_funct_set = None
    sum_x_prime = []
    sum_x_prime_sq = []

    global_f_cost_dict = {str([-1]): np.array([0, 0, -beta])}
    T_curr = [[-1]]

    for i in range(n):
        a = x_prime[i][0]
        b = x_prime[i][1]
        
        if i == 0:
            sum_x_prime.append(np.array([a, b]))
            sum_x_prime_sq.append(np.array([a ** 2, 2 * a * b, b ** 2]))
        else:
            sum_x_prime.append(sum_x_prime[i - 1] + np.array([a, b]))
            sum_x_prime_sq.append(sum_x_prime_sq[i - 1] + np.array([a ** 2, 2 * a * b, b ** 2]))

    for t in range(n):
        local_f_cost_dict = {}

        min_quad_term = np.Inf
        min_quad_term_funct_cost = None

        for tau in T_curr:
            if str(tau) not in global_f_cost_dict:
                f_tau = global_f_cost_dict[str(tau[:-1])] + ssq(tau[-2] + 1, tau[-1], sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])
                # print('added', str(tau), f_tau)
                global_f_cost_dict.update({str(tau): f_tau})

            if tau == [-1]:
                f_tau_t = ssq(0, t, sum_x_prime, sum_x_prime_sq)
                # print(f_tau_t)
                f_tau_t = check_coef_zero(f_tau_t)
                local_f_cost_dict.update({str(tau): f_tau_t})

                if check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
                    min_quad_term = f_tau_t[0]
                    min_quad_term_funct_cost = [str(tau), f_tau_t]
            else:
                f_tau_t = global_f_cost_dict[str(tau)] + ssq(tau[-1] + 1, t, sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])
                # print(f_tau_t)
                f_tau_t = check_coef_zero(f_tau_t)
                local_f_cost_dict.update({str(tau): f_tau_t})

                if check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
                    min_quad_term = f_tau_t[0]
                    min_quad_term_funct_cost = [str(tau), f_tau_t]

        opt_funct_set, list_remove_funct, list_z, list_opt_funct_corres_to_z = \
            find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost)

        T_new = T_curr[:]

        list_remove_key = second_pruning(list_remove_funct, list_z, list_opt_funct_corres_to_z, beta)

        for key in list_remove_key:
            key_list_type = ast.literal_eval(key)
            T_new.remove(key_list_type)

        for key, funct in opt_funct_set.items():
            key_list_type = ast.literal_eval(key)
            key_list_type.append(t)
            new_tau = key_list_type
            T_new.append(new_tau)

        T_curr = T_new[:]

    opt_funct = np.array([0, 0, - beta])
    for i in range(len(sg_results) - 1):
        curr_cp = sg_results[i]
        next_cp = sg_results[i + 1]
        opt_funct = opt_funct + ssq(curr_cp, next_cp - 1, sum_x_prime, sum_x_prime_sq) + np.array([0, 0, beta])

    return opt_funct_set, opt_funct
