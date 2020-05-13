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

def find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost):

    tempt_f_cost_dict = local_f_cost_dict.copy()
    opt_funct_set = {}
    z_curr = np.NINF
    funct_curr_key = min_quad_term_funct_cost[0]
    funct_curr = min_quad_term_funct_cost[1]

    opt_funct_set.update({funct_curr_key: funct_curr})

    while len(tempt_f_cost_dict) > 1:
        new_funct_set = tempt_f_cost_dict.copy()
        list_new_z = []

        for key, funct in tempt_f_cost_dict.items():
            if np.array_equal(funct, funct_curr):
                if key != funct_curr_key:
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
                    if c < 0:
                        print('error')

                    del new_funct_set[key]
                else:
                    x_1 = x_2 = - c / b
                    if x_1 <= z_curr:
                            del new_funct_set[key]
                    else:
                        list_new_z.append([x_1, key])
            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):
                    del new_funct_set[key]
                else:
                    if x_2 <= z_curr:
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

        opt_funct_set.update({funct_curr_key: funct_curr})

        tempt_f_cost_dict = new_funct_set.copy()

    return opt_funct_set


def fill_row_k(imin, imax, k, S, sum_x_prime, sum_x_prime_sq):

    for i in range(imin, imax + 1):
        local_f_cost_dict = {}
        min_quad_term = np.Inf
        min_quad_term_funct_cost = None

        key_idx = 0
        jmin = k

        for j in range(jmin, i + 1):
            for each_funct in S[k - 1][j - 1]:
                key_idx = key_idx + 1

                f_tau_t = each_funct + ssq(j, i, sum_x_prime, sum_x_prime_sq)
                f_tau_t = check_coef_zero(f_tau_t)
                local_f_cost_dict.update({key_idx: f_tau_t})

                if check_min_q_term(f_tau_t, min_quad_term, min_quad_term_funct_cost):
                    min_quad_term = f_tau_t[0]
                    min_quad_term_funct_cost = [key_idx, f_tau_t]

        opt_funct_set = find_opt_funct_set(local_f_cost_dict, min_quad_term_funct_cost)

        for key, funct in opt_funct_set.items():
            S[k][i].append(funct)


def fill_dp_matrix(x_prime, S, K, n):
    sum_x_prime = []
    sum_x_prime_sq = []

    for i in range(n):
        a = x_prime[i][0]
        b = x_prime[i][1]

        if i == 0:
            sum_x_prime.append(np.array([a, b]))
            sum_x_prime_sq.append(np.array([a ** 2, 2 * a * b, b ** 2]))
        else:
            sum_x_prime.append(sum_x_prime[i - 1] + np.array([a, b]))
            sum_x_prime_sq.append(sum_x_prime_sq[i - 1] + np.array([a ** 2, 2 * a * b, b ** 2]))

        S[0][i].append(ssq(0, i, sum_x_prime, sum_x_prime_sq))

    for k in range(1, K):
        if k < (K-1):
            imin = max(1, k)
        else:
            imin = n - 1

        fill_row_k(imin, n-1, k, S, sum_x_prime, sum_x_prime_sq)

    return S, sum_x_prime, sum_x_prime_sq

def dp_parametrized_x(x_prime, sg_results, n_segments, n):
    S = []
    opt_funct_set = {}

    for i in range(n_segments):
        S.append([])
        for j in range(n):
            S[i].append([])

    S, sum_x_prime, sum_x_prime_sq = fill_dp_matrix(x_prime, S, n_segments, n)

    opt_funct = np.array([0, 0, 0])
    for i in range(len(sg_results) - 1):
        curr_cp = sg_results[i]
        next_cp = sg_results[i + 1]
        opt_funct = opt_funct + ssq(curr_cp, next_cp - 1, sum_x_prime, sum_x_prime_sq)

    key_idx = 0
    for each_opt_funct in S[n_segments-1][n-1]:
        key_idx = key_idx + 1
        opt_funct_set.update({key_idx: each_opt_funct})

    check_coef_zero(opt_funct)

    return opt_funct_set, opt_funct
