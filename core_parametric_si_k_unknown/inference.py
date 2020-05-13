import numpy as np
from mpmath import mp
mp.dps = 500

err_threshold = 1e-10


def check_zero(value):
    if -err_threshold <= value <= err_threshold:
        return 0

    return value


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


def union(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        if range_1[1] < range_2[0]:
            return 2, [range_1, range_2]
        else:
            return 2, [range_2, range_1]
    else:
        return 1, [min(range_1[0], range_2[0]), max(range_1[1], range_2[1])]


def inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n):
    # big_sigma = sigma * sigma * np.identity(n)
    tn_sigma = np.sqrt(np.dot(eta_vec.T, np.dot(cov, eta_vec))[0][0])
    list_interval = []

    for k, func_cost in opt_funct_set.items():
        a = func_cost[0] - opt_funct[0]
        b = func_cost[1] - opt_funct[1]
        c = func_cost[2] - opt_funct[2]

        a = check_zero(a)
        b = check_zero(b)
        c = check_zero(c)

        if (a == 0) and (b != 0):
            print('a == 0 and b != 0')

        if a == 0:
            continue

        # c = c - cost

        x_1, x_2 = quadratic_solver(a, b, c)
        if (x_1 is None) and (x_2 is None):
            if a < 0:
                print('negative a')
            continue

        elif x_1 == x_2:
            if a > 0:
                list_interval.append([np.NINF, x_1])
            elif a < 0:
                list_interval.append([x_2, np.Inf])

        else:
            if a > 0:
                list_interval.append([x_1, x_2])
            elif a < 0:
                list_interval.append([np.NINF, x_1])
                list_interval.append([x_2, np.Inf])

    sorted_list_interval = sorted(list_interval)

    union_interval = [sorted_list_interval[0]]
    for element in sorted_list_interval:
        no_of_ranges, returned_interval = union(union_interval[-1], element)
        if no_of_ranges == 2:
            union_interval[-1] = returned_interval[0]
            union_interval.append(returned_interval[1])
        else:
            union_interval[-1] = returned_interval

    z_interval = [[np.NINF, union_interval[0][0]]]

    for i in range(len(union_interval) - 1):
        z_interval.append([union_interval[i][1], union_interval[i + 1][0]])

    z_interval.append([union_interval[-1][1], np.Inf])

    # print(z_interval)

    negative_eta = - abs(eta_T_x)
    positive_eta = abs(eta_T_x)

    numerator_1 = 0
    numerator_2 = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))

        if negative_eta >= ar:
            numerator_1 = numerator_1 + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (negative_eta >= al) and (negative_eta < ar):
            numerator_1 = numerator_1 + (mp.ncdf(negative_eta / tn_sigma) - mp.ncdf(al / tn_sigma))

        if positive_eta >= ar:
            numerator_2 = numerator_2 + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (positive_eta >= al) and (positive_eta < ar):
            numerator_2 = numerator_2 + (mp.ncdf(positive_eta / tn_sigma) - mp.ncdf(al / tn_sigma))

    if denominator != 0:
        p_value = numerator_1 / denominator + (1 - numerator_2 / denominator)
        return float(p_value)
    else:
        return None