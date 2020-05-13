import numpy as np
from mpmath import mp

mp.dps = 500

error_threshold = 1e-10

def intersect(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        return []
    else:
        return [lower, upper]


def intersect_range(prime_range, tilde_range, range_tau_greater_than_zero, list_2_range):
    initial_range = intersect(prime_range, tilde_range)
    initial_range = intersect(initial_range, range_tau_greater_than_zero)

    if len(initial_range) == 0:
        return []

    final_list = [initial_range]

    for each_2_range in list_2_range:

        lower_range = [np.NINF, each_2_range[0]]
        upper_range = [each_2_range[1], np.Inf]

        new_final_list = []

        for each_1_range in final_list:
            local_range_1 = intersect(each_1_range, lower_range)
            local_range_2 = intersect(each_1_range, upper_range)

            if len(local_range_1) > 0:
                new_final_list.append(local_range_1)

            if len(local_range_2) > 0:
                new_final_list.append(local_range_2)

        final_list = new_final_list

    return final_list


def inference(data, segment_index, global_list_conditioning_matrix,
              sigma, first_segment_index, second_segment_index):
    x = np.array(data)
    n = len(x)
    x = x.reshape((x.shape[0], 1))

    vector_1_C_a = np.zeros(n)
    vector_1_C_b = np.zeros(n)

    n_a = 0
    n_b = 0

    for i in range(n):
        if segment_index[i] == second_segment_index:
            n_a = n_a + 1
            vector_1_C_a[i] = 1.0

        elif segment_index[i] == first_segment_index:
            n_b = n_b + 1
            vector_1_C_b[i] = 1.0

    vector_1_C_a = np.reshape(vector_1_C_a, (vector_1_C_a.shape[0], 1))
    vector_1_C_b = np.reshape(vector_1_C_b, (vector_1_C_b.shape[0], 1))

    first_element = np.dot(vector_1_C_a.T, x)[0][0]
    second_element = np.dot(vector_1_C_b.T, x)[0][0]

    tau = first_element / n_a - second_element / n_b

    if tau < 0:
        temp = vector_1_C_a
        vector_1_C_a = vector_1_C_b
        vector_1_C_b = temp

        temp = n_a
        n_a = n_b
        n_b = temp

    first_element = np.dot(vector_1_C_a.T, x)[0][0]
    second_element = np.dot(vector_1_C_b.T, x)[0][0]

    tau = first_element / n_a - second_element / n_b

    big_sigma = sigma * sigma * np.identity(n)

    eta_a_b = vector_1_C_a / n_a - vector_1_C_b / n_b

    c_vector = np.dot(big_sigma, eta_a_b) / ((np.dot(eta_a_b.T, np.dot(big_sigma, eta_a_b)))[0][0])

    z = np.dot((np.identity(n) - np.dot(c_vector, eta_a_b.T)), x)

    L_prime = np.NINF
    U_prime = np.Inf

    L_tilde = np.NINF
    U_tilde = np.Inf

    list_2_range = []
    range_tau_greater_than_zero = [0, np.Inf]

    for matrix in global_list_conditioning_matrix:
        c_T_A_c = np.dot(c_vector.T, np.dot(matrix, c_vector))[0][0]
        z_T_A_c = np.dot(z.T, np.dot(matrix, c_vector))[0][0]
        c_T_A_z = np.dot(c_vector.T, np.dot(matrix, z))[0][0]
        z_T_A_z = np.dot(z.T, np.dot(matrix, z))[0][0]

        a = c_T_A_c
        b = z_T_A_c + c_T_A_z
        c = z_T_A_z

        if -error_threshold <= a <= error_threshold:
            a = 0

        if -error_threshold <= b <= error_threshold:
            b = 0

        if -error_threshold <= c <= error_threshold:
            c = 0

        x_T_A_x = np.dot(x.T, np.dot(matrix, x))[0][0]
        x_T_A_c = np.dot(x.T, np.dot(matrix, c_vector))[0][0]
        c_T_A_x = np.dot(c_vector.T, np.dot(matrix, x))[0][0]

        if a == 0:
            if b == 0:
                if c > 0:
                    print('z_T_A_z > 0')
            elif b < 0:
                temporal_lower_bound = -c / b
                L_prime = max(L_prime, temporal_lower_bound)

                if L_prime > tau:
                    print('L_prime > tau')
            elif b > 0:
                temporal_upper_bound = -c / b
                U_prime = min(U_prime, temporal_upper_bound)

                if U_prime < tau:
                    print('U_prime < tau')
        else:
            delta = b ** 2 - 4 * a * c
            check_delta = (x_T_A_c + c_T_A_x) ** 2 - 4 * c_T_A_c * x_T_A_x

            if abs(delta - check_delta) > 1e-8:
                print('delta - check_delta')

            if -error_threshold <= delta <= error_threshold:
                delta = 0

            if delta == 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta = 0')
            elif delta < 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta < 0')
            elif delta > 0:
                if a > 0:
                    x_lower = (-b - np.sqrt(delta)) / (2 * a)
                    x_upper = (-b + np.sqrt(delta)) / (2 * a)

                    if x_lower > x_upper:
                        print('x_lower > x_upper')

                    L_tilde = max(L_tilde, x_lower)
                    U_tilde = min(U_tilde, x_upper)

                else:
                    x_1 = (-b - np.sqrt(delta)) / (2 * a)
                    x_2 = (-b + np.sqrt(delta)) / (2 * a)

                    x_low = min(x_1, x_2)
                    x_up = max(x_1, x_2)
                    list_2_range.append([x_low, x_up])


    final_list_range = intersect_range([L_prime, U_prime], [L_tilde, U_tilde], range_tau_greater_than_zero,
                                       list_2_range)

    if len(final_list_range) == 0:
        print('NO SOLUTION')
        return None

    numerator = 0
    denominator = 0

    tn_sigma = np.sqrt(np.dot(eta_a_b.T, np.dot(big_sigma, eta_a_b))[0][0])

    for each_final_range in final_list_range:
        al = each_final_range[0]
        ar = each_final_range[1]

        denominator = denominator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        if tau >= ar:
            numerator = numerator + (mp.ncdf(ar / tn_sigma) - mp.ncdf(al / tn_sigma))
        elif (tau >= al) and (tau < ar):
            numerator = numerator + (mp.ncdf(tau / tn_sigma) - mp.ncdf(al / tn_sigma))

    if denominator != 0:
        F = numerator / denominator
        return 1 - F
    else:
        print('denominator = 0', final_list_range, tau, tn_sigma)
        return None