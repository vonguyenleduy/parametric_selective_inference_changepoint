import numpy as np
import time


def dp_si(data, sigma, beta):
    x = np.array(data[:])
    x = x.reshape((x.shape[0], 1))

    n = len(data)

    sum_x = np.zeros(n)
    sum_x_matrix = []

    list_matrix = [[]]
    list_condition_matrix = []

    F = np.zeros(n + 1)
    R = []
    cp = []

    for i in range(n):
        cp.append([])
        R.append([])

        list_matrix.append([])
        if i == 0:
            sum_x[0] = data[0]

            e_n_0 = np.zeros(n)
            e_n_0[0] = 1
            e_n_0 = e_n_0.reshape((e_n_0.shape[0], 1))

            sum_x_matrix.append(e_n_0)
        else:
            sum_x[i] = sum_x[i-1] + data[i]

            e_n_i = np.zeros(n)
            e_n_i[i] = 1
            e_n_i = e_n_i.reshape((e_n_i.shape[0], 1))

            sum_x_matrix.append(sum_x_matrix[i - 1] + e_n_i)

    cp.append([])
    R.append([])
    F[0] = - beta
    R[1].append(0)

    list_matrix[0] = [np.zeros((n, n)), - beta]

    for tstar in range(1, n + 1):
        F[tstar] = np.Inf
        current_opt_index = None

        list_matrix_A_plus_constant_b = []

        # for t in range(tstar):
        for t in R[tstar]:
            if t > 0:
                temp = F[t] - (sum_x[tstar - 1] - sum_x[t - 1])**2 / (tstar - 1 - (t - 1)) \
                       + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                matrix_A = list_matrix[t][0] - \
                           np.dot(sum_x_matrix[tstar - 1] - sum_x_matrix[t-1],
                                  (sum_x_matrix[tstar - 1] - sum_x_matrix[t-1]).T) / (tstar - 1 - (t - 1))

                constant_b = list_matrix[t][1] + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                list_matrix_A_plus_constant_b.append([matrix_A, constant_b])

                if temp < F[tstar]:
                    F[tstar] = temp
                    current_opt_index = t
                    list_matrix[tstar] = [matrix_A, constant_b]

            elif t == 0:
                temp = F[t] - (sum_x[tstar - 1])**2 / (tstar - 1 - (t - 1)) \
                       + sigma**2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                matrix_A = list_matrix[t][0] - \
                           np.dot(sum_x_matrix[tstar - 1],
                                  (sum_x_matrix[tstar - 1]).T) / (tstar - 1 - (t - 1))

                constant_b = list_matrix[t][1] + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n) + beta

                list_matrix_A_plus_constant_b.append([matrix_A, constant_b])

                if temp < F[tstar]:
                    F[tstar] = temp
                    current_opt_index = t
                    list_matrix[tstar] = [matrix_A, constant_b]

        if tstar < n:
            for t in R[tstar]:
                if t > 0:
                    temp = F[t] - (sum_x[tstar - 1] - sum_x[t - 1]) ** 2 / (tstar - 1 - (t - 1)) \
                           + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n)

                    matrix_A = list_matrix[t][0] - \
                               np.dot(sum_x_matrix[tstar - 1] - sum_x_matrix[t - 1],
                                      (sum_x_matrix[tstar - 1] - sum_x_matrix[t - 1]).T) / (tstar - 1 - (t - 1))

                    constant_b = list_matrix[t][1] + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n)

                    if temp <= F[tstar]:
                        R[tstar + 1].append(t)

                        list_condition_matrix.append(
                            [matrix_A - list_matrix[tstar][0], constant_b - list_matrix[tstar][1]])

                    else:
                        list_condition_matrix.append(
                            [list_matrix[tstar][0] - matrix_A, list_matrix[tstar][1] - constant_b])


                elif t == 0:
                    temp = F[t] - (sum_x[tstar - 1]) ** 2 / (tstar - 1 - (t - 1)) \
                           + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n)

                    matrix_A = list_matrix[t][0] - \
                               np.dot(sum_x_matrix[tstar - 1],
                                      (sum_x_matrix[tstar - 1]).T) / (tstar - 1 - (t - 1))

                    constant_b = list_matrix[t][1] + sigma ** 2 * np.log((tstar - 1 - (t - 1)) / n)

                    if temp <= F[tstar]:
                        R[tstar + 1].append(t)

                        list_condition_matrix.append(
                            [matrix_A - list_matrix[tstar][0], constant_b - list_matrix[tstar][1]])

                    else:
                        list_condition_matrix.append(
                            [list_matrix[tstar][0] - matrix_A, list_matrix[tstar][1] - constant_b])


            R[tstar + 1].append(tstar)

        for each_element in list_matrix_A_plus_constant_b:
            list_condition_matrix.append([list_matrix[tstar][0] - each_element[0], list_matrix[tstar][1] - each_element[1]])

        cp[tstar] = cp[current_opt_index - 1][:]
        cp[tstar].append(current_opt_index)

    cluster_index = np.zeros(n)

    final_cp = cp[-1][:]
    final_cp.append(n)

    for i in range(1, len(final_cp)):
        for j in range(final_cp[i - 1], final_cp[i]):
            cluster_index[j] = i

    return cluster_index, list_condition_matrix, len(final_cp) - 1, final_cp
