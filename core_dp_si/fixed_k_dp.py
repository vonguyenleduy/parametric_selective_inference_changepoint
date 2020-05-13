import numpy as np
import time


def ssq(j, i, sum_x, sum_x_sq):
    if j > 0:
        muji = (sum_x[i] - sum_x[j - 1]) / (i - j + 1)
        sji = sum_x_sq[i] - sum_x_sq[j - 1] - (i - j + 1) * muji ** 2
    else:
        sji = sum_x_sq[i] - sum_x[i] ** 2 / (i + 1)

    return 0 if sji < 0 else sji


def ssq_matrix(j, i, sum_x_matrix, sum_x_sq_matrix):
    if j > 0:
        muji_matrix = (sum_x_matrix[i] - sum_x_matrix[j - 1]) / (i - j + 1)
        dji_matrix = sum_x_sq_matrix[i] - sum_x_sq_matrix[j - 1] - (i - j + 1) * np.dot(muji_matrix, muji_matrix.T)
    else:
        dji_matrix = sum_x_sq_matrix[i] - (np.dot(sum_x_matrix[i], sum_x_matrix[i].T)) / (i + 1)

    return dji_matrix


def fill_dp_matrix(data, S, J, K, n):
    list_matrix = []
    list_condition_matrix = []

    sum_x = np.zeros(n, dtype=np.float_)
    sum_x_sq = np.zeros(n, dtype=np.float_)

    sum_x_matrix = []
    sum_x_sq_matrix = []

    shift = 0

    for i in range(n):
        if i == 0:
            sum_x[0] = data[0] - shift
            sum_x_sq[0] = (data[0] - shift) ** 2

            e_n_0 = np.zeros(n)
            e_n_0[0] = 1
            e_n_0 = e_n_0.reshape((e_n_0.shape[0], 1))

            sum_x_matrix.append(e_n_0)
            sum_x_sq_matrix.append(np.dot(e_n_0, e_n_0.T))

        else:
            sum_x[i] = sum_x[i - 1] + data[i] - shift
            sum_x_sq[i] = sum_x_sq[i - 1] + (data[i] - shift) ** 2

            e_n_i = np.zeros(n)
            e_n_i[i] = 1
            e_n_i = e_n_i.reshape((e_n_i.shape[0], 1))

            sum_x_matrix.append(sum_x_matrix[i - 1] + e_n_i)
            sum_x_sq_matrix.append(sum_x_sq_matrix[i - 1] + np.dot(e_n_i, e_n_i.T))

        S[0][i] = ssq(0, i, sum_x, sum_x_sq)
        J[0][i] = 0

        list_matrix.append(ssq_matrix(0, i, sum_x_matrix, sum_x_sq_matrix))

    for k in range(1, K):
        if (k < K - 1):
            imin = max(1, k)
        else:
            imin = n - 1

        imax = n - 1

        new_list_matrix = []

        for _ in range(n):
            new_list_matrix.append([])

        for i in range(imin, imax + 1):
            S[k][i] = S[k - 1][i - 1]
            J[k][i] = i

            list_matrix_Y_plus_Z = []

            matrix_Y = list_matrix[i - 1]

            new_list_matrix[i] = matrix_Y

            list_matrix_Y_plus_Z.append(matrix_Y)

            jmin = k
            for j in range(i - 1, jmin - 1, -1):
                sji = ssq(j, i, sum_x, sum_x_sq)
                SSQ_j = sji + S[k - 1][j - 1]

                matrix_Y = list_matrix[j - 1]

                matrix_Z = ssq_matrix(j, i, sum_x_matrix, sum_x_sq_matrix)

                list_matrix_Y_plus_Z.append(matrix_Y + matrix_Z)

                if SSQ_j < S[k][i]:
                    S[k][i] = SSQ_j
                    J[k][i] = j
                    new_list_matrix[i] = matrix_Y + matrix_Z

            matrix_X = new_list_matrix[i]

            for matrix_Y_plus_Z in list_matrix_Y_plus_Z:
                list_condition_matrix.append(matrix_X - matrix_Y_plus_Z)

        list_matrix = new_list_matrix[:]

    return list_condition_matrix


def dp_si(data, n_segments):
    n = len(data)

    S = np.zeros((n_segments, n))

    J = np.zeros((n_segments, n))

    list_condition_matrix = fill_dp_matrix(data, S, J, n_segments, n)

    segment_index = np.zeros(n)
    sg_results = []
    segment_right = n - 1

    for segment in range(n_segments - 1, -1, -1):
        segment_left = int(J[segment][segment_right])
        sg_results.append(segment_right + 1)

        for i in range(segment_left, segment_right + 1):
            segment_index[i] = segment + 1

        if segment > 0:
            segment_right = segment_left - 1

    sg_results.append(0)
    return segment_index, list_condition_matrix, list(reversed(sg_results))