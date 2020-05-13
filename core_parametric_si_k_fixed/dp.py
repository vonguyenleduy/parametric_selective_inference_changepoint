import numpy as np

def ssq(j, i, sum_x, sum_x_sq):
    if (j > 0):
        muji = (sum_x[i] - sum_x[j-1]) / (i - j + 1)
        sji = sum_x_sq[i] - sum_x_sq[j-1] - (i - j + 1) * muji ** 2
    else:
        sji = sum_x_sq[i] - sum_x[i] ** 2 / (i+1)

    return 0 if sji < 0 else sji


def fill_row_k(imin, imax, k, S, J, sum_x, sum_x_sq):

    for i in range(imin, imax + 1):
        S[k][i] = S[k - 1][i - 1]
        J[k][i] = i

        jmin = k
        # for j in range(i - 1, jmin - 1, -1):
        for j in range(jmin, i):
            sji = ssq(j, i, sum_x, sum_x_sq)
            SSQ_j = sji + S[k - 1][j - 1]

            if SSQ_j < S[k][i]:
                S[k][i] = SSQ_j
                J[k][i] = j


def fill_dp_matrix(x, S, J, K, n):
    sum_x = np.zeros(n)
    sum_x_sq = np.zeros(n)

    # median. used to shift the values of x to improve numerical stability
    # shift = x[n//2]
    shift = 0

    for i in range(n):
        if i == 0:
            sum_x[0] = x[0] - shift
            sum_x_sq[0] = (x[0] - shift) ** 2
        else:
            sum_x[i] = sum_x[i-1] + x[i] - shift
            sum_x_sq[i] = sum_x_sq[i-1] + (x[i] - shift) ** 2

        S[0][i] = ssq(0, i, sum_x, sum_x_sq)
        J[0][i] = 0

    for k in range(1, K):
        if (k < K-1):
            imin = max(1, k)
        else:
            imin = n-1

        fill_row_k(imin, n-1, k, S, J, sum_x, sum_x_sq)


def dp(x, n_segments):
    if n_segments <= 0:
        raise ValueError("Cannot classify into 0 or less segments")
    if n_segments > len(x):
        raise ValueError("Cannot generate more classes than there are x values")

    # if there's only one value, return it; there's no sensible way to split
    # it. This means that len(dp([x], 2)) may not == 2. Is that OK?
    unique = len(set(x))
    if unique == 1:
        return [x]

    n = len(x)

    S = np.zeros((n_segments, n))

    J = np.zeros((n_segments, n))

    fill_dp_matrix(x, S, J, n_segments, n)

    sg_results = []
    segment_right = n-1

    for segment in range(n_segments-1, -1, -1):
        segment_left = int(J[segment][segment_right])
        sg_results.append(segment_right + 1)

        if segment > 0:
            segment_right = segment_left - 1

    sg_results.append(0)
    return list(reversed(sg_results)), S[n_segments - 1][n - 1]
