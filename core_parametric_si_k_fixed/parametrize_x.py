import numpy as np

def parametrize_x(x, n, cp_idx_left, cp_idx_curr, cp_idx_right, cov):
    x_vec = np.reshape(x, (n, 1))

    seg_1_vec = np.zeros(n)
    seg_1_len = 0
    seg_2_vec = np.zeros(n)
    seg_2_len = 0

    for i in range(cp_idx_left, cp_idx_curr):
        seg_1_vec[i] = 1.0
        seg_1_len += 1

    for i in range(cp_idx_curr, cp_idx_right):
        seg_2_vec[i] = 1
        seg_2_len += 1

    eta = seg_1_vec / seg_1_len - seg_2_vec / seg_2_len
    eta_vec = eta.reshape(n, 1)
    eta_T_x = np.dot(eta_vec.T, x_vec)[0][0]

    eta_T_cov_eta = np.dot(eta_vec.T, np.dot(cov, eta_vec))[0][0]
    cov_eta = np.dot(cov, eta_vec)

    x_prime = []

    for i in range(n):
        a = cov_eta[i][0] / eta_T_cov_eta
        b = x[i] - (eta_T_x / eta_T_cov_eta) * cov_eta[i][0]
        x_prime.append([a, b])

    return x_prime, eta_vec, eta_T_x