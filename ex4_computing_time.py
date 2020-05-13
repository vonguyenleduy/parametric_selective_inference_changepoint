import numpy as np
import time

from core_parametric_si_k_unknown import parametrize_x
from core_parametric_si_k_unknown import pelt
from core_parametric_si_k_unknown import pelt_perturbed_x
from core_parametric_si_k_unknown import inference


def init_data(segment_size, sigma, mean_vector, N):
    size = segment_size

    mean_a = mean_vector[0]
    mean_b = mean_vector[1]
    mean_c = mean_vector[2]
    mean_d = mean_vector[3]
    mean_e = mean_vector[4]
    var = sigma ** 2

    x_a = np.random.normal(mean_a, var, size)
    x_b = np.random.normal(mean_b, var, size)
    x_c = np.random.normal(mean_c, var, size)
    x_d = np.random.normal(mean_d, var, size)
    x_e = np.random.normal(mean_e, var, size)

    seg = np.append(x_a, x_b)
    seg = np.append(seg, x_c)
    seg = np.append(seg, x_d)
    seg = np.append(seg, x_e)

    x = np.append(seg, seg)

    if N >= 400:
        x = np.append(x, seg)
        x = np.append(x, seg)

    if N >= 600:
        x = np.append(x, seg)
        x = np.append(x, seg)

    if N >= 800:
        x = np.append(x, seg)
        x = np.append(x, seg)

    if N >= 1000:
        x = np.append(x, seg)
        x = np.append(x, seg)

    if N >= 12000:
        x = np.append(x, seg)
        x = np.append(x, seg)

    cov = np.identity(len(x))

    return x, cov


def run(N):
    list_seed = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    count = 0
    for seed in list_seed:
        count = count + 1

        print("==================== Try " + str(count) + " ====================")
        start_time = time.time()

        np.random.seed(seed)
        mean_vector = [0, 2, -1, 3, -2]
        sigma = 1.0
        segment_size = 20

        x, cov = init_data(segment_size, sigma, mean_vector, N)
        n = len(x)
        beta = 2 * np.log(n)

        x_flip = np.flip(x)

        sg_results, cost = pelt.pelt(x, n, beta)
        print(n, sg_results)

        if len(sg_results) > 2:
            sg_results_len = len(sg_results)

            sg_results_flip, cost_flip = pelt.pelt(x_flip, n, beta)

            for i in range(1, sg_results_len - 1):
                if sg_results[i] > (n / 2):
                    x_prime, eta_vec, eta_T_x = parametrize_x.parametrize_x(x, n, sg_results[i - 1], sg_results[i],
                                                                            sg_results[i + 1], cov)
                    opt_funct_set, opt_funct = pelt_perturbed_x.pelt_perturbed_x(x_prime, sg_results, n, beta)
                    p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)

                    print('Changing point:', i, ', p-value:', p_value)

                else:
                    x_prime_flip, eta_vec, eta_T_x = \
                        parametrize_x.parametrize_x(
                            x_flip, n, sg_results_flip[sg_results_len - 1 - i - 1], sg_results_flip[sg_results_len - i - 1],
                            sg_results_flip[sg_results_len - 1 - i + 1], cov)

                    eta_vec = - eta_vec
                    eta_T_x = - eta_T_x
                    opt_funct_set, opt_funct = pelt_perturbed_x.pelt_perturbed_x(x_prime_flip, sg_results_flip, n, beta)
                    p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)
                    print('Changing point:', i, ', p-value:', p_value)
        else:
            print("No Changing Point")

        print("\n")
        print("Time: %0.2f seconds " % (time.time() - start_time))
        print("\n")


if __name__ == "__main__":
    # Please set N in {200, 400, 600, 800, 1000, 1200}
    N = 100

    if N not in [200, 400, 600, 800, 1000, 1200]:
        print("Please set the value of N in {200, 400, 600, 800, 1000, 1200}")
    else:
        run(N)
