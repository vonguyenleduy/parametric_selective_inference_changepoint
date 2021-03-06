import numpy as np
import matplotlib.pyplot as plt

from core_dp_si import fixed_k_dp
from core_dp_si import fixed_k_inference

from core_parametric_si_k_fixed import dp
from core_parametric_si_k_fixed import parametrize_x
from core_parametric_si_k_fixed import dp_parametrized_x
from core_parametric_si_k_fixed import inference


def power_optseg_si_method(list_url, sigma, n, n_segments, alpha):
    
    cov = sigma * sigma * np.identity(n)
    list_power = []
    
    for each_url in list_url:
        print(each_url)
        list_data = np.genfromtxt(each_url, delimiter=',')

        correct_detected = 0
        correct_rejected = 0

        count = 0
        for x in list_data:
            count = count + 1

            if count % 50 == 0:
                print("%2.0f %%" % (count * 100 / 250))

            sg_results, cost = dp.dp(x, n_segments)

            x_flip = np.flip(x)
            sg_results_flip, cost_flip = dp.dp(x_flip, n_segments)

            sg_results_len = len(sg_results)

            for i in range(1, sg_results_len - 1):
                each_element = sg_results[i]
                if each_element in [18, 19, 20, 21, 22, 38, 39, 40, 41, 42]:
                    correct_detected = correct_detected + 1

                    if each_element > (n / 2):
                        x_prime, eta_vec, eta_T_x = parametrize_x.parametrize_x(x, n, sg_results[i - 1], sg_results[i],
                                                                                sg_results[i + 1], cov)
                        opt_funct_set, opt_funct = dp_parametrized_x.dp_parametrized_x(x_prime, sg_results, n_segments, n)
                        p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)

                        if (p_value is not None) and (p_value < (alpha / 2)):
                            correct_rejected = correct_rejected + 1

                    else:
                        x_prime_flip, eta_vec, eta_T_x = parametrize_x.parametrize_x(x_flip, n, sg_results_flip[sg_results_len - 1 - i - 1],
                                                                                     sg_results_flip[sg_results_len - i - 1],
                                                                                     sg_results_flip[sg_results_len - 1 - i + 1], cov)
                        eta_vec = - eta_vec
                        eta_T_x = - eta_T_x

                        opt_funct_set, opt_funct = dp_parametrized_x.dp_parametrized_x(x_prime_flip, sg_results_flip,
                                                                                       n_segments, n)
                        p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)

                        if (p_value is not None) and (p_value < (alpha / 2)):
                            correct_rejected = correct_rejected + 1

        power = correct_rejected / correct_detected
        list_power.append(power)
        print("Power = %.3f" % power)
        print("--------------------")

    return list_power


def power_optseg_si_oc(list_url, sigma, n, number_of_segments, alpha):
    list_power = []
    for each_url in list_url:
        print(each_url)
        list_data = np.genfromtxt(each_url, delimiter=',')

        correct_detected = 0
        correct_rejected = 0

        count = 0
        for x in list_data:

            count = count + 1

            if count % 50 == 0:
                print("%2.0f %%" % (count * 100 / 250))

            segment_index, list_condition_matrix, sg_results = fixed_k_dp.dp_si(x, number_of_segments)

            for k_index in range(number_of_segments - 1):
                if sg_results[k_index + 1] in [18, 19, 20, 21, 22, 38, 39, 40, 41, 42]:
                    correct_detected = correct_detected + 1

                    first_segment_index = k_index + 1
                    second_segment_index = first_segment_index + 1

                    p_value = fixed_k_inference.inference(x, segment_index, list_condition_matrix,
                                                          sigma, first_segment_index, second_segment_index)

                    if (p_value is not None) and (p_value < (alpha / 2)):
                        correct_rejected = correct_rejected + 1

        power = correct_rejected / correct_detected
        list_power.append(power)
        print("Power = %.3f" % power)
        print("--------------------")
        
    return list_power

if __name__ == "__main__":
    list_url = ['./data_simulation/delta_mu_1.csv', './data_simulation/delta_mu_2.csv', './data_simulation/delta_mu_3.csv',
                './data_simulation/delta_mu_4.csv']

    sigma = 1
    alpha = 0.05
    n_segments = 3
    n = 60

    # create an index for each tick position for plotting results
    delta_mu_list = [1, 2, 3, 4]
    xi = list(range(len(delta_mu_list)))

    print("\n")
    print("============== Power for OptSeg-SI-oc  ==============")
    print("--------------------")
    list_power_optseg_si_oc = power_optseg_si_oc(list_url, sigma, n, n_segments, alpha)

    print("\n")

    print("============== Power for OptSeg-SI  ==============")
    print("--------------------")
    list_power_optseg_si_method = power_optseg_si_method(list_url, sigma, n, n_segments, alpha)

    # list_power_optseg_si_oc = [0.077, 0.234, 0.348, 0.426]
    # list_power_optseg_si_method = [0.470, 0.949, 0.994, 1]

    try:
        plt.plot(xi, list_power_optseg_si_oc, marker='o', linestyle='-', color='grey', label='OptSeg-SI-oc')
        plt.plot(xi, list_power_optseg_si_method, marker='o', linestyle='-', color='r', label='OptSeg-SI')
        plt.ylim((0, 1.03))
        plt.xlabel('delta mu')
        plt.ylabel('Power')
        plt.xticks(xi, delta_mu_list)
        plt.title("Power (K is fixed)")
        plt.legend(loc='lower right')
        plt.savefig("./figure_results/ex1_fixed_k_power_optseg_si_optseg_si_oc.pdf", format="pdf")
        plt.show()
    except:
        plt.switch_backend('agg')
        plt.plot(xi, list_power_optseg_si_oc, marker='o', linestyle='-', color='grey', label='OptSeg-SI-oc')
        plt.plot(xi, list_power_optseg_si_method, marker='o', linestyle='-', color='r', label='OptSeg-SI')
        plt.ylim((0, 1.03))
        plt.xlabel('delta mu')
        plt.ylabel('Power')
        plt.xticks(xi, delta_mu_list)
        plt.title("Power (K is fixed)")
        plt.legend(loc='lower right')
        plt.savefig("./figure_results/ex1_fixed_k_power_optseg_si_optseg_si_oc.pdf", format="pdf")
        plt.show()