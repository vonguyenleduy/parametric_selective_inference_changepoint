import numpy as np
import matplotlib.pyplot as plt

from core_dp_si import penalized_opt_par
from core_dp_si import penalized_inference
from core_parametric_si_k_unknown import parametrize_x
from core_parametric_si_k_unknown import pelt
from core_parametric_si_k_unknown import pelt_perturbed_x
from core_parametric_si_k_unknown import inference


def fpr_optseg_si(max_loop, size_list, sigma, alpha):
    list_fpr = []
    for n in size_list:
        print('N =', n)
        iter = 1
        count = 0
        total = 0
        while iter <= max_loop:
            np.random.seed(iter)
            x = np.random.normal(0, 1, n)
            cov = np.identity(n)
            beta = 1.5 * np.log(n)
            sg_results, cost = pelt.pelt(x, n, beta)

            if len(sg_results) > 2:
                total = total + 1
                i = np.random.randint(len(sg_results) - 2) + 1
                x_prime, eta_vec, eta_T_x = parametrize_x.parametrize_x(x, n, sg_results[i - 1], sg_results[i],
                                                                        sg_results[i + 1], cov)
                opt_funct_set, opt_funct = pelt_perturbed_x.pelt_perturbed_x(x_prime, sg_results, n, beta)
                p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)
                if p_value < alpha:
                    count = count + 1

            iter = iter + 1

        fpr = count / total
        list_fpr.append(fpr)
        print("FPR = %.3f" % fpr)
        print("--------------------")

    return list_fpr


def fpt_optseg_si_oc(max_loop, size_list, sigma, alpha):
    list_fpr = []
    for n in size_list:
        print('N =', n)
        iter = 1
        count = 0
        total = 0
        while iter <= max_loop:
            np.random.seed(iter)
            x = np.random.normal(0, sigma, n)
            beta = 1.5 * np.log(n)
            segment_index, list_condition_matrix, number_of_segments, sg_results = penalized_opt_par.dp_si(x, sigma,
                                                                                                           beta)
            if number_of_segments > 1:
                total = total + 1
                first_segment_index = np.random.randint(number_of_segments - 1) + 1
                second_segment_index = first_segment_index + 1
                p_value = penalized_inference.inference(x, segment_index, list_condition_matrix,
                                                        sigma, first_segment_index, second_segment_index)

                if p_value < alpha:
                    count = count + 1

            iter = iter + 1

        fpr = count / total
        list_fpr.append(fpr)
        print("FPR = %.3f" % fpr)
        print("--------------------")

    return list_fpr


if __name__ == "__main__":
    max_loop = 1000
    size_list = [10, 20, 30, 40]
    sigma = 1
    alpha = 0.05

    # create an index for each tick position for plotting results
    xi = list(range(len(size_list)))

    print("\n")

    print("============== FPR for OptSeg-SI-oc  ==============")
    print("--------------------")
    list_fpr_optseg_si_oc = fpt_optseg_si_oc(max_loop, size_list, sigma, alpha)

    print("\n")

    print("============== FPR for OptSeg-SI  ==============")
    print("--------------------")
    list_fpr_optseg_si = fpr_optseg_si(max_loop, size_list, sigma, alpha)


    try:
        plt.plot(xi, list_fpr_optseg_si_oc, marker='o', linestyle='-', color='green', label='OptSeg-SI-oc')
        plt.plot(xi, list_fpr_optseg_si, marker='o', linestyle='-', color='r', label='OptSeg-SI')
        plt.ylim((0, 0.12))
        plt.xlabel('N')
        plt.ylabel('FPR')
        plt.xticks(xi, size_list)
        plt.title("False Positive Rate (K is unknown)")
        plt.legend()
        plt.savefig("./figure_results/ex2_unknown_k_fpr_optseg_si_optseg_si_oc.pdf", format="pdf")
        plt.show()
    except:
        plt.switch_backend('agg')
        plt.plot(xi, list_fpr_optseg_si_oc, marker='o', linestyle='-', color='green', label='OptSeg-SI-oc')
        plt.plot(xi, list_fpr_optseg_si, marker='o', linestyle='-', color='r', label='OptSeg-SI')
        plt.ylim((0, 0.12))
        plt.xlabel('N')
        plt.ylabel('FPR')
        plt.xticks(xi, size_list)
        plt.title("False Positive Rate (K is unknown)")
        plt.legend()
        plt.savefig("./figure_results/ex2_unknown_k_fpr_optseg_si_optseg_si_oc.pdf", format="pdf")
        plt.show()