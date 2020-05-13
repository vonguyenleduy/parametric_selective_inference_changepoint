import numpy as np
import matplotlib.pyplot as plt

from core_dp_si import penalized_opt_par
from core_dp_si import penalized_inference

from core_parametric_si_k_unknown import parametrize_x
from core_parametric_si_k_unknown import pelt
from core_parametric_si_k_unknown import pelt_perturbed_x
from core_parametric_si_k_unknown import inference


def calculate_mean_for_each_segment(x, sg_results):
    list_mean = []

    for i in range(len(sg_results) - 1):
        no_elements = 0
        sum = 0
        for j in range(sg_results[i], sg_results[i + 1]):
            no_elements = no_elements + 1
            sum = sum + x[j]

        list_mean.append(sum / no_elements)

    return list_mean


def plot(x, sg_results, mean_vector, n, segment_size, label):
    list_mean = calculate_mean_for_each_segment(x, sg_results)

    # Estimated CP
    parametric_x = []

    for element in sg_results:
        if (element == 0) or (element == n):
            parametric_x.append(element)
        else:
            parametric_x.append(element + 0.5)
            parametric_x.append(element + 0.5)

    parametric_y = []
    for element in list_mean:
        parametric_y.append(element)
        parametric_y.append(element)

    # True CP
    true_x = [0]
    curr_idx = 0

    while curr_idx < (n - 20):
        curr_idx = curr_idx + segment_size
        true_x.append(curr_idx + 0.5)
        true_x.append(curr_idx + 0.5)

    true_x.append(n)

    vector = []
    for each_mean in mean_vector:
        vector.append(each_mean)
        vector.append(each_mean)

    vector = np.array(vector)

    true_y = np.append(vector, vector)

    # plt.figure(figsize=(13.5, 5.9))
    # plt.ylim((-2.5, 3.5))
    plt.plot(x, 'o', color='grey', fillstyle='none', markersize=6, markeredgewidth='0.4')
    plt.plot(true_x, true_y, color='blue', linewidth='2', label='True Signal')
    plt.plot(parametric_x, parametric_y, color='red', linewidth='2', label=label)
    plt.legend(loc='lower left')

    # plt.savefig('ex1_demonstrate_proposed.pdf')


def optseg_si_oc(x, mean_vector, segment_size,  sigma, alpha):
    n = len(x)
    beta = 3 * np.log(n)

    segment_index, list_condition_matrix, number_of_segments, sg_results = penalized_opt_par.dp_si(x, sigma, beta)
    print(sg_results)

    if number_of_segments > 1:
        new_sg_results = [0]

        for k_index in range(number_of_segments - 1):
            first_segment_index = k_index + 1
            second_segment_index = first_segment_index + 1

            selective_p = penalized_inference.inference(x, segment_index, list_condition_matrix,
                                                        sigma, first_segment_index, second_segment_index)

            p_value = float(selective_p)

            print('Changing point:', k_index, ', p-value:', p_value)

            if p_value < (alpha / (len(sg_results) - 2)):
                new_sg_results.append(sg_results[k_index + 1])

        new_sg_results.append(n)
        sg_results = new_sg_results
        plot(x, sg_results, mean_vector, n, segment_size, 'OptSeg-SI-oc')
    else:
        print("No Changing Point")


def optseg_si(x, mean_vector, segment_size,  sigma, alpha):
    n = len(x)
    cov = np.identity(len(x))

    beta = 2 * np.log(n)

    x_flip = np.flip(x)

    sg_results, cost = pelt.pelt(x, n, beta)
    print(sg_results)

    if len(sg_results) > 2:
        sg_results_len = len(sg_results)

        sg_results_flip, cost_flip = pelt.pelt(x_flip, n, beta)

        new_sg_results = [0]

        for i in range(1, sg_results_len - 1):
            if sg_results[i] > (n / 2):
                x_prime, eta_vec, eta_T_x = parametrize_x.parametrize_x(x, n, sg_results[i - 1], sg_results[i],
                                                                        sg_results[i + 1], cov)
                opt_funct_set, opt_funct = pelt_perturbed_x.pelt_perturbed_x(x_prime, sg_results, n, beta)
                p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)

                print('Changing point:', i, ', p-value:', p_value)

                if p_value < (alpha / (len(sg_results) - 2)):
                    new_sg_results.append(sg_results[i])

            else:
                x_prime_flip, eta_vec, eta_T_x = parametrize_x.\
                    parametrize_x(x_flip, n, sg_results_flip[sg_results_len - 1 - i - 1], sg_results_flip[sg_results_len - i - 1],
                                  sg_results_flip[sg_results_len - 1 - i + 1], cov)
                eta_vec = - eta_vec
                eta_T_x = - eta_T_x
                opt_funct_set, opt_funct = pelt_perturbed_x.pelt_perturbed_x(x_prime_flip, sg_results_flip, n, beta)
                p_value = inference.inference(opt_funct_set, opt_funct, eta_vec, eta_T_x, cov, n)

                print('Changing point:', i, ', p-value:', p_value)

                if p_value < (alpha / (len(sg_results) - 2)):
                    new_sg_results.append(sg_results[i])

        new_sg_results.append(n)
        sg_results = new_sg_results

        plot(x, sg_results, mean_vector, n, segment_size, 'OptSeg-SI')


    else:
        print("No Changing Point")


if __name__=='__main__':

    alpha = 0.05
    sigma = 1.0
    segment_size = 20
    mean_vector = [0, 2, 0, 2, -1]

    x = [-0.051964249505532176, -0.11119605035442469, 1.0417967990841295, -1.256739294064987, 0.7453876814972659, -1.7110537607276717, -0.20586438172426652, -0.234571291207228, 1.1281440438104338, -0.012625951627825618, -0.6132002869132012, 1.3736884997604164, 1.6109919764383442, -0.6892282712408866, 0.6919237106508977, -0.44811560257635547, 0.16234246679717987, 0.25722913412818593, -1.2754558593011862, 0.06400443492565071, 0.9381433817620777, 1.0106316064310734, 1.542276774005555, 0.01581838565764615, 0.5235578780045562, 2.231802959660187, 2.64415927444301, 2.8521226952659893, 1.5359812814151994, 2.6971765975188817, 3.5678821766222573, 3.1785562058590147, 0.6160431314222861, 0.2526662046684214, 2.4027237912358532, 3.2444827979614717, 1.9761636479623537, 2.952567708239096, 2.2449639420248504, 2.224097139557027, 0.29668119719494945, 0.22075338645325868, -0.42330083288973963, 1.8456151134344994, 0.9201145688058019, -0.5579162299499533, -0.28522503958304174, -1.0412666353051778, 0.48036942553296935, -1.4273776048040527, -0.33326641894406184, 0.7473084926814644, 0.5602296283679519, 0.5737089363593981, -1.1808805186330447, 0.7646500833837553, -0.13438498339181942, 1.3246376757477119, -0.27642764572092876, 1.6795509743129613, 2.415161874048822, 2.747681603950666, 1.6074695939047596, 1.593675927654062, 1.6141540281491746, 3.0009021770734328, 1.5523175679022665, 1.8839159816414588, 3.169014615043663, 2.5156139534847157, 1.7421327694168989, 2.28504581373337, 0.5954096493834018, 0.49091764589388753, 2.434307894818352, 4.7550032972015845, 4.010792808346443, 4.519898365204564, 2.570437575952756, 1.4734844947948293, 0.06822315426795522, -2.1945433691322274, -3.859687985885645, -0.5757929742781092, 0.03361263451613761, -0.29479642243633486, 0.550265023574215, -0.23458055320358573, -0.653691218096434, -2.0538173053469837, -1.1294887663571325, -2.181434458715149, -1.3688301866874089, 0.9027399295917449, -0.7897927753921131, -0.6232971896332453, -1.6939358007313337, 1.6028132784092461, -2.983277828363932, -1.9268636584418388, -0.051964249505532176, -0.11119605035442469, 1.0417967990841295, -1.256739294064987, 0.7453876814972659, -1.7110537607276717, -0.20586438172426652, -0.234571291207228, 1.1281440438104338, -0.012625951627825618, -0.6132002869132012, 1.3736884997604164, 1.6109919764383442, -0.6892282712408866, 0.6919237106508977, -0.44811560257635547, 0.16234246679717987, 0.25722913412818593, -1.2754558593011862, 0.06400443492565071, 0.9381433817620777, 1.0106316064310734, 1.542276774005555, 0.01581838565764615, 0.5235578780045562, 2.231802959660187, 2.64415927444301, 2.8521226952659893, 1.5359812814151994, 2.6971765975188817, 3.5678821766222573, 3.1785562058590147, 0.6160431314222861, 0.2526662046684214, 2.4027237912358532, 3.2444827979614717, 1.9761636479623537, 2.952567708239096, 2.2449639420248504, 2.224097139557027, 0.29668119719494945, 0.22075338645325868, -0.42330083288973963, 1.8456151134344994, 0.9201145688058019, -0.5579162299499533, -0.28522503958304174, -1.0412666353051778, 0.48036942553296935, -1.4273776048040527, -0.33326641894406184, 0.7473084926814644, 0.5602296283679519, 0.5737089363593981, -1.1808805186330447, 0.7646500833837553, -0.13438498339181942, 1.3246376757477119, -0.27642764572092876, 1.6795509743129613, 2.415161874048822, 2.747681603950666, 1.6074695939047596, 1.593675927654062, 1.6141540281491746, 3.0009021770734328, 1.5523175679022665, 1.8839159816414588, 3.169014615043663, 2.5156139534847157, 1.7421327694168989, 2.28504581373337, 0.5954096493834018, 0.49091764589388753, 2.434307894818352, 4.7550032972015845, 4.010792808346443, 4.519898365204564, 2.570437575952756, 1.4734844947948293, 0.06822315426795522, -2.1945433691322274, -3.859687985885645, -0.5757929742781092, 0.03361263451613761, -0.29479642243633486, 0.550265023574215, -0.23458055320358573, -0.653691218096434, -2.0538173053469837, -1.1294887663571325, -2.181434458715149, -1.3688301866874089, 0.9027399295917449, -0.7897927753921131, -0.6232971896332453, -1.6939358007313337, 1.6028132784092461, -2.983277828363932, -1.9268636584418388]

    try:
        plt.plot(0)
        plt.clf()
    except:
        plt.switch_backend('agg')

    print("\n")

    print("============== OptSeg-SI-oc  ==============")
    plt.subplot(2, 1, 1)
    optseg_si_oc(x, mean_vector, segment_size, sigma, alpha)

    print("\n")

    print("============== OptSeg-SI  ==============")
    plt.subplot(2, 1, 2)
    optseg_si(x, mean_vector, segment_size, sigma, alpha)

    plt.savefig("./figure_results/ex3_power_demo_optseg_si_optseg_si_oc.pdf", format="pdf")
    plt.show()