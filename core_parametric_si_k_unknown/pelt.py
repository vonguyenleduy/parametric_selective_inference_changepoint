import numpy as np

def ssq(j, i, sum_x, sum_x_sq):
    if j > 0:
        muji = (sum_x[i] - sum_x[j-1]) / (i - j + 1)
        sji = sum_x_sq[i] - sum_x_sq[j-1] - (i - j + 1) * muji ** 2
    else:
        sji = sum_x_sq[i] - sum_x[i] ** 2 / (i+1)

    return 0 if sji < 0 else sji


def pelt(x, n, beta):
    sum_x = np.zeros(n)
    sum_x_sq = np.zeros(n)

    F = np.zeros(n + 1)
    R = []
    cp = []

    for i in range(n):
        cp.append([])
        R.append([])
        if i == 0:
            sum_x[0] = x[0]
            sum_x_sq[0] = x[0] ** 2
        else:
            sum_x[i] = sum_x[i-1] + x[i]
            sum_x_sq[i] = sum_x_sq[i - 1] + x[i] ** 2

    # print(sum_x)
    # print(sum_x_sq)

    cp.append([])
    R.append([])
    F[0] = - beta
    R[1].append(0)

    for tstar in range(1, n + 1):
        F[tstar] = np.Inf
        current_opt_index = None

        for t in R[tstar]:
            temp = F[t] + ssq(t, tstar - 1, sum_x, sum_x_sq) + beta

            if temp < F[tstar]:
                F[tstar] = temp
                current_opt_index = t

        if tstar < n:
            for t in R[tstar]:
                temp = F[t] + ssq(t, tstar - 1, sum_x, sum_x_sq)

                if temp <= F[tstar]:
                    R[tstar + 1].append(t)

            R[tstar + 1].append(tstar)

        cp[tstar] = cp[current_opt_index][:]
        cp[tstar].append(current_opt_index)

    cp[-1].append(n)

    return cp[-1], F[-1]


if __name__=='__main__':
    sigma = 1.0
    size = 20

    mean_a = 1
    mean_b = 5
    mean_c = 10
    mean_d = 2
    var = sigma ** 2

    data_a = np.random.normal(mean_a, var, size)
    data_b = np.random.normal(mean_b, var, size)
    data_c = np.random.normal(mean_c, var, size)
    data_d = np.random.normal(mean_d, var, size)
    data = np.append(data_a, data_b)
    data = np.append(data, data_c)
    # data = np.append(data, data_d)
    # data = [1, 2, 5, 6, 8, -1, -2]

    # data = [1.328025177623915809e+00,-3.981684001263728234e+00,5.512747385071326001e-01,1.007942202086859806e+00,1.130528810893324243e+00,6.566196127615617772e-01,5.403436649233259725e-01,2.663480066242083666e+00,-5.121716878511333171e-01,-2.265334436188166656e-01,4.781713532758713470e-02,-1.947137874219153630e-01,1.799970511788192784e+00,-1.248902611681441677e+00,1.031717079384846247e+00,-1.537114615625591263e+00,-9.405501022952629242e-01,2.718739347656107697e+00,2.212978529196915067e-01,1.423770751726643580e+00,9.377614179199897038e+00,3.918241608093565365e+00,7.441295454727224090e+00,7.988471418952177494e+00,9.313277141650782909e+00,4.194161834688321377e+00,4.605318131293520523e+00,7.138472849900880490e+00,4.047856797051502475e+00,3.496867289323919437e+00,7.803957181019941736e+00,4.407497544811318058e+00,6.099893546964512581e+00,7.431039524499994720e+00,5.782334452203732766e+00,4.414984357394200032e+00,8.368774646339327106e+00,7.806600032211527207e+00,7.391688857016843528e+00,7.249990785416464689e+00,8.305357648263310466e+00,8.624581720798916606e+00,1.345286527067911031e+01,9.365526493389047857e+00,1.307330566973537245e+01,1.075018519320082788e+01,7.300493348985225950e+00,1.219315281138981355e+01,1.081998099233891075e+01,1.327395938215821580e+01,8.601631229288898339e+00,1.075067325075304403e+01,9.332403169918638497e+00,1.092229548246026738e+01,1.119041882106579067e+01,8.974119131476022915e+00,7.602146584474288815e+00,1.119603980638063234e+01,1.169795677782764365e+01,1.303261381956351173e+01]

    n = len(data)
    beta = 2 * np.log(n)
    results, cost = pelt(data, n, beta)
    print(results, cost)

    data = np.flip(data)
    results, cost = pelt(data, n, beta)
    print(results, cost)