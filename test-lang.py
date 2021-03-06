import numpy as np
from numpy.polynomial.polynomial import polyfit

from collections import OrderedDict

from math import tanh
import json

from icecream import ic


if __name__ == '__main__':
    # ic(json.dumps(OrderedDict(b=2, a=1)))

    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # ic({k: v for k, v in sorted(x.items(), key=lambda item: item[1])})

    # a = np.arange(10)
    # ic(a[:2], a[10-2:])
    # ic(a[2:10-2+1])
    # b = np.vectorize(lambda x: np.array([x, x+1, x+2]))(a)
    # ic(b)

    # x = np.array([3.08, 3.1, 3.12, 3.14, 3.16, 3.18, 3.2, 3.22, 3.24,
    #               3.26, 3.28, 3.3, 3.32, 3.34, 3.36, 3.38, 3.4, 3.42,
    #               3.44, 3.46, 3.48, 3.5, 3.52, 3.54, 3.56, 3.58, 3.6,
    #               3.62, 3.64, 3.66, 3.68])
    # y = np.array([0.000857, 0.001182, 0.001619, 0.002113, 0.002702, 0.003351,
    #               0.004062, 0.004754, 0.00546, 0.006183, 0.006816, 0.007362,
    #               0.007844, 0.008207, 0.008474, 0.008541, 0.008539, 0.008445,
    #               0.008251, 0.007974, 0.007608, 0.007193, 0.006752, 0.006269,
    #               0.005799, 0.005302, 0.004822, 0.004339, 0.00391, 0.003481,
    #               0.003095])
    # x = y = np.arange(10)
    # ic(polyfit(x, y, 1))

    # arr = np.arange(10) / 10
    # for a in arr:
    #     ic(a, tanh(a))
    # ic(t)
    # ic(np.tanh(np.arange(10)))

    # a = np.array([])
    # ic(a.mean())

    # d = {'kurtosis': 0.843158758634883,
    #      'mean_freq': 0.7899836006151681,
    #      'peak_freq': 0.823076708445556,
    #      'range_time': 0.8210213204803845,
    #      'rms_freq': 0.7021477539673981,
    #      'rms_time': 0.741005034189017,
    #      'rot_amp': 0.7417523215995581,
    #      'skewness': 0.7787196559142084}
    # l = sorted(d, key=d.get, reverse=True)
    # for feat in l:
    #     ic(feat, d[feat])

    # a = np.arange(10)
    # ic(a, a.reshape(-1, 1))
    #
    # a = np.array([1])
    # ic(a.std())

    a = np.arange(12).astype(np.float16) - 5
    ic(a.dtype, a)
    m = (a < 4) | (a % 2 == 1)
    m = (a <= 20) | (a >= 10)
    ic(m, type(m), m.sum())
    # idxs = np.where(a <= -3)[0]
    # ic(idxs, type(idxs))
    # ic(a[idxs])
    # idxs2 = np.where(a >= 3)[0]
    # ic(idxs2)
    # ic(np.concatenate([idxs, idxs2]))

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from scipy import stats
    #
    # train = [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6] * 2
    #
    # fig, ax = plt.subplots(1, 2)
    #
    # # Distribution from seaborn
    # sns.distplot(train, ax=ax[0])
    #
    # # QQ-plot plot from stats
    # stats.probplot(train, plot=ax[1])
    # plt.show()

    # import numpy as np
    # from scipy import stats
    # import matplotlib.pyplot as plt

    # nsample = 100
    # np.random.seed(7654321)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x = stats.t.rvs(3, size=nsample)
    # res = stats.probplot(x, plot=plt)
    #
    # ax.get_lines()[0].set_marker('p')
    # ax.get_lines()[0].set_markerfacecolor('r')
    # ax.get_lines()[0].set_markersize(12.0)
    # ax.get_lines()[1].set_linewidth(12.0)

    # plt.show()

    # data = np.random.normal(0, 1, 1000)

    # _, bins, _ = plt.hist(data, 20, density=1, alpha=0.5)
    # from scipy.stats import norm
    #
    # data = norm.rvs(10.0, 2.5, size=500)
    # mu, std = norm.fit(data)
    # plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)
    #
    # plt.show()

    # from unidip import UniDip
    # # create bi-modal distribution
    # # dat = np.concatenate([np.random.randn(200) - 3, np.random.randn(200) + 3])
    # dat = np.random.randn(200) + 3
    # # sort data so returned indices are meaningful
    # # dat = np.msort(dat)
    # # get start and stop indices of peaks
    # intervals = UniDip(dat).run()
    # ic(len(intervals))



